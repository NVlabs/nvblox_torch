#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import cv2
import numpy as np
import torch
from transforms3d import affines, euler


def prep_inputs(rgba, depth, pose, intrinsics):
    if len(rgba.shape) == 4:
        assert rgba.shape[0] == 1
        rgba = rgba[0]
        depth = depth[0]
        pose = pose[0]
        intrinsics = intrinsics[0]

    rgba = rgba.cuda(0)
    depth = depth.cuda(0)
    pose = pose.cuda(0)
    intrinsics = intrinsics  # this stays on CPU

    # We need color images in HxWx4 format for nvblox
    rgba = rgba.permute((1, 2, 0)).contiguous()

    # Check current limitations
    assert rgba.dtype == torch.uint8, "Only 8-bit RGB images supported"
    assert rgba.shape[2] == 4, "Only 4-channel RGBA images supported by nvblox"
    assert depth.dtype == torch.float, "CPP-side conversions assume 32-bit float tensors"
    assert pose.dtype == torch.float, "CPP-side conversions assume 32-bit float tensors"
    assert intrinsics.dtype == torch.float, "CPP-side conversions assume 32-bit float tensors"

    return rgba, depth, pose, intrinsics


def show_renders(mapper, pose, name, timer, mapper_id):
    timer.start("render")
    depth, color = mapper.render_depth_and_color_image(
        camera_pose=pose,
        intrinsics=torch.tensor([[480, 0, 320], [0, 480, 320], [0, 0, 1]], dtype=torch.float32),
        height=480,
        width=640,
        max_ray_length=10.0,
        max_steps=150,
        mapper_id=mapper_id,
    )
    timer.end("render")

    depth_np = (depth.clamp(0, 10.0) / depth.max()).detach().cpu().numpy()
    color_np = color[:, :, :3].detach().cpu().numpy()
    color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
    cv2.imshow(f"{name}_depth", depth_np)
    cv2.imshow(f"{name}_color", color_np)
    cv2.waitKey(1)


def global_pose(device="cuda"):
    eul = [-2.2, 0.08750253722127824, 1.4509891417591223]
    p = [6, 0, 4]
    # p = [0, 0, 5]
    # eul = [-1.59, 0, 0]
    # eul = [-2.0, 0, 0]
    # p = [1.0, -1.5, 0.5]
    R = euler.euler2mat(*eul)
    T = affines.compose(p, R, [1, 1, 1])
    return torch.Tensor(T).to(device)


def show_voxels(visualizer, mapper, source="esdf"):
    voxel_size = visualizer.voxel_size
    site_size = voxel_size * 1.73
    tensor_args = {"device": "cuda", "dtype": torch.float32}

    low = [-5, -5, -1]
    high = [5, 5, 2]
    range = [h - l for l, h in zip(low, high)]

    x = torch.linspace(low[0], high[0], int(range[0] // voxel_size), **tensor_args)
    y = torch.linspace(low[1], high[1], int(range[1] // voxel_size), **tensor_args)
    z = torch.linspace(low[2], high[2], int(range[2] // voxel_size), **tensor_args)
    w, l, h = x.shape[0], y.shape[0], z.shape[0]
    xyz = torch.stack(torch.meshgrid(x, y, z, indexing="ij")).permute((1, 2, 3, 0)).reshape(-1, 3)

    r = torch.zeros_like(xyz[:, 0:1])  # * voxel_size
    xyzr = torch.cat([xyz, r], dim=1)
    batch_size = xyzr.shape[0]

    if source == "occupancy":
        log_odds = mapper.query_occupancy(xyzr)
        odds = torch.exp(log_odds)
        p = odds / (odds + 1)
        # sdf_and_weights[sdf_and_weights < -99]
        surface_mask = p > 0.5
        surface_coords = xyz[surface_mask]
        if surface_coords.shape[0] == 0:
            return
        coords_np = surface_coords.detach().cpu().numpy()
        surface_p = p[surface_mask]
        colors = (
            (
                torch.stack(
                    [
                        (surface_p.float()),
                        (surface_p.float()),
                        torch.ones_like(surface_p),
                    ],
                    dim=1,
                )
            )
            * (surface_coords[:, 2:3] + 1)
            / 2
        )
        colors_np = colors.detach().cpu().numpy()
        visualizer.add_points(coords_np, colors_np)

    elif source == "tsdf":
        tsdf_and_weights = mapper.query_tsdf(xyzr)
        tsdf = tsdf_and_weights[0]
        weight = tsdf_and_weights[1]
        # sdf_and_weights[sdf_and_weights < -99]
        surface_mask = torch.logical_and(tsdf.abs() < site_size, weight > 0)
        surface_coords = xyz[surface_mask]
        if surface_coords.shape[0] == 0:
            return
        coords_np = surface_coords.detach().cpu().numpy()
        surface_tsdf = tsdf[surface_mask]
        sdf_viz = (surface_tsdf.clamp(-8 * voxel_size, 8 * voxel_size) + 8 * voxel_size) / (
            16 * voxel_size
        )
        colors = (
            (
                torch.stack(
                    [(sdf_viz.float()), (sdf_viz.float()), torch.ones_like(sdf_viz)],
                    dim=1,
                )
            )
            * (surface_coords[:, 2:3] + 1)
            / 2
        )
        colors_np = colors.detach().cpu().numpy()
        visualizer.add_points(coords_np, colors_np)

    elif source == "esdf":
        out_points_static = torch.zeros((batch_size, 4), **tensor_args) + 0.0
        out_points_dynamic = torch.zeros((batch_size, 4), **tensor_args) + 0.0

        sdf_static = mapper.query_sdf(xyzr, out_points_static, True, mapper_id=0)
        sdf_dynamic = mapper.query_sdf(xyzr, out_points_dynamic, True, mapper_id=1)

        df_static = sdf_static.abs()
        df_dynamic = sdf_dynamic.abs()
        surface_mask = torch.minimum(df_static, df_dynamic) < site_size
        surface_coords = xyz[surface_mask]
        if surface_coords.shape[0] == 0:
            return

        surface_df_static = df_static[surface_mask]
        surface_df_dynamic = df_dynamic[surface_mask]

        coords_np = surface_coords.detach().cpu().numpy()

        colors = (
            (
                torch.stack(
                    [
                        0.8 * (surface_df_static < site_size).float(),
                        0.8 * (surface_df_dynamic < site_size).float(),
                        torch.ones_like(surface_df_static),
                    ],
                    dim=1,
                )
            )
            * (surface_coords[:, 2:3] + 1)
            / 3
        )

        colors_np = colors.detach().cpu().numpy()

        visualizer.add_points(coords_np, colors_np)
    else:
        raise ValueError(f"Unknown voxel source: {source}. Supported: esdf, tsdf, occupancy")
    visualizer.show(clear_after=True)


def query_sdf_on_a_plane_grid(name, mapper, timer, mapper_id):
    batch_size = 10
    tensor_args = {"device": "cuda", "dtype": torch.float32}

    x = torch.linspace(-4, 4, 500, **tensor_args)
    y = torch.linspace(-4, 4, 500, **tensor_args)
    xy = torch.stack(torch.meshgrid(x, y, indexing="ij")).permute((1, 2, 0)).reshape(-1, 2)
    z = torch.ones_like(xy[:, 0:1]) * 0.1
    r = torch.zeros_like(xy[:, 0:1]) * 0.1
    xyzr = torch.cat([xy, z, r], dim=1)

    h = x.shape[0]
    w = y.shape[0]
    batch_size = xyzr.shape[0]

    out_points = torch.zeros((batch_size, 4), **tensor_args) + 0.0

    timer.start(f"query_{name}")
    r = mapper.query_sdf(xyzr, out_points, True, mapper_id=mapper_id)
    timer.end(f"query_{name}")
    sdf_grid = r.reshape(h, w)

    sdf_np = (sdf_grid.clamp(-0.5, 0.5) + 0.5).detach().cpu().numpy()
    sdf_np = sdf_np / (sdf_np.max())
    sdf_np = cv2.applyColorMap((sdf_np * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)

    cv2.imshow(f"sdf_grid_{name}", sdf_np)
    cv2.waitKey(1)


def show_inputs(rgba, depth):
    depth_norm = depth - depth.min()
    depth_norm = depth_norm / depth_norm.max()
    cv2.imshow("rgb_input", rgba[:, :, :3].detach().cpu().numpy())
    cv2.imshow("depth_input", depth_norm.detach().cpu().numpy())


def profile_sdf_query(n_spheres, mapper):
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        tensor_args = {"device": "cuda", "dtype": torch.float32}
        xyzr = torch.rand((n_spheres, 4), **tensor_args) * 4
        xyzr[:, 3] = 0.04
        out_points = torch.zeros((n_spheres, 4), **tensor_args) + 0.0
        r = mapper.query_sdf(xyzr, out_points, True, mapper_id=-1)

        xyzr = torch.rand((n_spheres, 4), **tensor_args)
        xyzr[:, 3] = 0.04
        out_points = torch.zeros((n_spheres, 4), **tensor_args) + 0.0
        r = mapper.query_sdf(xyzr, out_points, True, mapper_id=0)

        xyzr = torch.rand((n_spheres, 4), **tensor_args)
        xyzr[:, 3] = 0.04
        out_points = torch.zeros((n_spheres, 4), **tensor_args) + 0.0
        r = mapper.query_sdf(xyzr, out_points, True, mapper_id=1)

    print("Exporting the trace..")
    prof.export_chrome_trace("sdf_query.json")
    exit()


def split_depth_from_gt_segmentation(depth, dynamic_mask):
    static_mask = ~dynamic_mask

    dynamic_depth = depth.clone()
    static_depth = depth.clone()
    dynamic_depth[static_mask] = 0.0
    static_depth[dynamic_mask] = 0.0
    return static_depth, dynamic_depth


def split_depth_based_on_segmentation(
    seg_model,
    preprocess,
    class_to_idx,
    dynamic_class_name,
    depth,
    rgba,
    prob_threshold=0.5,
    cv2_show=False,
):
    input_tensor = preprocess(rgba[:, :, :3].permute((2, 0, 1))).unsqueeze(0)
    output = seg_model(input_tensor)["out"]
    normalized_masks = output.softmax(dim=1)
    # reduce the background weight because it is too greedy otherwise
    normalized_masks[:, class_to_idx["__background__"]] *= 0.2
    max_mask = normalized_masks.argmax(dim=1)[0] == class_to_idx[dynamic_class_name]
    dynamic_threshold = 1e-2
    t_mask = normalized_masks[0, class_to_idx[dynamic_class_name]] > dynamic_threshold
    mask = torch.logical_or(max_mask, t_mask).float()
    # resize back to same size as input
    mask = torch.nn.functional.interpolate(
        mask[None, None, :, :], size=depth.shape, mode="bilinear", align_corners=True
    )[0, 0]

    dynamic_mask = mask > prob_threshold
    static_mask = ~dynamic_mask

    dynamic_depth = depth.clone()
    static_depth = depth.clone()
    dynamic_depth[static_mask] = 0.0
    static_depth[dynamic_mask] = 0.0

    if cv2_show:
        cv2.imshow("rgb", rgba[:, :, :3].detach().cpu().numpy())
        cv2.imshow("dynamic_mask", mask.float().detach().cpu().numpy())
        cv2.waitKey(1)

    return static_depth, dynamic_depth, dynamic_mask


def init_segmentation():
    import torchvision.transforms as T
    from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50

    weights = FCN_ResNet50_Weights.DEFAULT
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    print(class_to_idx)
    seg_model = fcn_resnet50(weights=weights)
    seg_model.cuda().eval()
    preprocess = (
        weights.transforms()
    )  # T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return seg_model, preprocess, class_to_idx
