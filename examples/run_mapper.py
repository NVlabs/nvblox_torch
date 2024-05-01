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
import argparse

from torch.utils.data.dataloader import DataLoader

import demo_utils as demo_utils
from timer import Timer
from nvblox_torch.datasets.rtmv_dataset import RtmvDataset
from nvblox_torch.datasets.sun3d_dataset import Sun3dDataset
from nvblox_torch.mapper import (
    Mapper,
)  # create_nvblox_mapper_instance, get_nvblox_mapper_class

"""
This script implements similar functionality to fuse_3dmatch.cpp of nvblox, except through Python.
"""
from demo_utils import (
    global_pose,
    init_segmentation,
    query_sdf_on_a_plane_grid,
    show_inputs,
    show_renders,
    show_voxels,
    split_depth_based_on_segmentation,
    split_depth_from_gt_segmentation,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="nvblox_torch_example_data/sun3d-mit_76_studyroom-76-1studyroom2",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="sun3d",
        choices=["sun3d", "rtmv", "realsense", "mesh"],
    )
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--mesh-num-frames", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--update-mesh-every-n", type=int, default=-1)
    parser.add_argument("--update-esdf-every-n", type=int, default=2)
    parser.add_argument("--update-color-every-n", type=int, default=-1)
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument(
        "--integrator-type", type=str, default="tsdf", choices=["tsdf", "occupancy"]
    )
    parser.add_argument("--decay-occupancy-every-n", type=int, default=-1)
    parser.add_argument("--clear-map-every-n", type=int, default=-1)
    parser.add_argument("--dynamic-class", type=str, default=None)
    parser.add_argument("--visualize-voxels", action="store_true")
    parser.add_argument("--visualize-esdf-slice", action="store_true")
    args = parser.parse_args()

    # Initialize NvBlox with 2 mappers: mapper 0 uses tsdf layer, mapper 1 uses occupancy
    mapper = Mapper([args.voxel_size, args.voxel_size], ["tsdf", "occupancy"])
    # Pick which mapper is used based on desired integrator type
    if args.integrator_type == "tsdf":
        mapper_id = 0
    else:
        mapper_id = 1

    # Pick which dataset to use based on the args
    if args.dataset_format == "sun3d":
        dataloader = DataLoader(
            Sun3dDataset(root=args.dataset), batch_size=1, shuffle=False, num_workers=0
        )
    elif args.dataset_format == "rtmv":
        dataloader = DataLoader(
            RtmvDataset(root=args.dataset), batch_size=1, shuffle=False, num_workers=0
        )
    elif args.dataset_format == "realsense":
        from realsense_utils import RealsenseDataloader

        dataloader = RealsenseDataloader()
    elif args.dataset_format == "mesh":
        from nvblox_torch.datasets.mesh_dataset import MeshDataset

        dataloader = DataLoader(
            MeshDataset(
                mesh_file=args.dataset,
                n_frames=args.mesh_num_frames,
                image_size=800,
                save_data_dir="mesh_dataset_tmp",
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    else:
        raise ValueError(f"Unknown dataset format: {args.dataset_format}")
    print(f"Dataset has {len(dataloader)} frames")

    timer = Timer()

    if args.dynamic_class:
        seg_model, preprocess, class_to_idx = init_segmentation()
        # TODO: Switch to mapper_id = -1 when it is supported by everything
        mapper_id = 0
        # mapper_id = -1

    if args.visualize_voxels:
        from voxel_visualizer import SimpleVoxelVisualizer

        visualizer = SimpleVoxelVisualizer(args.voxel_size)

    timer.start("data")

    for frame_num, frame in enumerate(dataloader):
        print(f"Frame {frame_num} / {len(dataloader)}")

        rgba, depth, pose, intrinsics = demo_utils.prep_inputs(
            frame["rgba"], frame["depth"], frame["pose"], frame["intrinsics"]
        )

        timer.end("data")

        if (frame_num + 1) % args.clear_map_every_n == 0 and args.clear_map_every_n > 0:
            timer.start("clear")
            mapper.clear()
            timer.end("clear")

        # Predict dynamic class with a torchvision model
        if args.dynamic_class:
            timer.start("segmentation")
            (
                static_depth,
                dynamic_depth,
                dynamic_mask,
            ) = split_depth_based_on_segmentation(
                seg_model,
                preprocess,
                class_to_idx,
                args.dynamic_class,
                depth,
                rgba,
                cv2_show=True,
            )
            timer.end("segmentation")
            timer.start("update")
            mapper.add_depth_frame(static_depth, pose, intrinsics, mapper_id=0)
            mapper.add_depth_frame(dynamic_depth, pose, intrinsics, mapper_id=1)
            if (frame_num + 1) % args.update_color_every_n == 0 and args.update_color_every_n > 0:
                # Not sure if this rgba splitting is needed
                # static_rgba, dynamic_rgba = rgba.clone(), rgba.clone()
                # static_rgba[dynamic_mask][:, 3] = 0
                # dynamic_rgba[~dynamic_mask][:, 3] = 0
                mapper.add_color_frame(rgba, pose, intrinsics, mapper_id=0)
                mapper.add_color_frame(rgba, pose, intrinsics, mapper_id=1)

        # Ground truth segmentation provided for dynamic objects
        elif "seg" in frame and frame["seg"] is not None:
            seg = frame["seg"][0].to(depth.device)
            static_depth, dynamic_depth = split_depth_from_gt_segmentation(depth, seg)
            timer.start("update")
            mapper.add_depth_frame(static_depth, pose, intrinsics, mapper_id=0)
            mapper.add_depth_frame(dynamic_depth, pose, intrinsics, mapper_id=1)

        # No dynamic object segmentation. Use one of the mappers
        else:
            timer.start("update")
            mapper.add_depth_frame(depth, pose, intrinsics, mapper_id)
            if (frame_num + 1) % args.update_color_every_n == 0 and args.update_color_every_n > 0:
                mapper.add_color_frame(rgba, pose, intrinsics, mapper_id)

        # Update ESDF on both mappers
        if (frame_num + 1) % args.update_esdf_every_n == 0 and args.update_esdf_every_n > 0:
            mapper.update_esdf(-1)
        # UpdateMesh no longer works. Wait for PyTorch with C++11 ABI
        if (frame_num + 1) % args.update_mesh_every_n == 0 and args.update_mesh_every_n > 0:
            mapper.update_mesh(-1)
        if (frame_num + 1) % args.decay_occupancy_every_n == 0 and args.decay_occupancy_every_n > 0:
            # Occupancy decay only applies to the occupancy mapper with id=1
            mapper.decay_occupancy(mapper_id=1)

        timer.end("update")

        if args.render_every_n > 0 and (frame_num + 1) % args.render_every_n == 0:
            show_inputs(rgba, depth)
            # Rendering is only supported by the tsdf mapper
            show_renders(mapper, pose, "current_pose", timer, mapper_id=0)
            show_renders(mapper, global_pose(), "global_pose", timer, mapper_id=0)

        if args.max_frames > 0 and frame_num >= args.max_frames - 1:
            break

        if args.visualize_voxels:
            show_voxels(visualizer, mapper)

        if args.visualize_esdf_slice:
            query_sdf_on_a_plane_grid("combined_sdf", mapper, timer, mapper_id=-1)
            query_sdf_on_a_plane_grid("static_sdf", mapper, timer, mapper_id=0)
            query_sdf_on_a_plane_grid("dynamic_sdf", mapper, timer, mapper_id=1)

        timer.print()
        timer.start("data")

    timer.end("data")

    if args.visualize_voxels:
        visualizer.end()

    timer.print()
    if args.update_mesh_every_n > 0:
        #    mesh doesn't work for now:
        mapper.save_mesh("mesh_from_torch.ply")
