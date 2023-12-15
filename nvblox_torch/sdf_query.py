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
import torch


class SdfSphereBlox(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_spheres,
        c_mapper_instance,
        out_spheres=None,
        compute_closest_point=True,
        mapper_id=-1,
    ):
        if out_spheres is None:
            out_spheres = query_spheres.clone() * 0.0
        if mapper_id >= 0:
            r = c_mapper_instance.query_sdf(
                out_spheres,
                query_spheres,
                query_spheres.shape[0],
                compute_closest_point,
                mapper_id,
            )
        else:
            r = c_mapper_instance.query_multi_sdf(
                out_spheres,
                query_spheres,
                query_spheres.shape[0],
                compute_closest_point,
            )
        r = r[0]
        distance = r[:, 3]
        ctx.save_for_backward(r)
        return distance

    def backward(ctx, grad_output):
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            r[:, :3] = r[:, :3] / r[:, 3:4]
            r[:, 3] = 0.0
            r = torch.nan_to_num(r)
            grad_sph = grad_output.unsqueeze(-1) * r
        return grad_sph, None, None, None


class SdfSphereCostMultiBlox(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sphere_position_rad,
        out_distance,
        out_grad,
        sparsity_idx,
        weight,
        activation_distance,
        c_mapper,
        blox_pose,
        blox_enable,
        return_loss=False,
    ):
        b, n, h, _ = sphere_position_rad.shape
        r = c_mapper.query_sphere_sdf_cost(
            sphere_position_rad,
            out_distance,
            out_grad,
            sparsity_idx,
            weight,
            activation_distance,
            blox_pose,
            blox_enable,
            b,
            h,
            n,
            True,
        )
        distance = r[0]
        ctx.return_loss = return_loss
        ctx.save_for_backward(r[1])
        return distance

    def backward(ctx, grad_output):
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            grad_sph = r
            if ctx.return_loss:
                grad_sph = grad_sph * grad_output
        return grad_sph, None, None, None, None, None, None, None, None, None


class SdfSphereTrajectoryCostMultiBlox(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sphere_position_rad,
        out_distance,
        out_grad,
        sparsity_idx,
        weight,
        activation_distance,
        speed_dt,
        c_mapper,
        blox_pose,
        blox_enable,
        sweep_steps=0,
        enable_speed_metric=False,
        return_loss=False,
    ):
        b, n, h, _ = sphere_position_rad.shape
        r = c_mapper.query_sphere_trajectory_sdf_cost(
            sphere_position_rad,
            out_distance,
            out_grad,
            sparsity_idx,
            weight,
            activation_distance,
            speed_dt,
            blox_pose,
            blox_enable,
            b,
            h,
            n,
            sweep_steps,
            enable_speed_metric,
            sphere_position_rad.requires_grad,
        )
        distance = r[0]
        ctx.return_loss = return_loss
        ctx.save_for_backward(r[1])
        return distance

    def backward(ctx, grad_output):
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            grad_sph = r
            if ctx.return_loss:
                grad_sph = grad_sph * grad_output
        return (
            grad_sph,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
