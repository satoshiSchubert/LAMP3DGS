import math
import torch
t = torch
import numpy as np
from torch import nn

from model import gaussian_model
from model import pixel_generator

from model.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

import diff_gaussian_rasterization
import diff_gaussian_rasterization_extend


# Render example
# from original code; is not used
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, device='cuda:0'):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fovx * 0.5)
    tanfovy = math.tan(viewpoint_camera.fovy * 0.5)

    raster_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = diff_gaussian_rasterization.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    #if radii is None or radii.shape != pc.get_xyz.shape:
    #    radii = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device)
    #    print('Radii Error!')
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    
# extend channels in kernel function
def render_multichannel(
        viewpoint_camera,
        pc : GaussianModel,
        pipe,
        bg_color : torch.Tensor,
        scaling_modifier = 1.0,
        override_color = None, device='cuda:0'):
    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fovx * 0.5)
    tanfovy = math.tan(viewpoint_camera.fovy * 0.5)

    raster_settings = diff_gaussian_rasterization_extend.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.height),
        image_width=int(viewpoint_camera.width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = diff_gaussian_rasterization_extend.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    #shs = None
    #colors_precomp = None
    #if override_color is None:
    #    if pipe.convert_SHs_python:
    #        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #    else:
    #        shs = pc.get_features
    #else:
    #    colors_precomp = override_color
    
    shs = pc.get_features
    poses = pc.get_xyz
    
    tempShape = (poses.shape[0], 4) # rgba + ? features
    features = pc.get_extra

    input_ts = torch.cat([
        torch.zeros(
            tempShape,
            dtype=poses.dtype,
            layout=poses.layout,
            device=poses.device
        ),
        features,
        poses],
        dim=-1)

    # input_ts = torch.cat([
    #     torch.zeros(
    #         tempShape,
    #         dtype=poses.dtype,
    #         layout=poses.layout,
    #         device=poses.device
    #     ),
    #     poses],
    #     dim=-1)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    #if radii is None or radii.shape != pc.get_xyz.shape:
    #    radii = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device)
    #    print('Radii Error!')
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}