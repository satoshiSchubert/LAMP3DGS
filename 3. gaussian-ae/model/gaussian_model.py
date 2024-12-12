#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

extraFeatureChannelCount = 32

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation, self.device)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance, self.device)
            return symm
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
    def __init__(self, device, max_sh_degree=3):
        self.active_sh_degree = 0
        self.max_sh_degree = max_sh_degree  
        self.xyz = torch.empty(0)
        self.features_dc = torch.empty(0)
        self.features_rest = torch.empty(0)
        self.scaling = torch.empty(0)
        self.rotation = torch.empty(0)
        self.opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.extraFeature = torch.empty(0)

        self.setup_functions()
        self.device = device
        
    def capture(self):
        return (
            self.active_sh_degree,
            self.xyz,
            self.features_dc,
            self.features_rest,
            self.extraFeature,
            self.scaling,
            self.rotation,
            self.opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    def restore(self, model_args, training_args, aeModel, geoModel):
        (self.active_sh_degree, 
        self.xyz, 
        self.features_dc, 
        self.features_rest,
        self.extraFeature,
        self.scaling, 
        self.rotation, 
        self.opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args, aeModel, geoModel)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.features_dc.shape[1]*self.features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.features_rest.shape[1]*self.features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(extraFeatureChannelCount):
            l.append('extraFeature_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # properties
    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)
    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)
    @property
    def get_xyz(self):
        return self.xyz
    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)
    @property
    def get_extra(self):
        return self.extraFeature
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation)
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def loadPly(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"]).reshape((-1,1))

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        extraFeatureNames = [p.name for p in plydata.elements[0].properties if p.name.startswith('extraFeature_')]
        #extraFeature = np.random.randn(xyz.shape[0], extraFeatureChannelCount)
        extraFeature = np.zeros((xyz.shape[0], extraFeatureChannelCount))
        if len(extraFeatureNames) == extraFeatureChannelCount:
            extraFeatureNames = sorted(extraFeatureNames, key = lambda x : int(x.split('_')[-1]))
            for idx, attr_name in enumerate(extraFeatureNames):
                extraFeature[:, idx] = np.asarray(plydata.elements[0][attr_name])


        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self.features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self.extraFeature = nn.Parameter(torch.tensor(extraFeature, dtype=torch.float, device=self.device).requires_grad_(True))
        self.opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self.scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self.rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        
    def savePly(self, path):
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.zeros(self.xyz.shape[0], dtype=dtype_full)

        xyz = self.xyz.detach().cpu().numpy()
        elements['x'] = xyz[:,0]
        elements['y'] = xyz[:,1]
        elements['z'] = xyz[:,2]
        xyz = None

        dc = self.features_dc.detach().squeeze(1).cpu().numpy()
        elements['f_dc_0'] = dc[:,0]
        elements['f_dc_1'] = dc[:,1]
        elements['f_dc_2'] = dc[:,2]
        dc = None

        rest = self.features_rest.detach().cpu().numpy()
        for i in range(0, 15):
            elements['f_rest_{}'.format(i)] = rest[:,i,0]
        for i in range(15, 30):
            elements['f_rest_{}'.format(i)] = rest[:,i-15,1]
        for i in range(30, 45):
            elements['f_rest_{}'.format(i)] = rest[:,i-30,2]
        rest = None

        extra = self.extraFeature.detach().cpu().numpy()
        for i in range(extraFeatureChannelCount):
            elements['extraFeature_{}'.format(i)] = extra[:, i]
        extra = None

        opa = self.opacity.detach().cpu().numpy().reshape(-1)
        elements['opacity'] = opa[:]
        opa = None

        scale = self.scaling.detach().cpu().numpy().reshape((-1,3))
        elements['scale_0'] = scale[:,0]
        elements['scale_1'] = scale[:,1]
        elements['scale_2'] = scale[:,2]
        scale = None

        rot = self.rotation.detach().cpu().numpy().reshape((-1,4))
        elements['rot_0'] = rot[:,0]
        elements['rot_1'] = rot[:,1]
        elements['rot_2'] = rot[:,2]
        elements['rot_3'] = rot[:,3]
        rot = None
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            if group['name'] == 'PixelGen' or group['name'] == 'GeoLoss':
                continue

            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:

            if group['name'] == 'PixelGen' or group['name'] == 'GeoLoss':
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)

        self.xyz = optimizable_tensors["xyz"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.extraFeature = optimizable_tensors['extra']
        self.opacity = optimizable_tensors["opacity"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            
            # assert len(group["params"]) == 1
            if group['name'] == 'PixelGen' or group['name'] == 'GeoLoss':
                continue

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_extra, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        'extra' : new_extra,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self.xyz = optimizable_tensors["xyz"]
        self.features_dc = optimizable_tensors["f_dc"]
        self.features_rest = optimizable_tensors["f_rest"]
        self.extraFeature = optimizable_tensors['extra']
        self.opacity = optimizable_tensors["opacity"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.rotation[selected_pts_mask], self.device).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.features_rest[selected_pts_mask].repeat(N,1,1)
        new_extra = self.extraFeature[selected_pts_mask].repeat(N, 1)
        new_opacity = self.opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_extra, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self.xyz[selected_pts_mask]
        new_features_dc = self.features_dc[selected_pts_mask]
        new_features_rest = self.features_rest[selected_pts_mask]
        new_extra = self.extraFeature[selected_pts_mask]
        new_opacities = self.opacity[selected_pts_mask]
        new_scaling = self.scaling[selected_pts_mask]
        new_rotation = self.rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_extra, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def training_setup(self, kargs, aeModel : nn.Module, geoModel : nn.Module | None):

        self.percent_dense = kargs['percent_dense']
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        position_lr_init = kargs['position_lr_init']
        feature_lr = kargs['feature_lr']

        self.position_lr_init = position_lr_init
        self.feature_lr = feature_lr

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        l = [
            {'params': [self.xyz], 'lr': kargs.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.features_dc], 'lr': kargs.feature_lr, "name": "f_dc"},
            {'params': [self.features_rest], 'lr': kargs.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self.extraFeature], 'lr':kargs.extra_lr, 'name' : 'extra'},
            {'params': [self.opacity], 'lr': kargs.opacity_lr, "name": "opacity"},
            {'params': [self.scaling], 'lr': kargs.scaling_lr, "name": "scaling"},
            {'params': [self.rotation], 'lr': kargs.rotation_lr, "name": "rotation"}
        ]
        l.append({
            'params' : aeModel.parameters(),
            'lr' : 1e-4,
            'name' : 'PixelGen'
        })
        if geoModel:
            l.append({
                'params' : geoModel.parameters(),
                'lr' : 1e-4,
                'name' : 'GeoLoss'
            })

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=kargs.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=kargs.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=kargs.position_lr_delay_mult,
                                                    max_steps=kargs.position_lr_max_steps)
                                                    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def get_min_axis(self, cam_o):
        pts = self.get_xyz
        p2o = cam_o[None] - pts
        scales = self.get_scaling
        min_axis_id = torch.argmin(scales, dim = -1, keepdim=True)
        min_axis = torch.zeros_like(scales).scatter(1, min_axis_id, 1)

        rot_matrix = build_rotation(self.get_rotation, device=pts.device)
        ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)

        neg_msk = torch.sum(p2o*ndir, dim=-1) < 0
        ndir[neg_msk] = -ndir[neg_msk] # make sure normal orient to camera
        return ndir

