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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from gaussian_splatting.utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from gaussian_splatting.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_splatting.scene.embedding import Embedding
import math

from scipy.spatial import KDTree
import torch.nn.functional as F


def generate_mask(depth):
    """
    Generates a mask tensor of the same shape as the input depth tensor, 
    where 1/64 of the elements are randomly set to True and the rest are set to False.

    Args:
        depth (torch.Tensor([h, w])): A 2D tensor representing the depth values.

    Returns:
        torch.Tensor: A boolean mask tensor of the same shape as the input, 
                      with 1/64 of the elements set to True and the rest set to False.
    """
    h, w = depth.shape
    num_points = h * w
    num_true = num_points // 64

    # Create a mask with all False values
    mask = torch.zeros(h, w, dtype=torch.bool)

    # Randomly select num_true positions to set to True
    indices = torch.randperm(num_points)[:num_true]
    mask.view(-1)[indices] = True

    return mask

def generate_mask_grad(depth):
    """
    Generates a mask tensor of the same shape as the input depth tensor,
    where 1/64 of the elements are set to True based on the highest gradients,
    with non-maximum suppression applied to ensure only one maximum per 3x3 region.

    Args:
        depth (torch.Tensor): A 2D tensor representing the depth values.

    Returns:
        torch.Tensor: A boolean mask tensor of the same shape as the input,
                      with 1/64 of the elements set to True based on the highest gradients.
    """
    h, w = depth.shape
    num_points = h * w
    num_true = num_points // 64

    # Compute gradients
    depth = depth.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    grad_x = F.conv2d(depth, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32), padding=1)
    grad_y = F.conv2d(depth, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32), padding=1)
    grad = torch.sqrt(grad_x**2 + grad_y**2).squeeze()

    # Apply non-maximum suppression
    max_pool = F.max_pool2d(grad.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
    nms_grad = grad * (grad == max_pool.squeeze())

    # Flatten and sort by gradient magnitude
    flat_indices = torch.argsort(nms_grad.view(-1), descending=True)
    selected_indices = flat_indices[:num_true]

    # Create mask
    mask = torch.zeros(h, w, dtype=torch.bool)
    mask.view(-1)[selected_indices] = True

    return mask


class AnchorKDTree:
    def __init__(self) -> None:
        self._data = None
        self._tree = None
    
    def tree_shape(self):
        if self._data is not None:
            print('shape of the KDTree data:', self._data.shape)
        else:
            print('KDTree is not created yet.')
        
    def set_kdtree(self, x: torch.Tensor):
        self._data = x
        self._tree = KDTree(x.detach().cpu().numpy())
    
    def kdtree_detect(self, x: torch.Tensor):
        assert self._tree is not None, 'KDTree is not created yet!'
        
        distances, _ = self._tree.query(x.detach().cpu().numpy())
        result = torch.tensor(distances == 0)
        return result

    def kdtree_add(self, x: torch.Tensor):
        if self._tree is None:
            self.set_kdtree(x)
            return
        
        combined_data = torch.cat((self._data, x), dim=0)
        self.set_kdtree(combined_data)

class AnchorDict:
    '''
        Performance Metrics (Scale set to 10,000,000):
        
        - Time taken to create hash table: 0.0008 seconds
        - Time taken for hash detection: 0.0470 seconds 
        - (for 10,000,000 size of the hash table & 1,000,000 query)
        
        - Time taken by hash_table_add: 0.0138 seconds 
        - (for adding 10,000,000 size of a hash table to the 10,000,000 size of hash table)
        - Size of the hash table after removing duplicates: torch.Size([17,022,822])

        - Time taken by hash_table_add: 0.0037 seconds 
        - (for adding 4,000,000 size of a hash table to the 10,000,000 size of hash table)
        - Size of the hash table after removing duplicates: torch.Size([13,278,442])

        Note: All tests were performed on single RTX 4090.
    '''
    
    MULTIPLIER = torch.tensor([1, 2654435761, 805459861]) 
    # MULTIPLIER = torch.tensor([73856093, 19349663, 83492791])
    INCREMENT = 1
    MODULUS = 2**63 - 1 # data type long is 64-bit

    def __init__(self) -> None:
        self._hash_table = None
    
    def table_shape(self):
        print('shape of the hash table:', self._hash_table.shape)
        
    def _hash_func(self, in_tensor):
        """
        Returns hash tensor using method described in Instant-NGP
        https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/encodings.py#L401

        Args:
            in_tensor: Tensor to be hashed
        """

        in_tensor = in_tensor * self.MULTIPLIER.to(in_tensor.device)
        in_tensor = in_tensor.long()
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.MODULUS
        x += self.INCREMENT

        return x
    
    def set_hash_table(self, x:torch.tensor):
        # assert self._hash_table is None, 'A hash table has been created!'

        hashed_tensor = self._hash_func(x)
        self._hash_table = torch.unique(hashed_tensor)
    
    def hash_detect(self, x:torch.tensor):
        if self._hash_table is None:
            # self.set_hash_table(x)
            return torch.zeros((x.shape[0]), device=x.device, dtype=torch.bool)
            
        # assert self._hash_table is not None, 'Hash table is not existed!'

        x = self._hash_func(x)

        result = torch.isin(x, self._hash_table)
        return result

    def hash_table_add(self, x: torch.Tensor):
        if self._hash_table is None:
            self.set_hash_table(x)
            return
        
        x = self._hash_func(x)
        
        combined_tensor = torch.cat((self._hash_table, x))
        self._hash_table = torch.unique(combined_tensor)


class GaussiansOptParams:
    def __init__(self) -> None:
        self.iterations = 50       # optimization per frame
        self.position_lr_init = 0.0016
        self.position_lr_final = 0.000016
        self.position_lr_delay_mult = 0.0001
        self.position_lr_max_steps = 30_00000
        self.feature_lr = 0.025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.15
        self.lambda_dssim = 0.2
        self.densification_interval = 75
        self.opacity_reset_interval = 300
        self.densify_from_iter = 5
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.002
        self.random_background = False

        self.slide_window = 12
        self.upsample_scale = 3.16 # this parameter is constraint in [1, 8]
        self.saving_frame = 1
        self.frame_computed = 0

        # Scaffold-GS parameters
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 10 # 1500
        self.update_interval = 100
        self.update_until = 1500_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002


class GaussianModel:

    # Serilised function
    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.anchor_dict = {}

        for i in range(len(self.voxel_size_lis)):
            self.anchor_dict[i] = AnchorDict()
            # self.anchor_dict[i] = AnchorKDTree()


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size_lis: dict = None, # float=0.01,
                 distance_lis: dict = None,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 intrinsics = None,
                 config = None
                 ):
        
        # intrinsics: torch.tensor([fx, fy, cx, cy], dtype=torch.float)
        self.intrinsics = intrinsics
        self.config = config
        self.opt = GaussiansOptParams()

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size_lis = voxel_size_lis
        self.distance_lis = distance_lis
        self.max_level = len(voxel_size_lis) - 1
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.level_dim = 1 
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0).cuda()
        self._anchor_index = torch.empty(0).cuda()
        self._offset = torch.empty(0).cuda()
        self._anchor_feat = torch.empty(0).cuda()
        
        self.opacity_accum = torch.empty(0).cuda()

        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self._level = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        
        self.offset_gradient_accum = torch.empty(0).cuda()
        self.offset_denom = torch.empty(0).cuda()

        self.anchor_demon = torch.empty(0).cuda()

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim+self.level_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim+self.level_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.level_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()



    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(self.opt)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor
    @property
    def get_anchor_index(self):
        return self._anchor_index

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_level(self):
        return self._level

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        if isinstance(data, np.ndarray):
            np.random.shuffle(data)
            data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        elif isinstance(data, torch.Tensor):
            # indices = torch.randperm(data.size(0))
            # data = data[indices]
            data = torch.unique(torch.round(data / voxel_size), dim=0) * voxel_size
        else:
            raise TypeError("Unsupported data type. Please provide a numpy array or a torch tensor.")
        return data
    
    def pcd_from_depth(self, depth, w2c, mask=None, rgb=None,
                    compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
        '''
            get the point clouds from depth map, the fun is used
                before generating the voxels.

            code is adapted from SplaTAM.

            input:
                depth:  torch.tensor([h, w], device = 'cuda')
                
                w2c:    torch.tensor([4, 4])
                mask:   torch.tensor([h, w])
            return:
                pcd:    torch.tensor([n, 3], device = 'cuda')
        '''
        

        width, height = depth.shape[1], depth.shape[0]
        FX, FY, CX, CY = self.intrinsics

        x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                        torch.arange(height).cuda().float(),
                                        indexing='xy')
        xx = (x_grid - CX)/FX
        yy = (y_grid - CY)/FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth.reshape(-1)

        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
        

        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                scale_gaussian = depth_z / ((FX + FY)/2)
                mean3_sq_dist = scale_gaussian**2
            else:
                raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
        
        if mask is not None:
            pts = pts[mask.reshape(-1)]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return pts, mean3_sq_dist
        else:
            # If an RGB image is provided, extract corresponding colors
            if rgb is not None:
                # Adjust RGB shape if necessary (from (3, h, w) to (h, w, 3))
                rgb = rgb.permute(1, 2, 0)  # Change from (3, h, w) to (h, w, 3)
                rgb = rgb.reshape(-1, 3)
                if mask is not None:
                    rgb = rgb[mask.reshape(-1)]
                return pts, rgb
            else:
                return pts

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def create_from_pcd(self, pcd, cam_pos, anchor_index:int):

        points = pcd
        self.set_appearance(15120)
        
        for i in range(self.max_level+1):

            if self.voxel_size_lis[i] <= 0:
                init_points = torch.tensor(points).float().cuda()
                init_dist = distCUDA2(init_points).float().cuda()
                median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
                self.voxel_size_lis[i] = median_dist.item()
                del init_dist
                del init_points
                torch.cuda.empty_cache()

            fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size_lis[i])

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if i == 0:
                fused_point_cloud = fused_point_cloud[point_dist < self.distance_lis[0]]
            elif i == self.max_level:
                fused_point_cloud = fused_point_cloud[point_dist >= self.distance_lis[i-1]]
            else:
                fused_point_cloud = fused_point_cloud[(self.distance_lis[i-1] <= point_dist) & (point_dist < self.distance_lis[i])]

            if fused_point_cloud.shape[0] > 0:
                offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
                anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
                
                dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
                scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
                
                rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots[:, 0] = 1

                opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

                level_anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
                level_offset = nn.Parameter(offsets.requires_grad_(True))
                level_anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
                level_scaling = nn.Parameter(scales.requires_grad_(True))
                level_rotation = nn.Parameter(rots.requires_grad_(False))
                level_opacity = nn.Parameter(opacities.requires_grad_(False))
                level_level = torch.full((fused_point_cloud.shape[0],), i, device='cuda', dtype=torch.int)
                
                self._anchor = torch.cat([self._anchor.detach(), level_anchor.requires_grad_(True)], dim=0)
                self._offset = torch.cat([self._offset.detach(), level_offset.requires_grad_(True)], dim=0)
                self._anchor_feat = torch.cat([self._anchor_feat.detach(), level_anchor_feat.requires_grad_(True)], dim=0)
                self._scaling = torch.cat([self._scaling.detach(), level_scaling.requires_grad_(True)], dim=0)
                self._rotation = torch.cat([self._rotation.detach(), level_rotation.requires_grad_(False)], dim=0)
                self._opacity = torch.cat([self._opacity.detach(), level_opacity.requires_grad_(False)], dim=0)
                self._level = torch.cat([self._level.detach(), level_level.requires_grad_(False)], dim=0)

                self._anchor = nn.Parameter(self._anchor.requires_grad_(True))
                self._offset = nn.Parameter(self._offset.requires_grad_(True))
                self._anchor_feat = nn.Parameter(self._anchor_feat.requires_grad_(True))
                self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
                self._rotation = nn.Parameter(self._rotation.requires_grad_(False))
                self._opacity = nn.Parameter(self._opacity.requires_grad_(False))

                self.anchor_dict[i].set_hash_table(fused_point_cloud)
                # self.anchor_dict[i].set_kdtree(fused_point_cloud)
    
                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
    
        self.training_setup(self.opt)

    def add_pcd_level(self, point_voxelised, cam_pos, anchor_index:int):


        for level in range(self.max_level+1):
            fused_point_cloud = point_voxelised[level]
            result = self.anchor_dict[level].hash_detect(fused_point_cloud)
            # result = self.anchor_dict[i].kdtree_detect(fused_point_cloud)
            fused_point_cloud = fused_point_cloud[~result]

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if level == 0:
                fused_point_cloud = fused_point_cloud[point_dist < self.distance_lis[0] * 1.05]
            elif level == self.max_level:
                fused_point_cloud = fused_point_cloud[point_dist >= self.distance_lis[level-1]]
            else:
                fused_point_cloud = fused_point_cloud[(self.distance_lis[level-1] <= point_dist * 1.05) & (point_dist < self.distance_lis[level] * 1.05)]

            if fused_point_cloud.shape[0] > 0:
                new_scaling = torch.ones_like(fused_point_cloud).repeat([1,2]).float().cuda()*self.voxel_size_lis[level] # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([fused_point_cloud.shape[0], 4], device=fused_point_cloud.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

                new_offsets = torch.zeros_like(fused_point_cloud).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": fused_point_cloud,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                
                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                add_level = torch.full((fused_point_cloud.shape[0],), level, device='cuda', dtype=torch.int)
                self._level = torch.cat([self._level.detach(), add_level.requires_grad_(False)], dim=0)
                
                self.anchor_dict[level].hash_table_add(fused_point_cloud)
                # self.anchor_dict[level].kdtree_add(fused_point_cloud)

                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

    def add_pcd(self, pcd, cam_pos, anchor_index:int):
        points = pcd

        for i in range(self.max_level+1):
            fused_point_cloud = self.voxelize_sample(points, voxel_size=self.voxel_size_lis[i])

            result = self.anchor_dict[i].hash_detect(fused_point_cloud)
            # result = self.anchor_dict[i].kdtree_detect(fused_point_cloud)
            fused_point_cloud = fused_point_cloud[~result]

            point_dist = torch.norm(fused_point_cloud - cam_pos, dim=-1)
            if i == 0:
                fused_point_cloud = fused_point_cloud[point_dist < self.distance_lis[0] * 1.05]
            elif i == self.max_level:
                fused_point_cloud = fused_point_cloud[point_dist >= self.distance_lis[i-1]]
            else:
                fused_point_cloud = fused_point_cloud[(self.distance_lis[i-1] <= point_dist * 1.05) & (point_dist < self.distance_lis[i] * 1.05)]

            if fused_point_cloud.shape[0] > 0:
                new_scaling = torch.ones_like(fused_point_cloud).repeat([1,2]).float().cuda()*self.voxel_size_lis[i] # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([fused_point_cloud.shape[0], 4], device=fused_point_cloud.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

                new_offsets = torch.zeros_like(fused_point_cloud).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": fused_point_cloud,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                add_level = torch.full((fused_point_cloud.shape[0],), i, device='cuda', dtype=torch.int)
                self._level = torch.cat([self._level.detach(), add_level.requires_grad_(False)], dim=0)
                
                self.anchor_dict[i].hash_table_add(fused_point_cloud)
                # self.anchor_dict[i].kdtree_add(fused_point_cloud)
                
                add_anchor_index = torch.full((fused_point_cloud.shape[0],), anchor_index, dtype=torch.int).cuda()
                self._anchor_index = torch.cat([self._anchor_index, add_anchor_index])

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)


    def extend_gaussian(self, camera, depthmap, anchor_index:int, rgb = None, init = False):
        '''
            get the keyframe and extend gaussians

        '''
        T_w2c = torch.eye(4, device=camera.R.device)
        T_w2c[0:3, 0:3] = camera.R
        T_w2c[0:3, 3] = camera.T
        # T_w2c[2, 3] = -T_w2c[2, 3]
        cam_center = camera.camera_center
        # cam_center[2] = -cam_center[2]
        point_cloud, color = self.pcd_from_depth(depthmap, T_w2c, rgb=rgb, mask=generate_mask(depthmap))


        if init:
            self.create_from_pcd(point_cloud, cam_center, anchor_index)
        else:
            self.add_pcd(point_cloud, cam_center, anchor_index)


    
    def update_anchor_loop(self, update_point_cloud):
        updated_anchor = self._anchor.clone().detach()

        for level in range(self.max_level + 1):
            voxel_size = self.voxel_size_lis[level]
            level_mask = (self._level == level)
            level_indices = torch.nonzero(level_mask, as_tuple=False).squeeze(-1)

            if level_indices.numel() == 0:
                continue

            level_points = update_point_cloud[level_indices]

            voxelized_points = torch.round(level_points / voxel_size) * voxel_size

            if voxelized_points.shape[0] == 0:
                continue

            min_len = min(len(level_indices), len(voxelized_points))
            selected_indices = level_indices[:min_len]
            selected_voxelized = voxelized_points[:min_len]

            updated_anchor[selected_indices] = selected_voxelized

            self.anchor_dict[level].set_hash_table(selected_voxelized)
        
        self._anchor = nn.Parameter(updated_anchor.requires_grad_(True))

        self.replace_tensor_to_optimizer(self._anchor, 'anchor')



    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
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


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
    
    @torch.no_grad()
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._level = self._level[valid_points_mask]

        for level in range(self.max_level+1):
            self.anchor_dict[level].set_hash_table(self._anchor[self._level == level])

    def anchor_prune_outside(self, point_voxelised, cam_poses):
        new_anchor_dict = {}
        for i in range(self.max_level+1):
            new_anchor_dict[i] = AnchorDict()
            new_anchor_dict[i].set_hash_table(point_voxelised[i])
        
        mask = torch.zeros((self.get_anchor.shape[0]), dtype = torch.bool, device = 'cuda')
        for level in range(self.max_level+1):
            level_mask = self._level == level
            hash_mask = new_anchor_dict[level].hash_detect(self.get_anchor)
            for cam_pos in cam_poses:
                point_dist = torch.norm(self.get_anchor - cam_pos, dim=-1)
                if level == 0:
                    m = point_dist < self.distance_lis[0] * 1.05
                elif level == self.max_level:
                    m = point_dist >= self.distance_lis[level-1]
                else:
                    m = (self.distance_lis[level-1] <= point_dist * 1.05) 
                    m.bitwise_and_(point_dist < self.distance_lis[level] * 1.05)

                mask.bitwise_or_(m)
            mask.bitwise_and_(level_mask)
            mask.bitwise_and_(hash_mask)


        self.prune_anchor(~mask)
        self.opacity_accum = self.opacity_accum[mask]
        self.anchor_demon = self.anchor_demon[mask]
        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size_lis[0]*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                
                new_levels = torch.full((candidate_anchor.shape[0],), 0, device='cuda', dtype=torch.int)

                self._level = torch.cat([self._level.detach(), new_levels.requires_grad_(False)], dim=0)
                


                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }


                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]

                self.anchor_dict[0].hash_table_add(candidate_anchor)
                
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(input=grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
