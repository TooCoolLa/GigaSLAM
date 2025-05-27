import random

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, nearMean_map, image2canny
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose, get_pose
from utils.slam_utils import get_loss_mapping, update_viewpoints_from_poses, update_point_cloud_in_batches

from utils.anchor_utils import anchor_in_frustum
import numpy as np

class BackEnd(mp.Process):
    def __init__(self, config, dataset):
        super().__init__()
        self.dataset = dataset

        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        self.pose_opt = False

        cam_params = {}
        cam_params['fx'] = self.dataset.fx,
        cam_params['fy'] = self.dataset.fy,
        cam_params['cx'] = self.dataset.cx,
        cam_params['cy'] = self.dataset.cy,
        cam_params['width'] = self.dataset.width,
        cam_params['height'] = self.dataset.height,

        self.width = float(cam_params['width'][0])
        self.height = float(cam_params['height'][0])
        self.fx = float(cam_params['fx'][0])
        self.fy = float(cam_params['fy'][0])
        self.cx = float(cam_params['cx'][0])
        self.cy = float(cam_params['cy'][0])

        self.intrinsics = torch.tensor([
                [self.fx,   0,          self.cx ],
                [0,         self.fy,    self.cy ],
                [0,         0,          1       ]
            ], device='cuda')
        
        self.trajectory = 255 + np.zeros((700, 700, 3), dtype=np.uint8)

        self.color_refinement_iter = config['Hierarchical']['color_refinement_iter']
        self.viz = config['SLAM']['viz']

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, rgb = None):
        Log(f'BACKEND: add_next_kf idx: {frame_idx}')
        self.gaussians.extend_gaussian(
            viewpoint, anchor_index = frame_idx,  init=init, depthmap=depth_map, rgb = rgb
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                        intrinsics=self.intrinsics, 
                                        pose=get_pose(viewpoint), 
                                        cam_center=viewpoint.camera_center,
                                        distance_lis = self.gaussians.distance_lis,
                                        levels=self.gaussians.get_level, 
                                        h=self.height, 
                                        w=self.width)
            opt_mask.bitwise_or_(m)
            
            render_pkg = render(
                viewpoint, 
                self.gaussians, 
                self.pipeline_params, 
                self.background, 
                visible_mask=opt_mask
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1, time_return = False):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        
        frustum_time = 0

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)

                opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                            intrinsics=self.intrinsics, 
                                            pose=get_pose(viewpoint), 
                                            cam_center=viewpoint.camera_center,
                                            distance_lis = self.gaussians.distance_lis,
                                            levels=self.gaussians.get_level, 
                                            h=self.height, 
                                            w=self.width)
                opt_mask.bitwise_or_(m)

                
                render_pkg = render(
                    viewpoint, 
                    self.gaussians, 
                    self.pipeline_params, 
                    self.background, 
                    visible_mask=opt_mask
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')

                m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                            intrinsics=self.intrinsics, 
                                            pose=get_pose(viewpoint), 
                                            cam_center=viewpoint.camera_center,
                                            distance_lis = self.gaussians.distance_lis,
                                            levels=self.gaussians.get_level, 
                                            h=self.height, 
                                            w=self.width)
                opt_mask.bitwise_or_(m)
                
                render_pkg = render(
                    viewpoint, 
                    self.gaussians, 
                    self.pipeline_params, 
                    self.background, 
                    visible_mask=opt_mask
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()

            gaussian_split = False
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()


                self.gaussians.optimizer.step()

                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                if self.pose_opt:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        
        if time_return:
            return gaussian_split, frustum_time
        else:
            return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = self.color_refinement_iter * len(self.viewpoints.keys())
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                        intrinsics=self.intrinsics, 
                                        pose=get_pose(viewpoint_cam), 
                                        cam_center=viewpoint_cam.camera_center,
                                        distance_lis = self.gaussians.distance_lis,
                                        levels=self.gaussians.get_level, 
                                        h=self.height, 
                                        w=self.width)
            opt_mask.bitwise_or_(m)
            
            render_pkg = render(
                viewpoint_cam, 
                self.gaussians, 
                self.pipeline_params, 
                self.background, 
                visible_mask=opt_mask
            )
            image, visibility_filter, radii, scaling = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["scaling"],
            )

            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image, gt_image)

            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * ssim_loss + 0.05*scaling_reg

            loss.backward()


            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        import time
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                if self.last_sent >= 10:
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.orig_viewpoint = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True, rgb=viewpoint.original_image
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "loop_update":
                    cur_frame_idx = data[1]
                    poses_old = data[2]
                    poses_update = data[3]
                    kf_indices = data[4]

                    update_viewpoints_from_poses(self.viewpoints, torch.tensor(poses_update))
                    updated_pcd = update_point_cloud_in_batches(self.gaussians.get_anchor,
                                                                poses_old, 
                                                                poses_update, 
                                                                self.gaussians.get_anchor_index,
                                                                kf_indices)
                    self.gaussians.update_anchor_loop(updated_pcd.cuda())



                elif data[0] == "keyframe":
                    begin = time.time()
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, rgb=viewpoint.original_image)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 50
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    pose_window = self.current_window[len(self.current_window) // 2:]

                    if len(pose_window) == 0 or (pose_window == [0]):
                        self.pose_opt = False
                    else:
                        self.pose_opt = True

                    for cam_idx in range(len(pose_window)):
                        if pose_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[pose_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    if self.pose_opt:
                        self.keyframe_optimizers = torch.optim.Adam(opt_params)



                    _, time_return = self.map(self.current_window, iters=iter_per_kf, time_return = True)


                    viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in self.current_window]
                    viewpoint = viewpoint_stack[-1]
                    opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                    m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                                intrinsics=self.intrinsics, 
                                                pose=get_pose(viewpoint), 
                                                cam_center=viewpoint.camera_center,
                                                distance_lis = self.gaussians.distance_lis,
                                                levels=self.gaussians.get_level, 
                                                h=self.height, 
                                                w=self.width)
                    opt_mask.bitwise_or_(m)
                    
                    render_pkg = render(
                        viewpoint, 
                        self.gaussians, 
                        self.pipeline_params, 
                        self.background, 
                        visible_mask=opt_mask
                    )
                    (
                        image,
                        viewspace_point_tensor,
                        visibility_filter,
                        radii,
                        depth,
                        opacity,
                        n_touched,
                    ) = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        render_pkg["n_touched"],
                    )


                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
