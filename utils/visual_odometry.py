import numpy as np
import cv2
# from scale_recovery import find_scale_from_depth

import torch
import kornia
import kornia as K
import kornia.feature as KF

import os
import numpy as np
import cv2
from utils.loop_refinement import make_pypose_Sim3, ransac_umeyama, SE3_to_Sim3, pose_refinement
from utils.scale_recovery import find_scale_from_depth

import numpy as np
from einops import asnumpy, rearrange, repeat

from utils.loop_closure.retrieval import ImageCache, RetrievalDBOW
# import time

from utils.pose_utils import poses_to_c2w_tensor, poses_to_quaternions, quaternions_to_poses, convert_pose_numpy_to_opencv_vectorized, convert_quat_opencv_to_c2w_vectorized
from utils.slam_utils import get_matched_camera_points_vectorized, compute_sim3_open3d

import pypose as pp

import json


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt



class CameraModel(object):
    """
    Class that represents a pin-hole camera model (or projective camera model).
    In the pin-hole camera model, light goes through the camera center (cx, cy) before its projection
    onto the image plane.
    """
    def __init__(self, params):
        """
        Creates a camera model

        Arguments:
            params {dict} -- Camera parameters
        """

        # Image resolution

        self.width = float(params['width'][0])
        self.height = float(params['height'][0])
        # Focal length of camera
        self.fx = float(params['fx'][0])
        self.fy = float(params['fy'][0])
        # Optical center (principal point)
        self.cx = float(params['cx'][0])
        self.cy = float(params['cy'][0])
        
        self.mat = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]])

class ClassicTracking:

    def __init__(self, cam_params, loop_enable, viz):

        self.loop_enable = loop_enable
        self.viz = viz

        if self.loop_enable:
            self.retrieval = RetrievalDBOW()
            self.imcache = ImageCache()
        
        self.cam = CameraModel(cam_params)

        self.K = np.array([
            [cam_params['fx'][0], 0, cam_params['cx'][0]],
            [0, cam_params['fy'][0], cam_params['cy'][0]],
            [0,                   0,                   1]
        ])

        self.prev_frame = None 
        self.feat_ref = None 
        self.feat_curr = None 
        self.detector_method = "FAST" 
        self.matching_method = "LightGlue"  
        self.min_num_features = 2500
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.prev_depth = None
        self.pose = np.zeros((4, 4)) 
        self.w2c = np.zeros((4, 4))

        self.mean_depth_array = np.array([])
        self.flag = True

        # flow status
        self.prev_kf = None
        self.feat_prev_kf = None


        self.kp_frame_id = 0 # frame id so far

        self.detector = KF.DISK.from_pretrained("depth").to("cuda").eval()
        self.lightglue = KF.LightGlue("disk").to("cuda").eval()

        self.lg_matcher = KF.LightGlueMatcher("disk").eval().to("cuda")
        self.disk = KF.DISK.from_pretrained("depth").to("cuda")
        
        self.pose = np.eye(4)  # pose matrix [R | t; 0  1]

        self.pose_ref = np.eye(4)

        self.mean_depth_array = np.array([])

        self.kp_ref = None
        self.kp_curr = None


        n_max = 5000 
        self.pose_history = np.repeat(np.eye(4)[np.newaxis, :, :], n_max, axis=0)  # Initialize
        self.pose_history_old = np.repeat(np.eye(4)[np.newaxis, :, :], n_max, axis=0)  # Initialize

        self.kp_history = []
        self.depth_history = []
        self.frame_history = []
        self.loop_kf_idx = 0


        # Loop Closure Vars
        self.loop_ii = torch.zeros(0, dtype=torch.long)
        self.loop_jj = torch.zeros(0, dtype=torch.long)


    def get_loop_update(self):
        return self.pose_history_old[:self.kp_frame_id, :, :], self.pose_history[:self.kp_frame_id, :, :]

    def set_curr_keyframe(self, frame):

        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_uint8 = frame_normalized.astype(np.uint8)
        frame = cv2.cvtColor(np.transpose(frame_uint8, (1, 2, 0)), cv2.COLOR_RGB2GRAY)

        self.prev_kf = frame
        feat_ref = self.detect_features(frame)
        self.feat_prev_kf = np.array([x.pt for x in feat_ref], dtype=np.float32)

    def motion_flow_keyframe(self, frame, prev_kf = True):

        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_uint8 = frame_normalized.astype(np.uint8)
        frame = cv2.cvtColor(np.transpose(frame_uint8, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        if prev_kf:
            # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids
            kp2, st, err = cv2.calcOpticalFlowPyrLK(self.prev_kf, frame,
                                                    self.feat_prev_kf, None,
                                                    winSize=(21, 21),
                                                    criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            st = st.reshape(st.shape[0])  # status of points from frame to frame
            # Keypoints
            kp1 = self.feat_prev_kf[st == 1]
            kp2 = kp2[st == 1]
        else:
            kp2, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame,
                                                    self.feat_ref, None,
                                                    # maxLevel=6,
                                                    winSize=(21, 21),
                                                    criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            st = st.reshape(st.shape[0])  # status of points from frame to frame
            # Keypoints
            kp1 = self.feat_ref[st == 1]
            kp2 = kp2[st == 1]

        flow_vectors = kp2 - kp1
        flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)

        return np.median(flow_magnitudes)

    def detect_features(self, frame):
        """
        Point-feature detector: search for salient keypoints that are likely to match well in other image frames.
            - corner detectors: Moravec, Forstner, Harris, Shi-Tomasi, and FAST.
            - blob detectors: SIFT, SURF, and CENSURE.

        Args:
            frame {ndarray}: frame to be processed
        """

        if self.detector_method == "FAST":
            detector = cv2.FastFeatureDetector_create()  # threshold=25, nonmaxSuppression=True)
            return detector.detect(frame)

        elif self.detector_method == "ORB":
            detector = cv2.ORB_create(nfeatures=3000)
            kp1, des1 = detector.detectAndCompute(frame, None)
            return kp1

    def feature_matching(self, frame, fliter = False, lc_frame_id = None):
        """
        The feature-matching: looks for corresponding features in other images.

        Args:
            frame {ndarray}: frame to be processed
        """

        if lc_frame_id is not None:
            prev_frame = self.frame_history[lc_frame_id]
            feat_ref = self.kp_history[lc_frame_id]
        else:
            prev_frame = self.prev_frame
            feat_ref = self.feat_ref

        if self.matching_method == "LightGlue":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            img1 = K.utils.image_to_tensor(prev_frame).unsqueeze(0).float() / 255.0
            img2 = K.utils.image_to_tensor(frame).unsqueeze(0).float() / 255.0

            img1 = img1.to(device)
            img2 = img2.to(device)

            num_features = 2048 if lc_frame_id == None else 256

            hw1 = torch.tensor(img1.shape[2:], device=device)
            hw2 = torch.tensor(img2.shape[2:], device=device)


            with torch.inference_mode():
                inp = torch.cat([img1, img2], dim=0)
                features1, features2 = self.disk(inp, num_features, pad_if_not_divisible=True)
                kps1, descs1 = features1.keypoints, features1.descriptors
                kps2, descs2 = features2.keypoints, features2.descriptors

                lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
                lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
                dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)


            kp1 = kps1[idxs[:, 0]].cpu().numpy()  
            kp2 = kps2[idxs[:, 1]].cpu().numpy()  

        if self.matching_method == "OF_PyrLK":
            # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids
            kp2, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                                    feat_ref, None,
                                                    # maxLevel=6,
                                                    winSize=(21, 21),
                                                    criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            st = st.reshape(st.shape[0])  # status of points from frame to frame
            # Keypoints
            kp1 = feat_ref[st == 1]
            kp2 = kp2[st == 1]


        if fliter == True:

            threshold = 5.0

            movement_vectors = kp2 - kp1
            movement_magnitudes = np.linalg.norm(movement_vectors, axis=1)

            valid_indices = movement_magnitudes > threshold
            kp1 = kp1[valid_indices]
            kp2 = kp2[valid_indices]

        movement_vectors = np.array(kp2) - np.array(kp1)
        movement_magnitude = np.linalg.norm(movement_vectors, axis=1)
        avg_movement = np.mean(movement_magnitude)

        return kp1, kp2

    def motion_estimation_init(self, frame):
        """
        Processes first frame to initialize the reference features and the matrix R and t.
        Only for frame_id == 0.

        Args:
            frame {ndarray}: frame to be processed
            frame_id {int}: integer corresponding to the frame idself.loop_kf_idx
        """
        feat_ref = self.detect_features(frame)
        self.feat_ref = np.array([x.pt for x in feat_ref], dtype=np.float32)

    def loop_kf(self, frame, frame_id):
        if self.loop_enable:
            frame = frame.transpose(1, 2, 0) # (3, height, width) -> (height, width, 3)
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self.retrieval(frame, frame_id)
        else:
            print('Loop Closure is disabled!')

    def motion_estimation(self, frame, depth, frame_id):
        """
        Estimates the motion from current frame and computed keypoints

        Args:
            frame {ndarray}: frame to be processed
            frame_id {int}: integer corresponding to the frame id
            pose {list}: list with ground truth pose [x, y, z]
        """
        
        self.feat_ref, self.feat_curr = self.feature_matching(frame)

        R, t = compute_pose_2d2d(self.feat_ref, self.feat_curr, self.cam)

        depth = preprocess_depth(depth, crop=[[0.3, 1], [0, 1]], depth_range=[0, 50])


        cands = None


        E_pose = np.eye(4)
        if frame_id == 1:
            R, t = compute_pose_2d2d(self.feat_ref, self.feat_curr, self.cam)
            self.R = R
            self.t = t
            E_pose[:3, :3] = R
            E_pose[: 3, 3:] = t
            
        else:
            E_pose = np.eye(4)
            E_pose[:3, :3] = R
            E_pose[: 3, 3:] = t

            if frame_id > 10 and self.loop_enable:
                
                cands = self.retrieval.detect_loop(thresh=0.033, num_repeat=3)

                if cands is not None:
                    print(' -----> LOOP DETECTED!!')
                    print(cands)
                self.retrieval.save_up_to(frame_id)

            if cands is None or not self.loop_enable:

                scale = find_scale_from_depth(
                    self.cam,
                    self.feat_ref,
                    self.feat_curr,
                    np.linalg.inv(E_pose),
                    depth
                )

                if np.linalg.norm(t) == 0 or scale == -1.0:
                    R, t = compute_pose_3d2d(
                            self.feat_ref,
                            self.feat_curr,
                            self.prev_depth,
                            self.cam
                        )  # pose: from cur->ref
                scale = 1 if scale == -1.0 else scale

                # estimate camera motion
                self.t = self.t + scale * self.R.dot(t)
                self.R = self.R.dot(R)

            else:
                (i, j) = cands # e.g. cands = (812, 67)
                loop_frame_id = cands[1]

                feat_ref, feat_curr = self.feature_matching(self.frame_history[i], lc_frame_id = j)

                
                """ Avoid multiple back-to-back detections """
                self.retrieval.confirm_loop(i, j)
                self.retrieval.found.clear()


                """ Sim(3) Optimization """

                pts_world1, pts_world2 = get_matched_camera_points_vectorized(
                    depth_map1=self.depth_history[i], feat_points1=feat_curr,
                    depth_map2=self.depth_history[j], feat_points2=feat_ref,
                    K=self.K
                )
                
                s, r, t = compute_sim3_open3d(pts_world1, pts_world2)

                far_rel_pose = make_pypose_Sim3(r, t, s)[None] # shape([1, 8])

                poses_quat = torch.tensor(convert_pose_numpy_to_opencv_vectorized(self.pose_history, inv = True)).unsqueeze(0) # shape([1, n, 7])
                Gi = pp.SE3(poses_quat[:,self.loop_ii])
                Gj = pp.SE3(poses_quat[:,self.loop_jj])
                Gij = Gj * Gi.Inv() # shape([1, k, 7])
                prev_sim3 = SE3_to_Sim3(Gij).data[0].cpu() # shape([k, 8])
                loop_poses = pp.Sim3(torch.cat((prev_sim3, far_rel_pose))) # shape([k+1, 8])
                loop_ii = torch.cat((self.loop_ii, torch.tensor([i]))) # shape([k+1])
                loop_jj = torch.cat((self.loop_jj, torch.tensor([j]))) # shape([k+1])

                pred_poses = pp.SE3(poses_quat.squeeze()[:frame_id]).Inv().cpu() # shape([n, 7]) C2W
                # pred_poses = pp.SE3(poses_quat.squeeze()[:frame_id]).cpu() # shape([n, 7]) W2C

                self.loop_ii = loop_ii
                self.loop_jj = loop_jj

                # make sure that type of pred_poses and loop_poses is torch.float32
                # shape of return: shape([n, 8])
                final_est = pose_refinement(pred_poses.data.to(torch.float32), loop_poses.data.to(torch.float32), loop_ii, loop_jj)
                safe_i = final_est.shape[0] - 1
                aa = SE3_to_Sim3(pred_poses.cpu()).to(torch.float32)
                final_est = (aa[[safe_i]] * final_est[[safe_i]].Inv()) * final_est
                output = final_est

                output_matrix = poses_to_c2w_tensor(output[:, :7]).numpy()


                E_pose_loop = output_matrix[-1, :, :]
                t_loop = E_pose_loop[:3, 3:]
                R_loop = E_pose_loop[:3, :3]
                self.t = t_loop
                self.R = R_loop

                self.pose_history_old = self.pose_history.copy()
                self.pose_history[:output_matrix.shape[0], :, :] = output_matrix

                self.loop_kf_idx = output_matrix.shape[0]

        E_pose[:3, :3] = self.R
        E_pose[: 3, 3:] = self.t
        self.prev_depth = depth

        # check if number of features is enough (some features are lost in time due to the moving scene)
        if self.feat_ref.shape[0] < self.min_num_features:
            self.feat_curr = self.detect_features(frame)
            self.feat_curr = np.array([x.pt for x in self.feat_curr], dtype=np.float32)

        # update reference features
        self.feat_ref = self.feat_curr
        self.pose_ref = E_pose

        # self.pose_history.append(E_pose)
        self.pose_history[frame_id] = E_pose
        self.depth_history.append(depth)
        self.kp_history.append(self.feat_curr)
        self.frame_history.append(frame)

        if cands is None or not self.loop_enable:
            return False
        else:
            return True

    def update(self, frame, depth, frame_id):
        """
        Computes the camera motion between the current image and the previous one.
        frame shape: np.array([3, h, w]) and would be convert to [h, w]
        """

        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

        frame_uint8 = frame_normalized.astype(np.uint8)
        frame = cv2.cvtColor(np.transpose(frame_uint8, (1, 2, 0)), cv2.COLOR_RGB2GRAY)


        loop_flag = False

        if frame_id == 0:
            self.motion_estimation_init(frame)
        else:
            loop_flag = self.motion_estimation(frame, depth, frame_id)

        self.prev_frame = frame

        # Pose matrix
        pose = np.eye(4)
        pose[:3, :3] = self.R
        pose[:3, 3:] = self.t
        self.pose = pose

        self.w2c = np.linalg.inv(pose)

        return loop_flag
    
    def get_pose(self):
        return torch.from_numpy(self.w2c[:3, :3]), torch.from_numpy(self.w2c[:3, 3:]).squeeze()
    
    def update_tracking_pose(self, c2w):
        self.pose = c2w
        self.w2c = np.linalg.inv(self.pose)

        


def preprocess_depth(depth, crop, depth_range):
    """
    https://github.com/Huangying-Zhan/DF-VO
    """

    # set cropping region
    min_depth, max_depth = depth_range
    h, w = depth.shape
    y0, y1 = int(h * crop[0][0]), int(h * crop[0][1])
    x0, x1 = int(w * crop[1][0]), int(w * crop[1][1])
    depth_mask = np.zeros((h, w))
    depth_mask[y0:y1, x0:x1] = 1

    # set range mask
    depth_range_mask = (depth < max_depth) * (depth > min_depth)

    # set invalid pixel to zero depth
    valid_mask = depth_mask * depth_range_mask
    depth = depth * valid_mask
    return depth


def compute_pose_3d2d(kp1, kp2, depth_1, cam_intrinsics):
    """
    https://github.com/Huangying-Zhan/DF-VO
    """
    max_depth = 50
    min_depth = 0

    outputs = {}
    height, width = depth_1.shape

    # Filter keypoints outside image region
    x_idx = (kp1[:, 0] >= 0) * (kp1[:, 0] < width)
    kp1 = kp1[x_idx]
    kp2 = kp2[x_idx]
    x_idx = (kp2[:, 0] >= 0) * (kp2[:, 0] < width)
    kp1 = kp1[x_idx]
    kp2 = kp2[x_idx]
    y_idx = (kp1[:, 1] >= 0) * (kp1[:, 1] < height)
    kp1 = kp1[y_idx]
    kp2 = kp2[y_idx]
    y_idx = (kp2[:, 1] >= 0) * (kp2[:, 1] < height)
    kp1 = kp1[y_idx]
    kp2 = kp2[y_idx]

    # Filter keypoints outside depth range
    kp1_int = kp1.astype(int)
    kp_depths = depth_1[kp1_int[:, 1], kp1_int[:, 0]]
    non_zero_mask = (kp_depths != 0)
    depth_range_mask = (kp_depths < max_depth) * (kp_depths > min_depth)
    valid_kp_mask = non_zero_mask * depth_range_mask

    kp1 = kp1[valid_kp_mask]
    kp2 = kp2[valid_kp_mask]

    # Get 3D coordinates for kp1
    XYZ_kp1 = unprojection_kp(kp1, kp_depths[valid_kp_mask], cam_intrinsics)

    # initialize ransac setup
    best_rt = []
    best_inlier = 0
    max_ransac_iter = 3

    for _ in range(max_ransac_iter):
        # shuffle kp (only useful when random seed is fixed)
        new_list = np.arange(0, kp2.shape[0], 1)
        np.random.shuffle(new_list)
        new_XYZ = XYZ_kp1.copy()[new_list]
        new_kp2 = kp2.copy()[new_list]

        if new_kp2.shape[0] > 4:
            # PnP solver
            flag, r, t, inlier = cv2.solvePnPRansac(
                objectPoints=new_XYZ,
                imagePoints=new_kp2,
                cameraMatrix=cam_intrinsics.mat,
                distCoeffs=None,
                iterationsCount=100,  # number of iteration
                reprojectionError=1,  # inlier threshold value
            )

            # save best pose estimation
            if flag and inlier.shape[0] > best_inlier:
                best_rt = [r, t]
                best_inlier = inlier.shape[0]

    # format pose
    R = np.eye(3)
    t = np.zeros((3, 1))
    if len(best_rt) != 0:
        r, t = best_rt
        R = cv2.Rodrigues(r)[0]
    E_pose = np.eye(4)
    E_pose[:3, :3] = R
    E_pose[: 3, 3:] = t
    E_pose = np.linalg.inv(E_pose)
    R = E_pose[:3, :3]
    t = E_pose[: 3, 3:]
    return R, t


def unprojection_kp(kp, kp_depth, cam_intrinsics):
    """
    https://github.com/Huangying-Zhan/DF-VO
    """
    N = kp.shape[0]
    # initialize regular grid
    XYZ = np.ones((N, 3, 1))
    XYZ[:, :2, 0] = kp

    inv_K = np.ones((1, 3, 3))
    inv_K[0] = np.linalg.inv(cam_intrinsics.mat)  # cam_intrinsics.inv_mat
    inv_K = np.repeat(inv_K, N, axis=0)

    XYZ = np.matmul(inv_K, XYZ)[:, :, 0]
    XYZ[:, 0] = XYZ[:, 0] * kp_depth
    XYZ[:, 1] = XYZ[:, 1] * kp_depth
    XYZ[:, 2] = XYZ[:, 2] * kp_depth
    return XYZ



def compute_fundamental_residual(F, kp1, kp2):
    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1, 0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1, 0))

    Fm0 = F @ m0  # 3xN
    Ftm1 = F.T @ m1  # 3xN

    m1Fm0 = (np.transpose(Fm0, (1, 0)) @ m1).diagonal()
    res = m1Fm0 ** 2 / (np.sum(Fm0[:2] ** 2, axis=0) + np.sum(Ftm1[:2] ** 2, axis=0))
    return res


def compute_homography_residual(H_in, kp1, kp2):
    n = kp1.shape[0]
    H = H_in.flatten()

    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1, 0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1, 0))

    G0 = np.zeros((3, n))
    G1 = np.zeros((3, n))

    G0[0] = H[0] - m1[0] * H[6]
    G0[1] = H[1] - m1[0] * H[7]
    G0[2] = -m0[0] * H[6] - m0[1] * H[7] - H[8]

    G1[0] = H[3] - m1[1] * H[6]
    G1[1] = H[4] - m1[1] * H[7]
    G1[2] = -m0[0] * H[6] - m0[1] * H[7] - H[8]

    magG0 = np.sqrt(G0[0] * G0[0] + G0[1] * G0[1] + G0[2] * G0[2])
    magG1 = np.sqrt(G1[0] * G1[0] + G1[1] * G1[1] + G1[2] * G1[2])
    magG0G1 = G0[0] * G1[0] + G0[1] * G1[1]

    alpha = np.arccos(magG0G1 / (magG0 * magG1))

    alg = np.zeros((2, n))
    alg[0] = m0[0] * H[0] + m0[1] * H[1] + H[2] - \
             m1[0] * (m0[0] * H[6] + m0[1] * H[7] + H[8])

    alg[1] = m0[0] * H[3] + m0[1] * H[4] + H[5] - \
             m1[1] * (m0[0] * H[6] + m0[1] * H[7] + H[8])

    D1 = alg[0] / magG0
    D2 = alg[1] / magG1

    res = (D1 * D1 + D2 * D2 - 2.0 * D1 * D2 * np.cos(alpha)) / np.sin(alpha)

    return res


def calc_GRIC(res, sigma, n, model):
    """
    Calculate GRIC
    """
    R = 4
    sigmasq1 = 1. / sigma ** 2

    K = {
        "FMat": 7,
        "EMat": 5,
        "HMat": 8,
    }[model]
    D = {
        "FMat": 3,
        "EMat": 3,
        "HMat": 2,
    }[model]

    lam3RD = 2.0 * (R - D)

    sum_ = 0
    for i in range(n):
        tmp = res[i] * sigmasq1
        if tmp <= lam3RD:
            sum_ += tmp
        else:
            sum_ += lam3RD

    sum_ += n * D * np.log(R) + K * np.log(R * n)

    return sum_


def compute_pose_2d2d(kp_ref, kp_cur, cam_intrinsics):
    """
    https://github.com/Huangying-Zhan/DF-VO
    """
    principal_points = (cam_intrinsics.cx, cam_intrinsics.cy)

    # initialize ransac setup
    R = np.eye(3)
    t = np.zeros((3, 1))
    best_Rt = [R, t]
    best_inlier_cnt = 0
    max_ransac_iter = 3
    best_inliers = np.ones((kp_ref.shape[0], 1)) == 1

    # method GRIC of validating E-tracker
    if kp_cur.shape[0] > 10:


        H, H_inliers = cv2.findHomography(
            kp_cur,
            kp_ref,
            method=cv2.RANSAC,
            confidence=0.99,
            ransacReprojThreshold=1,
        )

        H_res = compute_homography_residual(H, kp_cur, kp_ref)
        H_gric = calc_GRIC(
            res=H_res,
            sigma=0.8,
            n=kp_cur.shape[0],
            model="HMat"
        )

        valid_case = True
    else:
        valid_case = False

    if valid_case:
        num_valid_case = 0
        for i in range(max_ransac_iter):  # repeat ransac for several times for stable result
            # shuffle kp_cur and kp_ref (only useful when random seed is fixed)
            new_list = np.arange(0, kp_cur.shape[0], 1)
            np.random.shuffle(new_list)
            new_kp_cur = kp_cur.copy()[new_list]
            new_kp_ref = kp_ref.copy()[new_list]

            E, inliers = cv2.findEssentialMat(
                new_kp_cur,
                new_kp_ref,
                focal=cam_intrinsics.fx,
                pp=principal_points,
                method=cv2.RANSAC,
                # method=cv2.LMEDS,
                prob=0.99,
                threshold=0.2,
            )


            # get F from E
            K = cam_intrinsics.mat
            F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
            E_res = compute_fundamental_residual(F, new_kp_cur, new_kp_ref)

            E_gric = calc_GRIC(
                res=E_res,
                sigma=0.8,
                n=kp_cur.shape[0],
                model='EMat'
            )
            valid_case = H_gric > E_gric

            # inlier check
            inlier_check = inliers.sum() > best_inlier_cnt

            # save best_E
            if inlier_check:
                best_E = E
                best_inlier_cnt = inliers.sum()

                revert_new_list = np.zeros_like(new_list)
                for cnt, i in enumerate(new_list):
                    revert_new_list[i] = cnt
                best_inliers = inliers[list(revert_new_list)]
            num_valid_case += (valid_case * 1)

        major_valid = num_valid_case > (max_ransac_iter / 2)
        
        if major_valid:
            cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, kp_cur, kp_ref,
                                                      focal=cam_intrinsics.fx,
                                                      pp=principal_points,
                                                      )

            # cheirality_check
            if cheirality_cnt > kp_cur.shape[0] * 0.1:
                best_Rt = [R, t]

    R, t = best_Rt
    return R, t
