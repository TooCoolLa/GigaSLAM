import csv
import glob
import os
import time

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from gaussian_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    pass



class KittiParser:
    def __init__(self, color_folder, pose_path):
        self.input_folder = color_folder
        self.color_paths = sorted(glob.glob(f"{color_folder}/*.[jp][pn]g"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{pose_path}")

    def load_poses(self, path):
        self.poses = []
        frames = []
        for i in range(self.n_img):
            pose = np.eye(4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames



class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        print(calibration)

        if True:
            self.fx_ori = calibration["fx"]
            self.fy_ori = calibration["fy"]
            self.cx_ori = calibration["cx"]
            self.cy_ori = calibration["cy"]
            self.width_ori = calibration["width"]
            self.height_ori = calibration["height"]

            w_set = 480
            h0, w0 =  self.height, self.width
            ratio = w_set / w0
            h_set = ratio * h0
            h1 = int(h0 * np.sqrt((h_set * w_set) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((h_set * w_set) / (h0 * w0)))

            self.height, self.width = h1, w1
            self.fx *= ratio
            self.fy *= ratio
            self.cx *= ratio
            self.cy *= ratio
            self.ratio = ratio


        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }



    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image_ori = np.array(Image.open(color_path))
        image = cv2.resize(image_ori, (self.width, self.height))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth_ori = np.array(Image.open(depth_path)) / self.depth_scale
            depth = cv2.resize(depth_ori, (self.width, self.height))

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, image_ori.transpose(2, 0, 1), depth_ori, pose


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose


class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        color_folder = config["Dataset"]["color_path"]
        pose_path = config["Dataset"]["pose_path"]
        parser = KittiParser(color_folder, pose_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        self.pipeline = rs.pipeline()
        self.h, self.w = 720, 1280
        
        self.depth_scale = 0
        if self.config["Dataset"]["sensor_type"] == "depth":
            self.has_depth = True 
        else: 
            self.has_depth = False

        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        if self.has_depth:
            self.rs_config.enable_stream(rs.stream.depth)

        self.profile = self.pipeline.start(self.rs_config)

        if self.has_depth:
            self.align_to = rs.stream.color
            self.align = rs.align(self.align_to)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # rgb_sensor.set_option(rs.option.enable_auto_white_balance, True)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
                self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
        
        


    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        depth = None

        frameset = self.pipeline.wait_for_frames()

        if self.has_depth:
            aligned_frames = self.align.process(frameset)
            rgb_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)
        else:
            rgb_frame = frameset.get_color_frame()

        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )

        return image, depth, pose


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "kitti":
        return KITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
