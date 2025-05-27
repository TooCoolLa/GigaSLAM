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
    def __init__(self, color_folder):
        self.input_folder = color_folder
        self.color_paths = sorted(glob.glob(f"{color_folder}/*.[jp][pn]g"))
        self.n_img = len(self.color_paths)



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

            w_set = config['Hierarchical']['rendering_width']
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
        # pose = self.poses[idx]
        pose = None

        image_ori = np.array(Image.open(color_path))
        if len(image_ori.shape) == 2:
            # gray to rgb
            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image_ori, (self.width, self.height))
        depth = None
        depth_ori = None

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

        # pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, image_ori.transpose(2, 0, 1), depth_ori, pose


class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        color_folder = config["Dataset"]["color_path"]
        # pose_path = config["Dataset"]["pose_path"]
        parser = KittiParser(color_folder)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        # self.poses = parser.poses


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "kitti":
        return KITTIDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
