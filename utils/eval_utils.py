import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.pose_utils import get_pose
from utils.logging_utils import Log

from utils.anchor_utils import anchor_in_frustum

import torchvision
import torch.nn.functional as F


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


# 定义一个函数，将张量转换为 JSON 可序列化的格式
def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):  # 如果是张量
        return obj.cpu().tolist()  # 移动到 CPU 并转换为列表
    elif isinstance(obj, (list, tuple)):  # 如果是列表或元组
        return [tensor_to_serializable(item) for item in obj]  # 递归处理每个元素
    elif isinstance(obj, dict):  # 如果是字典
        return {key: tensor_to_serializable(value) for key, value in obj.items()}  # 递归处理每个键值对
    else:
        return obj  # 其他类型直接返回


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    intrinsics,
    height,
    width,
    iteration="final",
    upsampling_method = 'bicubic'
):

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    img_save_path = os.path.join(save_dir, "img")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path, exist_ok=True)

    pose_txt_path = os.path.join(save_dir, 'poses_est.txt')
    pose_idx_txt_path = os.path.join(save_dir, 'poses_idx.txt')
    with open(pose_idx_txt_path, 'w') as f:
        for idx in frames.keys():
            f.write(f'{idx}' + '\n')

    with open(pose_txt_path, 'w') as f:
        for idx in frames.keys():
            saved_frame_idx.append(idx)
            frame = frames[idx]
            gt_image, _, gt_image_ori, _, _ = dataset[idx]

            gt_image_ori = gt_image_ori / 255

            h_ori, w_ori = gt_image_ori.shape[1], gt_image_ori.shape[2]

            pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T)) # W2C
            # pose_est = gen_pose_matrix(frame.R, frame.T) # C2W
            
            flat_matrix = pose_est.flatten()
            line = ' '.join(map(str, flat_matrix))
            f.write(line + '\n')


            opt_mask = torch.zeros(gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=gaussians.get_anchor, 
                                        intrinsics=intrinsics, 
                                        pose=get_pose(frame), 
                                        cam_center=frame.camera_center,
                                        distance_lis = gaussians.distance_lis,
                                        levels=gaussians.get_level, 
                                        h=height, 
                                        w=width)
            opt_mask.bitwise_or_(m)

            render_pkg = render(
                frame, 
                gaussians, 
                pipe, 
                background, 
                visible_mask=opt_mask
            )
            rendering = render_pkg['render']
            torchvision.utils.save_image(rendering, os.path.join(img_save_path, f'downsample_img_{idx}.png'))
            
            rendering = F.interpolate(rendering.unsqueeze(0), 
                                      size = (h_ori, w_ori), 
                                      mode = upsampling_method, 
                                      align_corners=False)
            rendering = rendering.squeeze(0)

            

            Log(f'Saving img_{idx}.png')
            torchvision.utils.save_image(rendering, os.path.join(img_save_path, f'img_{idx}.png'))

            image = torch.clamp(rendering, 0.0, 1.0)
            image = image.permute(1, 2, 0)

            gt_image_ori = torch.tensor(gt_image_ori, dtype=torch.float32).cuda().permute(1, 2, 0)

            gt = (gt_image_ori.detach().cpu().numpy()).astype(np.uint8)
            pred = (image.detach().cpu().numpy() * 255).astype(
                np.uint8
            )
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            img_pred.append(pred)
            img_gt.append(gt)

            mask = gt_image_ori > 0

            psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image_ori[mask]).unsqueeze(0))
            ssim_score = ssim((image).unsqueeze(0), (gt_image_ori).unsqueeze(0))
            lpips_score = cal_lpips((image.permute(2, 0, 1)).unsqueeze(0), (gt_image_ori.permute(2, 0, 1)).unsqueeze(0))

            psnr_array.append(psnr_score.item())
            ssim_array.append(ssim_score.item())
            lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
