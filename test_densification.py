import os
import numpy as np
import cv2
import pykitti
from third_party.semantic_kitti_api.auxiliary.laserscan import SemLaserScan
import yaml
import matplotlib.pyplot as plt
# from src.gaussian_model import GaussianModel
import torch
import open3d as o3d
# from src.utils.utils import get_render_settings, render_gaussian_model
# from src.utils.mapping_utils import visualize_image
# from src.utils.gaussian_model_utils import build_rotation
from src.arguments import OptimizationParams
from argparse import ArgumentParser
# from src.utils.pose_utils import compute_camera_opt_params, multiply_quaternions
import torchvision
from ultralytics import YOLO
# from src.losses import l1_loss, isotropic_loss, ssim
# from src.utils.utils import np2ptcloud
import torch.nn.functional as F
from copy import deepcopy

from scipy.spatial import KDTree
from sklearn.neighbors import kneighbors_graph
import alphashape


def create_pcd_from_rgbd(depth, cam_intrinsics, rgb=None, uv=None):
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    xx = uv[0, :] - cx
    yy = uv[1, :] - cy

    xx = xx * depth / fx
    yy = yy * depth / fy

    zz = depth

    pcd = np.stack([xx, yy, zz], axis=1)

    if rgb is not None:
        pcd = np.concatenate([pcd, rgb], axis=1)
    
    return pcd


def expand_points_via_alpha_shape_and_knn(
    uv, # Shape (2, N): x,y coordinates of projected points in image plane
    image_gray, # Sized by image input
    lidar_pnts, # Shape (N, 3): 3D points in camera coordinates (x, y, z).
    cam_intrinsics, 
    alpha_value=0.1, 
    k=3,
    show_mask=False
):
    """
    Expands a sparse set of scanned points:
      1. Computing a concave hull (via alpha-shape) to form a mask.
      2. Performing kNN-based interpolation of z-values for pixels inside the hull.
      3. Back-projecting those 2D pixels to camera-centric 3D coordinates.
    """

    if uv.shape[0] == 2 and uv.shape[1] > 2:
        uv = uv.T  

    points = uv.astype(np.float32)

    concave_hull = alphashape.alphashape(points, alpha_value)

    if concave_hull.geom_type == 'Polygon':
        hull_points = np.array(concave_hull.exterior.coords, dtype=np.int32)
    elif concave_hull.geom_type == 'MultiPolygon':
        # Take the largest polygon by area in case there's more than one
        hull_points = np.array(
            max(concave_hull.geoms, key=lambda a: a.area).exterior.coords,
            dtype=np.int32
        )
    else:
        raise ValueError(
            f"Unexpected geometry type: {concave_hull.geom_type}"
        )
    mask = np.zeros_like(image_gray)
    cv2.fillPoly(mask, [hull_points], 255)

    if show_mask:
        plt.figure()
        plt.title("Alpha-shape Mask")
        plt.imshow(mask, cmap='gray')
        plt.show()

    uv_for_kdtree = points  # shape (N, 2)
    kdtree = KDTree(uv_for_kdtree)
    Z_vals = lidar_pnts[:, 2]
    K_inv = np.linalg.inv(cam_intrinsics)

    # Find all in-mask pixels
    mask_rows, mask_cols = np.where(mask > 0)
    expanded_pixels = np.stack([mask_cols, mask_rows], axis=1)  # shape (M, 2)

    new_pts_3d_list = []

    for (px, py) in expanded_pixels:
        distances, indices = kdtree.query([px, py], k=k)

        distances = np.atleast_1d(distances)
        indices  = np.atleast_1d(indices)

        if np.any(distances < 1e-6):
            z_interp = Z_vals[indices[np.argmin(distances)]]
        else:
            weights = 1.0 / (distances ** 2)
            z_interp = np.sum(weights * Z_vals[indices]) / np.sum(weights)

        # Back-project to xyz
        uv_hom = np.array([px, py, 1.0], dtype=np.float32)
        xyz_cam = z_interp * (K_inv @ uv_hom)

        new_pts_3d_list.append(xyz_cam)

    new_pts_3d = np.array(new_pts_3d_list, dtype=np.float32)
    new_pixels = np.stack([mask_cols, mask_rows], axis=0)

    return new_pts_3d, new_pixels, mask


if __name__ == "__main__":
    
    mapping = {
        "00": "2011_10_03_drive_0027",
        "01": "2011_10_03_drive_0042",
        "02": "2011_10_03_drive_0034",
        "03": "2011_09_26_drive_0067",
        "04": "2011_09_30_drive_0016",
        "05": "2011_09_30_drive_0018",
        "06": "2011_09_30_drive_0020",
        "07": "2011_09_30_drive_0027",
        "08": "2011_09_30_drive_0028",
        "09": "2011_09_30_drive_0033",
        "10": "2011_09_30_drive_0034"
    }
    seq_lenths = {0: 4541, 5: 2761, 6: 1101, 8: 4071}
    # Modified, Attention
    drive_dict = {0: "0027", 5: "0020", 6: "0020", 8: "0020"}
    
    seq_id = 0
    visualize_results = True
    optimize_pose = False
    seq_len = seq_lenths[seq_id]
    date_drive = mapping["%02d" % seq_id]
    date = date_drive[:10]
    drive = drive_dict[seq_id]

    # kitti_root = '/home/hanwen/data/kitti_odometry/dataset/'
    # raw_data_root = '/home/hanwen/data/kitti_raw'
    # semkitti_root = '/home/hanwen/shared/data/semanticKITTI/dataset/sequences'
    # config_file = 'third_party/semantic_kitti_api/config/semantic-kitti-all.yaml'
    # shape_splat_root = '/home/hanwen/data/ShapeSplat'
    # model_ply_path = os.path.join(shape_splat_root, 'car', "02958343-1a0bc9ab92c915167ae33d942430658c.ply")

    kitti_root = '/media/michael/My Passport/data_odometry_color/dataset'
    raw_data_root = '/media/michael/My Passport/kitti_raw_calib'
    semkitti_root = '/media/michael/My Passport/data_odometry_velodyne/sequences'
    config_file = 'third_party/semantic_kitti_api/config/semantic-kitti-all.yaml'
    
    
    # load calibration
    raw_data = pykitti.raw(raw_data_root, date, drive)
    o2Tv = raw_data.calib.T_cam2_velo
    o3Tv = raw_data.calib.T_cam3_velo
    o3To2 = o3Tv @ np.linalg.inv(o2Tv)
    print('Relative camera pose:\n', o3To2)
    baseline = abs(o3To2[0, 3])
    print('Baseline: ', baseline)

    P2 = raw_data.calib.P_rect_20
    cam_intrinsics = P2[:3, :3]

    # load semantic KITTI config
    config = yaml.safe_load(open(config_file, 'r'))
    color_dict = config['color_map']
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=False)

    frame_id = 12
    ######### simulate depth from stereo images #########
    left_img_file = os.path.join(kitti_root, 'sequences/%02d' % seq_id, 'image_2', '%06d.png' % frame_id)
    right_img_file = os.path.join(kitti_root, 'sequences/%02d' % seq_id, 'image_3', '%06d.png' % frame_id)

    # create a yolox for segementation
    left_img = cv2.imread(left_img_file)
    # left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    img_h, img_w = left_img.shape[:2]
    # visualize_image(left_img, 'left image')

    # load lidar data
    lidar_file = os.path.join(semkitti_root, '%02d' % seq_id, 'velodyne', '%06d.bin' % frame_id)
    label_file = os.path.join(semkitti_root, '%02d' % seq_id, 'labels', '%06d.label' % frame_id)

    scan.open_scan(lidar_file)
    scan.open_label(label_file)
    scan.colorize()

    lidar_all = scan.points
    sem_labels_all = scan.sem_label
    inst_labels_all = scan.inst_label

    lidar_all_incam = o2Tv[:3, :3] @ lidar_all.T + o2Tv[:3, 3][:, None]
    valid_mask_incam = lidar_all_incam[2, :] > 0
    lidar_pnts = lidar_all_incam[:, valid_mask_incam].T
    sem_labels = sem_labels_all[valid_mask_incam]
    inst_labels = inst_labels_all[valid_mask_incam]

    coi = 10
    class_mask = np.zeros_like(sem_labels, dtype=bool)
    class_mask = np.logical_or(class_mask, sem_labels == coi)

    lidar_pnts = lidar_pnts[class_mask, :]
    inst_labels = inst_labels[class_mask]

    unique_inst_ids = np.unique(inst_labels)
    print('instance ids: ', unique_inst_ids)
    inst_id = unique_inst_ids[1]
    inst_lidar_mask = inst_labels == inst_id
    lidar_pnts = lidar_pnts[inst_lidar_mask, :]
    print('num points: ', lidar_pnts.shape[0])

    # project lidar points to image
    normalized_pnts = lidar_pnts / lidar_pnts[:, 2][:, None]
    uv = (cam_intrinsics @ normalized_pnts.T)[:2, :].astype(int)
    valid_uv_mask = (uv[0, :] > 0) & (uv[0, :] < left_img.shape[1]) \
        & (uv[1, :] > 0) & (uv[1, :] < left_img.shape[0])
    uv = uv[:, valid_uv_mask]
    print('max u: ', uv[0, :].max())
    print('max v: ', uv[1, :].max())

    left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    left_img_gray[:,:] = 0
    left_img_gray[uv[1, :].astype(int), uv[0, :].astype(int)] = 255

    new_pts_3d, new_pixels, mask = expand_points_via_alpha_shape_and_knn(
        uv=uv,
        image_gray=left_img_gray,
        lidar_pnts=lidar_pnts,
        cam_intrinsics=cam_intrinsics,
        alpha_value=0.1,    # or your chosen alpha
        k=3,
        show_mask=False     # set True to visualize the mask
    )

    # print("Original point cloud size:", lidar_pnts.shape[0])
    # print("Newly added points:", new_pts_3d.shape[0])
    # print("Uv size:", uv.shape[0], uv.shape[1])
    # print("Pixel size:", new_pixels.shape[0], new_pixels.shape[1])

    pts_original = create_pcd_from_rgbd(lidar_pnts[:, 2], cam_intrinsics, left_img[uv[1, :], uv[0, :], ::-1], uv)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_original[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pts_original[:, 3:] / 255.)
    o3d.visualization.draw_geometries([pcd])

    pts = create_pcd_from_rgbd(new_pts_3d[:, 2], cam_intrinsics, left_img[new_pixels[1, :], new_pixels[0, :], ::-1], new_pixels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.)
    o3d.visualization.draw_geometries([pcd])
        
