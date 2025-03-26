import json
import os
import argparse
import numpy as np
import cv2
import pykitti
from third_party.semantic_kitti_api.auxiliary.laserscan import SemLaserScan
import yaml
import matplotlib.pyplot as plt
import torch
import open3d as o3d
from src.arguments import OptimizationParams
from argparse import ArgumentParser
import torchvision
from ultralytics import YOLO
import torch.nn.functional as F
from copy import deepcopy
from scipy.spatial import KDTree
from sklearn.neighbors import kneighbors_graph
import alphashape
import matplotlib
matplotlib.use("TkAgg")

def create_pcd_from_rgbd(depth, cam_intrinsics, rgb=None, uv=None):
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    xx = (uv[0, :] - cx) * depth / fx
    yy = (uv[1, :] - cy) * depth / fy
    zz = depth
    pcd = np.stack([xx, yy, zz], axis=1)
    if rgb is not None:
        pcd = np.concatenate([pcd, rgb], axis=1)
    return pcd

def expand_points_via_alpha_shape_and_knn(uv, image_gray, lidar_pnts, cam_intrinsics, alpha_value=0.1, k=3, show_mask=False):
    if uv.shape[0] == 2 and uv.shape[1] > 2:
        uv = uv.T
    points = uv.astype(np.float32)
    concave_hull = alphashape.alphashape(points, alpha_value)
    if concave_hull.geom_type == 'Polygon':
        hull_points = np.array(concave_hull.exterior.coords, dtype=np.int32)
    elif concave_hull.geom_type == 'MultiPolygon':
        hull_points = np.array(max(concave_hull.geoms, key=lambda a: a.area).exterior.coords, dtype=np.int32)
    else:
        raise ValueError(f"Unexpected geometry type: {concave_hull.geom_type}")
    mask = np.zeros_like(image_gray)
    cv2.fillPoly(mask, [hull_points], 255)
    if show_mask:
        plt.figure()
        plt.title("Alpha-shape Mask")
        plt.imshow(mask, cmap='gray')
        plt.show()
    uv_for_kdtree = points
    kdtree = KDTree(uv_for_kdtree)
    Z_vals = lidar_pnts[:, 2]
    K_inv = np.linalg.inv(cam_intrinsics)
    mask_rows, mask_cols = np.where(mask > 0)
    expanded_pixels = np.stack([mask_cols, mask_rows], axis=1)
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
        uv_hom = np.array([px, py, 1.0], dtype=np.float32)
        xyz_cam = z_interp * (K_inv @ uv_hom)
        new_pts_3d_list.append(xyz_cam)
    new_pts_3d = np.array(new_pts_3d_list, dtype=np.float32)
    new_pixels = np.stack([mask_cols, mask_rows], axis=0)
    return new_pts_3d, new_pixels, mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SemanticKITTI frame")
    parser.add_argument("--frame_id", type=int, default=53, help="Frame ID to use (default: 53)")
    args = parser.parse_args()
    frame_id = args.frame_id

    # Setup parameters
    mapping = {
        "00": "2011_10_03_drive_0027", "01": "2011_10_03_drive_0042",
        "02": "2011_10_03_drive_0034", "03": "2011_09_26_drive_0067",
        "04": "2011_09_30_drive_0016", "05": "2011_09_30_drive_0018",
        "06": "2011_09_30_drive_0020", "07": "2011_09_30_drive_0027",
        "08": "2011_09_30_drive_0028", "09": "2011_09_30_drive_0033",
        "10": "2011_09_30_drive_0034"
    }
    seq_lengths = {0: 4541, 5: 2761, 6: 1101, 8: 4071}
    drive_dict = {0: "0027", 5: "0020", 6: "0020", 8: "0020"}
    seq_id = 0
    seq_len = seq_lengths[seq_id]
    date_drive = mapping["%02d" % seq_id]
    date = date_drive[:10]
    drive = drive_dict[seq_id]

    # Replace Path
    kitti_root = '/media/michael/My Passport/data_odometry_color/dataset'
    raw_data_root = '/media/michael/My Passport/kitti_raw_calib'
    semkitti_root = '/media/michael/My Passport/data_odometry_velodyne/sequences'
    config_file = 'third_party/semantic_kitti_api/config/semantic-kitti-all.yaml'

    # Load calibration and raw data
    raw_data = pykitti.raw(raw_data_root, date, drive)
    o2Tv = raw_data.calib.T_cam2_velo
    o3Tv = raw_data.calib.T_cam3_velo
    o3To2 = o3Tv @ np.linalg.inv(o2Tv)
    print('Relative camera pose:\n', o3To2)
    baseline = abs(o3To2[0, 3])
    print('Baseline: ', baseline)
    P2 = raw_data.calib.P_rect_20
    cam_intrinsics = P2[:3, :3]

    # Load SemanticKITTI config and lidar data
    config = yaml.safe_load(open(config_file, 'r'))
    color_dict = config['color_map']
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=False)

    left_img_file = os.path.join(kitti_root, 'sequences/%02d' % seq_id, 'image_2', '%06d.png' % frame_id)
    right_img_file = os.path.join(kitti_root, 'sequences/%02d' % seq_id, 'image_3', '%06d.png' % frame_id)
    left_img = cv2.imread(left_img_file)
    img_h, img_w = left_img.shape[:2]

    lidar_file = os.path.join(semkitti_root, '%02d' % seq_id, 'velodyne', '%06d.bin' % frame_id)
    label_file = os.path.join(semkitti_root, '%02d' % seq_id, 'labels', '%06d.label' % frame_id)
    scan.open_scan(lidar_file)
    scan.open_label(label_file)
    scan.colorize()

    lidar_all = scan.points
    sem_labels_all = scan.sem_label
    inst_labels_all = scan.inst_label

    # Filter valid lidar points
    lidar_all_incam = o2Tv[:3, :3] @ lidar_all.T + o2Tv[:3, 3][:, None]
    valid_mask_incam = lidar_all_incam[2, :] > 0
    lidar_valid = lidar_all_incam[:, valid_mask_incam].T
    sem_labels_valid = sem_labels_all[valid_mask_incam]
    inst_labels_valid = inst_labels_all[valid_mask_incam]

    # Process coi=10 points (selecting a specific instance)
    coi = 10
    class_mask = (sem_labels_valid == coi)
    lidar_pnts_coi10 = lidar_valid[class_mask, :]
    inst_labels_coi10 = inst_labels_valid[class_mask]
    unique_inst_ids = np.unique(inst_labels_coi10)
    print('instance ids (coi=10): ', unique_inst_ids)
    inst_id = unique_inst_ids[1]
    inst_mask = (inst_labels_coi10 == inst_id)
    lidar_pnts_coi10 = lidar_pnts_coi10[inst_mask, :]
    print('num points (coi=10): ', lidar_pnts_coi10.shape[0])

    normalized_pnts = lidar_pnts_coi10 / lidar_pnts_coi10[:, 2][:, None]
    uv = (cam_intrinsics @ normalized_pnts.T)[:2, :].astype(int)
    valid_uv_mask = (uv[0, :] > 0) & (uv[0, :] < left_img.shape[1]) & (uv[1, :] > 0) & (uv[1, :] < left_img.shape[0])
    uv = uv[:, valid_uv_mask]
    print('max u (coi=10): ', uv[0, :].max())
    print('max v (coi=10): ', uv[1, :].max())

    left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    left_img_gray[:, :] = 0
    left_img_gray[uv[1, :].astype(int), uv[0, :].astype(int)] = 255

    new_pts_3d, new_pixels, mask = expand_points_via_alpha_shape_and_knn(
        uv=uv,
        image_gray=left_img_gray,
        lidar_pnts=lidar_pnts_coi10,
        cam_intrinsics=cam_intrinsics,
        alpha_value=0.1,
        k=3,
        show_mask=False
    )

    pts = create_pcd_from_rgbd(new_pts_3d[:, 2], cam_intrinsics,
                               left_img[new_pixels[1, :], new_pixels[0, :], ::-1],
                               new_pixels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.)
    filename = '/home/michael/Desktop/output_car.ply'
    o3d.io.write_point_cloud(filename, pcd, write_ascii=False, compressed=False, print_progress=False)

    # # -------------------------------------------------------------------------------------
    # # Save for 3DGS

    # points_tensor = torch.from_numpy(pts[:, :3]).float()  # Point coordinates, shape: (N, 3)
    # colors_tensor = torch.from_numpy(pts[:, 3:]).float()    # Color information, shape: (N, 3)

    # np.savetxt("/home/michael/Desktop/points.txt", pts[:, :3], delimiter=",")
    # np.savetxt("/home/michael/Desktop/colors.txt", pts[:, 3:], delimiter=",")

    # print("Saved point cloud data (points, scales, colors) to text files.")
    # # -------------------------------------------------------------------------------------

    # Process coi=40 (and related classes) points
    coi_values = [40, 44, 48, 49]
    class_mask_40 = np.isin(sem_labels_valid, coi_values)
    lidar_pnts_coi40 = lidar_valid[class_mask_40, :]
    print('num points (coi=40): ', lidar_pnts_coi40.shape[0])
    normalized_pnts_40 = lidar_pnts_coi40 / lidar_pnts_coi40[:, 2][:, None]
    uv_40 = (cam_intrinsics @ normalized_pnts_40.T)[:2, :].astype(int)
    valid_uv_mask_40 = (uv_40[0, :] > 0) & (uv_40[0, :] < left_img.shape[1]) & (uv_40[1, :] > 0) & (uv_40[1, :] < left_img.shape[0])
    uv_40 = uv_40[:, valid_uv_mask_40]
    depth_40 = lidar_pnts_coi40[:, 2][valid_uv_mask_40]
    print('max u (coi=40): ', uv_40[0, :].max())
    print('max v (coi=40): ', uv_40[1, :].max())
    colors_40 = left_img[uv_40[1, :], uv_40[0, :], ::-1]

    # Select 50 uv_40 points nearest to coi=10 uv mask points
    kdtree_10 = KDTree(uv.T)
    distances, _ = kdtree_10.query(uv_40.T, k=1)
    num_select = 50
    selected_indices = np.argsort(distances)[:num_select]
    selected_uv_40 = uv_40[:, selected_indices]
    selected_depth_40 = depth_40[selected_indices]
    pts_selected = create_pcd_from_rgbd(selected_depth_40, cam_intrinsics,
                                        left_img[selected_uv_40[1, :], selected_uv_40[0, :], ::-1],
                                        selected_uv_40)
    pcd_selected = o3d.geometry.PointCloud()
    pcd_selected.points = o3d.utility.Vector3dVector(pts_selected[:, :3])
    pcd_selected.colors = o3d.utility.Vector3dVector(pts_selected[:, 3:] / 255.)

    # Combine coi=10 and selected uv_40 point clouds
    pcd_combined = o3d.geometry.PointCloud()
    points_coi10 = np.asarray(pcd.points)
    colors_coi10 = np.asarray(pcd.colors)
    points_selected = np.asarray(pcd_selected.points)
    colors_selected = np.asarray(pcd_selected.colors)
    points_combined = np.vstack((points_coi10, points_selected))
    colors_combined = np.vstack((colors_coi10, colors_selected))
    pcd_combined.points = o3d.utility.Vector3dVector(points_combined)
    pcd_combined.colors = o3d.utility.Vector3dVector(colors_combined)
    filename_combined = '/home/michael/Desktop/output_cargnd.pcd'
    o3d.io.write_point_cloud(filename_combined, pcd_combined, write_ascii=False, compressed=False, print_progress=False)
    # Visualize the point cloud and coordinate frame
    o3d.visualization.draw_geometries([pcd_combined])

    # Plane fitting using RANSAC on pcd_selected
    ransac_threshold = 0.01
    ransac_n = 3
    num_iterations = 1000
    plane_model, inliers = pcd_selected.segment_plane(distance_threshold=ransac_threshold,
                                                      ransac_n=ransac_n,
                                                      num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print("RANSAC plane: {}x + {}y + {}z + {} = 0".format(a, b, c, d))
    plane_normal_ransac = np.array([a, b, c])

    # Project coi=10 point cloud onto the RANSAC plane
    pcd_points = np.asarray(pcd.points)
    [a, b, c, d] = plane_model
    n = np.array([a, b, c])
    n_norm = n / np.linalg.norm(n)
    d_norm = d / np.linalg.norm(n)
    distances = np.dot(pcd_points, n_norm) + d_norm
    projected_points = pcd_points - distances[:, None] * n_norm[None, :]

    # Build a 2D coordinate system in the projection plane
    u = np.array([1, 0, 0]) if abs(n_norm[0]) < 0.9 else np.array([0, 1, 0])
    x_axis_plane = u - np.dot(u, n_norm) * n_norm
    x_axis_plane = x_axis_plane / np.linalg.norm(x_axis_plane)
    y_axis_plane = np.cross(n_norm, x_axis_plane)
    y_axis_plane = y_axis_plane / np.linalg.norm(y_axis_plane)
    origin = projected_points.mean(axis=0)
    points_2d = np.zeros((projected_points.shape[0], 2))
    for i in range(projected_points.shape[0]):
        p_rel = projected_points[i] - origin
        points_2d[i, 0] = np.dot(p_rel, x_axis_plane)
        points_2d[i, 1] = np.dot(p_rel, y_axis_plane)

    # Create a binary image from the 2D points for Hough transform
    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)
    range_xy = max_xy - min_xy
    scale = 500.0 / np.max(range_xy)
    img_width = int(np.ceil(range_xy[0] * scale)) + 10
    img_height = int(np.ceil(range_xy[1] * scale)) + 10
    img = np.zeros((img_height, img_width), dtype=np.uint8)
    pts_img = np.zeros((points_2d.shape[0], 2), dtype=np.int32)
    pts_img[:, 0] = ((points_2d[:, 0] - min_xy[0]) * scale).astype(np.int32)
    pts_img[:, 1] = ((points_2d[:, 1] - min_xy[1]) * scale).astype(np.int32)
    for pt in pts_img:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < img_height and 0 <= x < img_width:
            img[y, x] = 255

    # Extract 2D lines using HoughLinesP
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=80, minLineLength=80, maxLineGap=5)
    if lines is None:
        lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    models = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pt1 = np.array([x1 / scale + min_xy[0], y1 / scale + min_xy[1]])
            pt2 = np.array([x2 / scale + min_xy[0], y2 / scale + min_xy[1]])
            d = pt2 - pt1
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-6:
                continue
            d = d / norm_d
            models.append((pt1, d))

    # Compute slopes and split lines into two groups
    slopes = []
    for i, model in enumerate(models):
        p0, d = model
        if d[0] < 0:
            d = -d
        slope = np.inf if abs(d[0]) < 1e-6 else d[1] / d[0]
        slopes.append(slope)
    slopes = np.array(slopes)
    finite_slopes = slopes[np.isfinite(slopes)]
    avg_slope = np.mean(finite_slopes) if len(finite_slopes) > 0 else 0
    print("Average slope: ", avg_slope)
    group1_indices = np.where(slopes > avg_slope)[0]
    group2_indices = np.where(slopes <= avg_slope)[0]

    def average_line_parameters(models, indices):
        p0_list, d_list = [], []
        for i in indices:
            p0, d = models[i]
            if d[0] < 0:
                d = -d
            p0_list.append(p0)
            d_list.append(d)
        avg_p0 = np.mean(p0_list, axis=0)
        avg_d = np.mean(d_list, axis=0)
        avg_d = avg_d / np.linalg.norm(avg_d)
        return avg_p0, avg_d

    if len(group1_indices) > 0:
        avg_p0_1, avg_d_1 = average_line_parameters(models, group1_indices)
    else:
        avg_p0_1, avg_d_1 = None, None
    if len(group2_indices) > 0:
        avg_p0_2, avg_d_2 = average_line_parameters(models, group2_indices)
    else:
        avg_p0_2, avg_d_2 = None, None
    print("Group1 average: ", avg_p0_1, avg_d_1)
    print("Group2 average: ", avg_p0_2, avg_d_2)

    # Choose main direction based on group counts
    if len(group1_indices) > len(group2_indices):
        directions_group1 = []
        for i in group1_indices:
            _, d = models[i]
            if d[0] < 0:
                d = -d
            directions_group1.append(d)
        directions_group1 = np.array(directions_group1)
        avg_direction_2d = np.mean(directions_group1, axis=0)
        avg_direction_2d = avg_direction_2d / np.linalg.norm(avg_direction_2d)
    else:
        directions_group2 = []
        for i in group2_indices:
            _, d = models[i]
            if d[0] < 0:
                d = -d
            directions_group2.append(d)
        directions_group2 = np.array(directions_group2)
        avg_direction_2d = np.mean(directions_group2, axis=0)
        avg_direction_2d = avg_direction_2d / np.linalg.norm(avg_direction_2d)
    print("2D average direction (x-axis):", avg_direction_2d)

    # Convert 2D average direction to 3D using the 2D plane basis
    x_axis_3d = avg_direction_2d[0] * x_axis_plane + avg_direction_2d[1] * y_axis_plane
    x_axis_3d = x_axis_3d / np.linalg.norm(x_axis_3d)
    print("3D x-axis:", x_axis_3d)
    z_axis_3d = n_norm
    y_axis_3d = np.cross(z_axis_3d, x_axis_3d)
    y_axis_3d = y_axis_3d / np.linalg.norm(y_axis_3d)
    print("3D y-axis:", y_axis_3d)
    print("3D z-axis:", z_axis_3d)

    # Set coordinate frame origin as the mean of pcd points
    pcd_points = np.asarray(pcd.points)
    origin_3d = (np.min(pcd_points, axis=0) + np.max(pcd_points, axis=0)) / 2

    # Build coordinate frame using Open3D LineSet
    frame_scale = 1.0
    frame_points = [
        origin_3d,
        origin_3d + x_axis_3d * frame_scale,
        origin_3d + y_axis_3d * frame_scale,
        origin_3d + z_axis_3d * frame_scale,
    ]
    frame_lines = [[0, 1], [0, 2], [0, 3]]
    frame_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    frame = o3d.geometry.LineSet()
    frame.points = o3d.utility.Vector3dVector(frame_points)
    frame.lines = o3d.utility.Vector2iVector(frame_lines)
    frame.colors = o3d.utility.Vector3dVector(frame_colors)

    # Visualize the point cloud and coordinate frame
    o3d.visualization.draw_geometries([pcd, frame])

    # ---------------------------------------------------
    # Perform symmetry completion of the point cloud based on the computed coordinate system
    # The symmetry plane is the x-z plane of the local coordinate system (i.e., y=0 plane),
    # with its normal vector given by the computed y_axis_3d,
    # and the origin is origin_3d (previously computed as the mean of pcd.points)

    # Get the original point cloud data (assuming pcd is the previously processed point cloud object)
    pcd_points = np.asarray(pcd.points)

    # Perform mirror reflection using the computed origin_3d and y_axis_3d
    # For each point p, compute the component of (p - origin_3d) along y_axis_3d and invert that component
    # Reflection formula: p' = p - 2 * ((p - origin_3d) dot y_axis_3d) * y_axis_3d
    origin_3d = np.asarray(origin_3d)  # Computed local coordinate system origin

    centered_points = pcd_points - origin_3d   # Translate the point cloud to the local coordinate system

    # Calculate projections along x_axis_3d and y_axis_3d
    proj_x = np.dot(centered_points, x_axis_3d)
    proj_y = np.dot(centered_points, y_axis_3d)
    # Compute the minimum and maximum values along the new x and y axes
    min_x, max_x = np.min(proj_x), np.max(proj_x)
    min_y, max_y = np.min(proj_y), np.max(proj_y)
    print("Coordinate range along new x-axis: min = {:.3f}, max = {:.3f}".format(min_x, max_x))
    print("Coordinate range along new y-axis: min = {:.3f}, max = {:.3f}".format(min_y, max_y))

    if (max_x > max_y):
        axis = y_axis_3d              # Use the computed y-axis as the symmetry direction
    else:
        axis = x_axis_3d
    proj = np.dot(centered_points, axis)       # Compute the projection of each point onto the chosen axis
    mirrored_points = pcd_points - 2 * np.outer(proj, axis)  # Obtain the reflected points

    # Merge the original point cloud with the mirrored point cloud
    points_symmetry = np.vstack((pcd_points, mirrored_points))
    colors_original = np.asarray(pcd.colors)
    # Copy the colors directly (adjust as needed)
    colors_symmetry = colors_original.copy()
    colors_combined = np.vstack((colors_original, colors_symmetry))

    # Create the symmetry-completed point cloud object
    pcd_symmetry = o3d.geometry.PointCloud()
    pcd_symmetry.points = o3d.utility.Vector3dVector(points_symmetry)
    pcd_symmetry.colors = o3d.utility.Vector3dVector(colors_combined)

    # Save the symmetry-completed point cloud to a file
    symmetry_filename = '/home/michael/Desktop/output_car_symmetry_axes.ply'
    o3d.io.write_point_cloud(symmetry_filename, pcd_symmetry, write_ascii=False, compressed=False, print_progress=False)
    print("Symmetry-completed point cloud saved as:", symmetry_filename)

    # Visualize the symmetry-completed point cloud
    o3d.visualization.draw_geometries([pcd_symmetry])

    # # ---------------------------------------------------
    # # Save for 3DGS
    # print(points_symmetry.shape)
    # print(colors_combined.shape)

    # np.savetxt("/home/michael/Desktop/points_com.txt", points_symmetry, delimiter=",")
    # np.savetxt("/home/michael/Desktop/colors_com.txt", colors_combined, delimiter=",")
    # # -------------------------------------------------------------------------------------

    # # Output img_masked
    # filename_img = '/home/michael/Desktop/img.png'
    # img_masked = np.zeros_like(left_img)
    # img_masked[new_pixels[1, :], new_pixels[0, :]] = left_img[new_pixels[1, :], new_pixels[0, :]]

    # filename_mask = '/home/michael/Desktop/mask.png'
    # masked = np.zeros_like(left_img)
    # masked[new_pixels[1, :], new_pixels[0, :]] = 255

    # cv2.imwrite(filename_img, img_masked)
    # print("Image saved to:", filename_img)

    # cv2.imwrite(filename_mask, masked)
    # print("Mask saved to:", filename_mask)

    # calib_data = {
    #     "relative_pose": o2Tv.tolist(),
    #     "intrinsics": cam_intrinsics.tolist(),
    #     "image_size": [int(img_w), int(img_h)]
    # }
    # calib_file = '/home/michael/Desktop/calibration.json'
    # with open(calib_file, "w") as f:
    #     json.dump(calib_data, f, indent=4)
    # print("Calibration data saved to:", calib_file)
