import os

from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt
from matplotlib.path import Path as plt_path
from pandaset import geometry
import numpy as np
from sklearn.neighbors import LocalOutlierFactor  # RANSAC outlier removal 
import open3d as o3d

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_funcs import panda_funcs
from src.gt import vis_gt_normal
import src.utils.data_utils as data
import src.utils.linalg as linalg

# Video generation script:
# ./src/_experiments/scripts/gen_gt_video.sh --seq-num 41
# ./src/_experiments/scripts/gen_gt_video.sh --seq-num 41 --type proj_filtered

# OPTIIONAL TODOS:
# TODO: filter out points from back of the car (use ego frame)
# TODO: investigate: why there is an x-rotation in ego frame? (check if it's a bug)
# TODO: (IDEA) use raw points for image based result visualization (plane and normal on it with ROI)
#       OR just calculate ROI size in 3D with points
# TODO: add semantic category filtering
# TODO: try plotting in Open3D? (open3D github issue)

def process_data(panda_root, seq_num, lidar_sensor=None):
	dataset, seq_list = panda_funcs.read_pandaset(panda_root)
	cam_obj = panda_funcs.load_panda_seq_cam(dataset, seq_list, seq_num)
	images = panda_funcs.get_panda_seq_images(cam_obj, pil=True)
	camera_K = cam_obj.intrinsics
	lidar_obj = panda_funcs.load_panda_seq_lidar(dataset, seq_list, seq_num)
	pcs = panda_funcs.parse_panda_seq_pc(lidar_obj, sensor=lidar_sensor)
	cam_poses = cam_obj.poses[:]
	lidar_poses = panda_funcs.parse_panda_seq_lidar_poses(lidar_obj, sensor=lidar_sensor)
	return images, camera_K, pcs, cam_poses, lidar_poses


def calculate_scaled_rois(rois, src_img_size, dst_img_size):
	src_width, src_height = src_img_size
	dst_width, dst_height = dst_img_size

	# scale factors for width and height
	scale_x = dst_width / src_width
	scale_y = dst_height / src_height

	# scale each ROI
	scaled_rois = []
	for roi in rois:
		scaled_roi = [
			(int(point[0] * scale_x), int(point[1] * scale_y)) for point in roi
		]
		scaled_rois.append(scaled_roi)
	return scaled_rois


def filter_points_in_roi(points2d, points3d, roi):
	"""
	Filters 2D and corresponding 3D points based on whether they are inside the ROI.

	:param points2d: Nx2 array of 2D projected points.
	:param points3d: Nx3 array of corresponding 3D points.
	:param roi: List of (x, y) tuples defining the ROI polygon.
	:return: (Filtered 2D and 3D points, Boolean mask)
	"""
	roi_path = plt_path(roi)  # Create a matplotlib.Path object for the ROI polygon
	mask = roi_path.contains_points(points2d)  # Boolean mask of points inside the ROI
	return points2d[mask], points3d[mask], mask


def tf_pts_to_ego(pts, curr_pose):
	"""
	Transforms LiDAR points from the current frame (world or local) into the ego frame.
	
	:param lidar_points: Nx3 numpy array of LiDAR points in the current frame.
	:param pose: 4x4 pose matrix (from world frame to ego frame).
	:return: Transformed Nx3 numpy array of points in the ego frame.
	"""
	return geometry.lidar_points_to_ego(pts[:, :3], curr_pose)


def tf_pts_to_first_pose(pts, curr_pose, first_pose):
	"""
	Transforms points to the first frame.

	:param pts: Nx3 numpy array of LiDAR points in the current frame.
	:param curr_pose: Dictionary with 'heading' and 'position' keys.
	:param first_pose: Dictionary with 'heading' and 'position' keys.

	:return: Transformed Nx3 numpy array of points in the first frame.
	"""
	first_pose_mat = geometry._heading_position_to_mat(
		first_pose['heading'], first_pose['position']
	)
	curr_pose_mat = geometry._heading_position_to_mat(
		curr_pose['heading'], curr_pose['position']
	)

	# compute relative transformation: T_relative = inv(T_0) * T_n
	tf_to_first_pose = np.linalg.inv(first_pose_mat) # @ curr_pose_mat

	# apply rotation and translation to the points
	transformed_pts = (
		tf_to_first_pose[:3, :3] @ pts.T + tf_to_first_pose[:3, [3]]
	).T
	
	return transformed_pts


def get_cam_tf_mat(cam_pose, lidar_first_pose):
	"""
	Return the camera to LiDAR transformation matrix along with
	the transformation into the sequence's first frame.
	"""
	first_pose_mat = geometry._heading_position_to_mat(
		lidar_first_pose['heading'], lidar_first_pose['position']
	)
	tf_to_first_pose = np.linalg.inv(first_pose_mat)[:3, :3]
	
	cam_pose_mat = geometry._heading_position_to_mat(
		cam_pose['heading'], cam_pose['position']
	)
	tf_to_cam = cam_pose_mat[:3, :3]
	
	# R = inv(R_0) * R_cam (only rotations)
	return tf_to_first_pose @ tf_to_cam


def get_cam_tf_mat_static(lidar_first_pose):
	"""
	Return the camera to LiDAR transformation matrix along with
	the transformation into the sequence's first frame.
	"""

	rotation, translation = data.read_calibration_from_yaml(
     "src/_experiments/gt/calib_data.yaml"
    )
	cam_pose_mat = geometry._heading_position_to_mat(
		rotation, translation
	)
	tf_to_cam = cam_pose_mat[:3, :3]
 
	first_pose_mat = geometry._heading_position_to_mat(
		lidar_first_pose['heading'], lidar_first_pose['position']
	)
	tf_to_first_pose = np.linalg.inv(first_pose_mat)[:3, :3]
	
	# R = inv(R_0) * R_cam (only rotations)
	return tf_to_first_pose @ tf_to_cam


def calc_ground_plane_ransac(points3d, threshold=0.01, max_iterations=100):
	"""
	Estimates a ground plane using RANSAC.

	:param points3d: Nx3 numpy array of 3D points.
	:param threshold: Distance threshold to consider a point an inlier.
	:param max_iterations: Maximum number of RANSAC iterations.
	:return: Plane coefficients (a, b, c, d) for the equation ax + by + cz + d = 0.
	"""
	best_plane = None
	max_inliers = 0

	for _ in range(max_iterations):
		# Randomly sample 3 points
		sample_indices = np.random.choice(points3d.shape[0], 3, replace=False)
		sample_points = points3d[sample_indices]

		# Calculate the plane coefficients (a, b, c, d)
		v1 = sample_points[1] - sample_points[0]
		v2 = sample_points[2] - sample_points[0]
		normal = np.cross(v1, v2)  # Normal vector to the plane

		if np.linalg.norm(normal) == 0:
			continue

		normal /= np.linalg.norm(normal)
		a, b, c = normal
		d = -np.dot(normal, sample_points[0])

		# calculate distances to the plane
		distances = np.abs(np.dot(points3d, normal) + d) / np.linalg.norm(normal)
		inliers = distances < threshold
		num_inliers = np.sum(inliers)

		# update the best plane if this one has more inliers
		if num_inliers > max_inliers:
			max_inliers = num_inliers
			best_plane = (a, b, c, d)

	return best_plane


def calc_ground_plane_normal(plane_coeffs):
	"""
	Extracts the normal vector from plane coefficients.

	:param plane_coeffs: Coefficients (a, b, c, d) of the plane equation.
	:return: Normalized normal vector (a, b, c).
	"""
	a, b, c, d = plane_coeffs
	normal = np.array([a, b, c])
	normal /= np.linalg.norm(normal)
	return normal


def calc_ground_plane_center(filtered_3d_inliers):
	centroid = np.mean(filtered_3d_inliers, axis=0)
	return centroid


def calc_gt_normal(
	panda_root, seq_num, rois, cam_ref_pt, plane_ctr_pt, 
	results_root,
	tf_id=0, lidar_sensor=None, vis=False, 
	save_vis=False, single_frame=False
):
	ground_truth = {}

	images, camera_K, pcs, cam_poses, lidar_poses = process_data(panda_root, seq_num, lidar_sensor)

	# iterate over images
	for i in range(len(images) - 1):
		img = images[i]
		pc = pcs[i]
		cam_pose = cam_poses[i]
		lidar_pose = lidar_poses[i]

		# inner_idxs are the indices in the original pc that are projected (verify this!)
		proj_pts_2d, cam_pts_3d, inner_idxs = geometry.projection(
			lidar_points=pc, 
			camera_data=img,
			camera_pose=cam_pose,
			camera_intrinsics=camera_K,
			filter_outliers=True
		)

		for roi_idx, roi in enumerate(rois):
			print(f"Processing: image {i}, ROI {roi}")
   
			filtered_2d, filtered_3d, roi_mask = filter_points_in_roi(proj_pts_2d, cam_pts_3d, roi)
			if filtered_3d.size == 0:
				print(f"[WARN] No points in ROI for image {i}. Skipping visualization and RANSAC.")
				continue

	 		# map the ROI mask back to the original point cloud indices
			roi_idxs = inner_idxs[roi_mask]
			# select the corresponding original points
			pc_og_filtered = pc[roi_idxs]
				

			
			# IMPORTANT: transform to ego frame after filtering, not before
			if tf_id == 0:
				pc_og_transformed = tf_pts_to_ego(
					pc_og_filtered, lidar_pose
				)
				frame = "ego"
			elif tf_id == 1:
				pc_og_transformed = tf_pts_to_first_pose(
					pc_og_filtered, lidar_pose, lidar_poses[0]
				)
				frame = "first"
			elif tf_id == 2:
				pc_og_transformed = pc_og_filtered
				frame = "world"
			else:
				raise ValueError("Invalid tf_id value in config. Choose from [0, 1, 2].")

			# RANSAC ground plane estimation (with outlier removal)
			lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
			inliers = lof.fit_predict(pc_og_transformed) > 0
			filtered_3d_inliers = pc_og_transformed[inliers]

			ground_plane_coeffs = calc_ground_plane_ransac(
				filtered_3d_inliers, threshold=0.01, max_iterations=100
			)
			if ground_plane_coeffs is None:
				print("[WARN] Could not compute ground plane!")
				continue
   
			ground_normal = calc_ground_plane_normal(ground_plane_coeffs)
			ground_normal = linalg.align_normal_ref_pt(
				ground_normal,
				ref_pt=np.array(cam_ref_pt),
				plane_ctr=np.array(plane_ctr_pt)
			)			
			plane_centroid = calc_ground_plane_center(filtered_3d_inliers)


			# get camera to LiDAR transformation matrix (rotation only)
			rot_cam_to_lidar = get_cam_tf_mat(cam_pose, lidar_poses[0])

			# visualize the results
			# vis_gt_normal.visualize_filtered_projection(
			# 	img, proj_pts_2d, cam_pts_3d, results_root, seq_num, i, 
			# 	show_img=True, vis=vis, save_vis=save_vis
			# )  # full projection without filtering

			# vis_gt_normal.visualize_filtered_projection(
	   		# 	img, filtered_2d, pc_og_transformed, results_root, seq_num,
			# 	i, show_img=True, vis=vis, save_vis=save_vis
			# )

			# gt_vis.plot_filtered_3d_points(
			# 	pc_og_transformed, results_root, seq_num, i, vis=vis,
			# 	save_vis=save_vis
			# )
			
			# gt_vis.plot_plane_with_normal(
			# 	pc_og_transformed, ground_plane_coeffs, ground_normal, 
			# 	results_root, seq_num, i, vis=vis, save_vis=save_vis
			# )

			# vis_gt_normal.plot_complex_visualization(
			# 	img, filtered_2d, pc_og_transformed, ground_plane_coeffs,
			# 	ground_normal, plane_centroid, results_root, seq_num, i,
			# 	vis, save_vis
			# )
		
			# save gt for current ROI
			ground_truth.setdefault(i, {})[roi_idx] = {
				"proj_pts_2d": filtered_2d.tolist(),
				"cam_pts_3d": pc_og_transformed.tolist(),
				"plane_coefficients": list(ground_plane_coeffs),
				"normal": ground_normal.tolist(),
				"centroid": plane_centroid.tolist(),
				"ref_frame": frame,
				"rot_cam_to_lidar": rot_cam_to_lidar.tolist()
			}

		if single_frame:
			break
	
	# save to disk
	output_path = os.path.join(results_root, seq_num, "gt", f"gt_data_{seq_num}.pkl")
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	data.save_pickle_data(ground_truth, output_path)


@hydra.main(version_base="1.3", config_path="../configs/gt", config_name="gt_normal_panda.yaml")
def main(cfg: DictConfig) -> None:
	"""
	Main entry point for ground truth generation.
	"""
	if cfg.panda_root == "wsl_root":
		panda_root = "/mnt/e/PandaSet"
	elif cfg.panda_root == "linux_root":
		panda_root = "/media/norbert/T7/PandaSet"
	else:
		raise ValueError("Invalid panda_root value in config.")
 
	gt_root = os.path.join(cfg.repo_root, "results", "gt_panda")
	os.makedirs(gt_root, exist_ok=True)

	# load data (pcs: list, np.ndarray inside list for each frame, access with pcs[frame_idx])
	images, _, _, _, _ = process_data(panda_root, cfg.seq_num, lidar_sensor=cfg.lidar_sensor)

	# handle ROIs
	if not cfg.roi or len(cfg.roi) == 0:
		default_roi = [(550, 430), (730, 430), (950, 720), (265, 720)]
		print("No ROI provided. Using default ROI:", default_roi)
		rois = [default_roi]
	else:
		rois = [list(map(tuple, roi.points)) for roi in cfg.roi]

	# resize ROI for current image size
	roi_img_src_size = cfg.img_src_size
	roi_img_dst_size = images[0].size
	rois = calculate_scaled_rois(rois, roi_img_src_size, roi_img_dst_size)

	calc_gt_normal(
		panda_root, cfg.seq_num, rois, cfg.cam_ref_pt, cfg.plane_ctr_pt,
		gt_root, 
  		tf_id=cfg.tf_id, lidar_sensor=cfg.lidar_sensor,
	 	vis=cfg.vis, save_vis=cfg.save_vis, single_frame=cfg.single_frame
	)

	print(f"Ground truth generation complete for sequence {cfg.seq_num}!")


if __name__ == "__main__":
	main()
