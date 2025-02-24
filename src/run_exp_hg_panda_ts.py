import os
import argparse
import warnings

from datetime import datetime
import cv2
import numpy as np
from omegaconf import DictConfig
import hydra

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import src.data_funcs.panda_funcs as panda
import src.algos.homography.hg_funcs as hg_funcs
import src.algos.homography.hg_decomp as hg_decomp
import src.algos.homography.hg_vis as hg_vis
import src.utils.data_utils as data_utils
import src.utils.linalg as linalg
import src.eval.eval_norm as eval_norm

import src.utils.time_series as ts


def calc_homography_interframe(img1, img2, roi_pts, invert_direction=False):
	"""
	Calculate homography between two consecutive frames.

	Parameters:
		img1 (numpy.ndarray): first image (t-1).
		img2 (numpy.ndarray): second image (t).
		roi_pts (list of tuples): List of four (x, y) tuples defining the ROI polygon.

	Returns:
		homography_matrix (numpy.ndarray): The calculated homography matrix.
		mask (numpy.ndarray): Mask of inliers used in homography calculation.
		keypoints1, keypoints2 (list of cv2.KeyPoint): Keypoints from both images.
		matches (list of cv2.DMatch): Matched keypoints.
		src_pts, dst_pts (numpy.ndarray): Source and destination points used for homography.
	"""
	# Generate ROI masks
	mask1 = hg_funcs.generate_roi_mask(img1, roi_pts)
	mask2 = hg_funcs.generate_roi_mask(img2, roi_pts)

	# Detect keypoints and descriptors
	detector = 'SIFT'
	keypoints1, descriptors1 = hg_funcs.detect_keypoints(img1, detector, mask=mask1)
	keypoints2, descriptors2 = hg_funcs.detect_keypoints(img2, detector, mask=mask2)

	# Check if descriptors are valid
	if descriptors1 is None or descriptors2 is None:
		warnings.warn("Descriptors not found for one of the images.", UserWarning)
		return None, None, keypoints1, keypoints2, None, None, None

	# Match keypoints between images
	matches = hg_funcs.match_features(descriptors1, descriptors2, detector)

	# Filter and select keypoints for homography
	pts_set_1, pts_set_2 = hg_funcs.filter_matches(matches, keypoints1, keypoints2, num_points=100)

	# vheck if we have enough points to compute a homography (minimum is 4)
	if pts_set_1 is None or pts_set_2 is None or len(pts_set_1) < 4 or len(pts_set_2) < 4:
		warnings.warn("Not enough points for homography calculation.", UserWarning)
		return None, None, keypoints1, keypoints2, matches, pts_set_1, pts_set_2

	if invert_direction:
		src_pts, dst_pts = pts_set_2, pts_set_1
	else:
		src_pts, dst_pts = pts_set_1, pts_set_2

	# Calculate homography matrix
	hg_mat, mask = hg_funcs.calc_homography(src_pts, dst_pts, method=cv2.RANSAC, threshold=2.0)

	return hg_mat, mask, keypoints1, keypoints2, matches, src_pts, dst_pts


def process_data(panda_root, seq_num):
	"""
	Load and process data for a single sequence.
	"""
	dataset, seq_list = panda.read_pandaset(panda_root)
	cam_obj = panda.load_panda_seq_cam(dataset, seq_list, seq_num)
	gps_obj = panda.load_panda_seq_gps(dataset, seq_list, seq_num)
	_, abs_rot_mats, pitch_xyz, vels = panda.parse_panda_seq_poses(cam_obj, gps_obj)
	camera_K = panda.get_panda_seq_intrinsics(cam_obj)
	images = panda.get_panda_seq_images(cam_obj, pil=False)
	return images, camera_K, pitch_xyz


def decompose(hg_mat, intrinsics, last_n_norms, handle_outliers=False, debug=False):
	"""
	Denormalize, decompose and extract normal vector from homography matrix
	along with the euler angles (ego-motion).
	"""
	hg_decomp.normalize_homography(hg_mat, intrinsics)
	solution, all_sols = hg_decomp.decompose_homography(
		hg_mat, intrinsics, last_n_norms, handle_outliers, debug=debug,
	)
	yaw, pitch, roll = hg_decomp.tf_rot_mat_to_euler(solution["rotation"])
	return solution, all_sols, yaw, pitch, roll


@hydra.main(version_base="1.3", config_path="./configs", config_name="exp_hg_panda_ts.yaml")
def main(cfg: DictConfig):
	"""
	Main entry point for homography-based experiment pipeline.
	"""
	if cfg.panda_root == "wsl_root":
		panda_root = "/mnt/e/PandaSet"
	elif cfg.panda_root == "linux_root":
		panda_root = "/media/norbert/T7/PandaSet"
	else:
		raise ValueError("Invalid panda_root value in config.")

	output_dir = os.path.join(cfg.output_dir)
	os.makedirs(output_dir, exist_ok=True)
	eval_output_path = os.path.join(output_dir, cfg.seq_num, f"eval_data_{cfg.seq_num}.pkl")
	os.makedirs(os.path.join(output_dir, cfg.seq_num), exist_ok=True)
 
	vis = cfg.vis
	save_vis = cfg.save_vis


	# load data
	images, camera_K, pitch_xyz = process_data(panda_root, cfg.seq_num)
	if len(images) < 2:
		print("Not enough images in the sequence to compute homography.")
		return

	# handle ROIs
	if not cfg.roi or len(cfg.roi) == 0:
		default_roi = [(550, 430), (730, 430), (950, 720), (265, 720)]
		print("No ROI provided. Using default ROI:", default_roi)
		rois = [default_roi]
	else:
		rois = [list(map(tuple, roi.points)) for roi in cfg.roi]
	# resize ROI for current image size
	roi_img_src_size = cfg.img_src_size
	roi_img_dst_size = (images[0].shape[1], images[0].shape[0])
	rois = hg_funcs.calculate_scaled_rois(rois, roi_img_src_size, roi_img_dst_size)


	# load ground truth
	gt_dir = os.path.join(cfg.gt_dir, f"{cfg.seq_num}", "gt")
	gt_path = os.path.join(gt_dir, f"gt_data_{cfg.seq_num}.pkl")
	gt_data = data_utils.load_pickle_data(gt_path)


	# init data stores
	pitch_vals = []
	pitch_vals_gt = []
	last_n_norms = {"normals": [], "rotations": [], "translations": []}
	MAX_NORM_HISTORY = 5
	data_store = {}
	eval_data_store = {}


	running_normal_calc = ts.RunningCalculator(
		strategy=ts.BasicKalmanFilterStrategy(
			process_noise=1.0, measurement_noise=1.0, huber_delta=2.0, burn_in=5
		)
	)
	running_pitch_calc = ts.RunningCalculator(
		strategy=ts.BasicKalmanFilterStrategy(
			process_noise=1.0, measurement_noise=1.0, huber_delta=2.0, burn_in=5
		)
	)

	# adjust the values to the start value (if enabled)
	if cfg.accumulate_values:
		first_gt = gt_data[0][0]  # first frame, first ROI
		accum_rotation = np.eye(3)  # start with identity rotation
		accum_normal =  np.array(first_gt["normal"])
		accum_pitch = hg_funcs.calculate_pitch_from_normal(accum_normal)

		print(f"Initialized accum_pitch: {accum_pitch}")
		print(f"Initialized accum_normal: {accum_normal}")


	# iterate over consecutive image pairs
	for i in range(len(images) - 1):
		img1 = images[i]
		img2 = images[i+1]

		# enumerate ROIs here also so we can load the correct ground truth
		for roi_idx, roi in enumerate(rois):
			print(f"Processing: image {i} and {i+1}, ROI {roi}")

			# process images (resize) - no change for now
			img1 = hg_funcs.proc_img(img1, roi_img_dst_size)
			img2 = hg_funcs.proc_img(img2, roi_img_dst_size)

			# calculate homography
			hg_mat, mask, keypoints1, keypoints2, matches, src_pts, dst_pts = calc_homography_interframe(
				img1, img2, roi, invert_direction=cfg.invert_direction
			)
			
			# ---- #
			# Check if homography was successfully computed
			if hg_mat is None:
				print(f"[WARN] Homography computation failed for images {i}-{i+1}, ROI {roi_idx}")
				# Fallback: if we have a previous valid solution, use it;
				# otherwise, use default values (identity rotation, zero translation, default normal).
				if last_valid_solution is None:
					solution = {
						"rotation": np.eye(3),
						"translation": np.zeros(3),
						"normal": np.array([0.0, 0.0, 1.0])
					}
				else:
					solution = last_valid_solution
				all_sols = [solution]
				yaw, pitch, roll = hg_decomp.tf_rot_mat_to_euler(solution["rotation"])
				# set fallback values for mask and matches to avoid visualization errors.
				mask = np.array([])   # or np.array([]) if you prefer an array
				matches = []
				# provide a valid placeholder for the homography matrix.
				hg_mat = np.eye(3, dtype=np.float32)
			else:
			
			# ---- #
				# decompose homography (normalize before), also extract euler angles
				solution, all_sols, yaw, pitch, roll = decompose(
					hg_mat, camera_K, last_n_norms, cfg.handle_outliers, debug=False
				)

				# Save the last valid solution for fallback usage (also new)
				last_valid_solution = solution

			# extract results
			normal = solution["normal"]

			# update the last n normals
			last_n_norms["normals"].append(normal)
			last_n_norms["rotations"].append(solution["rotation"])
			last_n_norms["translations"].append(solution["translation"])
			
			if len(last_n_norms["normals"]) > MAX_NORM_HISTORY:		
				for key in last_n_norms:
					last_n_norms[key].pop(0)


			# access gt data (current frame pair)
			proj_pts_2d_gt = np.array(gt_data[i][roi_idx]["proj_pts_2d"])
			cam_pts_3d_gt = np.array(gt_data[i][roi_idx]["cam_pts_3d"])
			plane_coeffs_gt = np.array(gt_data[i][roi_idx]["plane_coefficients"])
			normal_gt = np.array(gt_data[i][roi_idx]["normal"])
			centroid_gt = np.array(gt_data[i][roi_idx]["centroid"])
			ref_frame = gt_data[i][roi_idx]["ref_frame"]
   
			# Load the camera-to-LiDAR rotation matrix from the ground truth
			R_cam_to_lidar = np.array(gt_data[i][roi_idx]["rot_cam_to_lidar"])

			# # post-processing functions
			# R_cam_to_lidar = hg_funcs.get_cam_to_lidar_rot(do_rot_z=False, rot_z_deg=180)

			# post-process chosen normal vector
			normal = linalg.rotate_vector(normal, R_cam_to_lidar)
			normal = linalg.align_normal_ref_vec(normal, normal_gt)

			# post-process normal candidates
			if cfg.plot_norm_candidates:
				norm_candidates =[]
				for sol in all_sols:
					norm_candidate = sol["normal"]
					norm_candidate = linalg.rotate_vector(norm_candidate, R_cam_to_lidar)
					print(f"Norm candidate: {norm_candidate}")
					# TODO: also try without aligning the normal
					norm_candidate = linalg.align_normal_ref_vec(norm_candidate, normal_gt) 
					norm_candidates.append(norm_candidate)
			else:
				norm_candidates = None

			# calculate pitch
			pitch_from_norm = hg_funcs.calculate_pitch_from_normal(normal)
			pitch_from_norm_gt = hg_funcs.calculate_pitch_from_normal(normal_gt)

			# (post-process sequence) inform time series about the current values
			
			# Update the time-series (Kalman filter) calculators if a new measurement exists.
			# For frames with failed homography, you might choose to skip this update,
			# which means the filter will simply return its last (predicted) value.
			if hg_mat is not None:			
				running_normal_calc.update(normal)
				normal = running_normal_calc.get_current_value()

				running_pitch_calc.update(pitch_from_norm)
				pitch_from_norm = running_pitch_calc.get_current_value()
			else:
				# No update; simply use the current (predicted) value
				normal = running_normal_calc.get_current_value()
				pitch_from_norm = running_pitch_calc.get_current_value()

			
			# evaluation for a single example
			signed_norm_angle_deg, pitch_error_deg = eval_norm.eval_example(
				normal, normal_gt,
				pitch_from_norm, pitch_from_norm_gt, 
				i, print_results=True
			)


			# (post-process sequence) accumulation / adjustment to start value
			if cfg.accumulate_values:
				# normal
				accum_rotation = accum_rotation @ solution["rotation"]
				# (optional) re-orthogonalize to prevent drift
				U, _, Vt = np.linalg.svd(accum_rotation)
				accum_rotation = U @ Vt

				# update the accumulated normal by applying to init normal
				accum_normal = accum_rotation @ first_gt["normal"]
				accum_normal /= np.linalg.norm(accum_normal)

				# calc accumulated pitch from the accumulated normal
				accum_pitch = hg_funcs.calculate_pitch_from_normal(accum_normal)

				# update the values for the current frame pair
				normal = accum_normal
				pitch_from_norm = accum_pitch


			# store results
			pitch_vals.append(pitch_from_norm)
			pitch_vals_gt.append(pitch_from_norm_gt)

			data_store[(i, i+1)] = {
				"homography_matrix": hg_mat,
				"normal": normal,
				"normal_gt": normal_gt,
				"pitch": pitch_from_norm,
				"pitch_gt": pitch_from_norm_gt,
				"signed_norm_angle_deg": signed_norm_angle_deg,
				"pitch_error_deg": pitch_error_deg
			}

			# store evaluation data
			eval_data_store[i] = {
				"normal": normal,
				"normal_gt": normal_gt,
			}
   

			# # visualizations
			# hg_vis.vis_matches(
			# 	img1, img2, keypoints1, keypoints2, matches, roi,
			# 	output_dir, cfg.seq_num, i, vis, save_vis
			# )

			# hg_vis.vis_ransac(
			# 	img1, img2, keypoints1, keypoints2, matches, mask,
			# 	output_dir, cfg.seq_num, i, vis, save_vis
			# )

			# warped_img = hg_vis.vis_warp_if(
			# 	img1, img2, hg_mat, cfg.invert_direction,
			# 	output_dir, cfg.seq_num, i, vis, save_vis
			# )

			# blended_img = hg_vis.vis_blend(
			# 	img1, img2, hg_mat, keypoints1, keypoints2, matches,
			# 	roi, cfg.invert_direction, 0.5, 
			# 	output_dir, cfg.seq_num, i, vis, save_vis
			# )

			# # baseline plots
			# hg_vis.plot_complex_normal(
			# 	blended_img, hg_mat, normal,
			# 	cam_pts_3d_gt, plane_coeffs_gt, centroid_gt, normal_gt,
			# 	signed_norm_angle_deg,
			# 	output_dir, cfg.seq_num, i, vis, save_vis,
			# 	normal_candidates=norm_candidates
			# )

			# hg_vis.plot_complex_pitch(
			# 	blended_img, hg_mat, pitch_vals,
			# 	pitch_vals_gt, pitch_error_deg,
			# 	output_dir, cfg.seq_num, i, vis, save_vis
			# )			

		if cfg.single_frame:
			break


	# TODO: implement pitch sensitivity analysis after implementing pitch calculation
	# plot pitch sensitivity analysis
	hg_vis.plot_hg_param_sensitivity(
		pitch_vals, pitch_vals_gt,
		[data_store[(i, i+1)]["homography_matrix"] for i in range(len(images)-1)],
		output_dir, cfg.seq_num, vis, save_vis
	)
	
	timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	eval_norm.eval_sequence(data_store, output_dir, cfg.seq_num, timestamp, print_results=True)

	# save data for evaluation
	data_utils.save_pickle_data(eval_data_store, eval_output_path)

if __name__ == '__main__':
	main()
