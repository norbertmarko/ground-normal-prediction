import argparse
import os
import shutil
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
from omegaconf import DictConfig
import hydra

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pandaset import geometry
import src.algos.homography.hg_vis as hg_vis
import src.utils.data_utils as data_utils
import src.utils.linalg as linalg
import src.algos.homography.hg_funcs as hg_funcs
from scipy.spatial.transform import Rotation as R

import src.data_funcs.panda_funcs as panda_funcs

from src._ref.ground_normal_filter.src.filter import GroundNormalFilterIEKF
from src._ref.ground_normal_filter.src.visualizer import Visualization
import src.eval.eval_norm as eval_norm


# TODO: import later (move func from gen_gt_normal_panda.py first)
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


# TODO: import from linalg.py later
def rotate_vector(vec, R):
	"""Rotate a vector using a rotation matrix R, then normalize it."""
	vec_tf = R @ vec
	vec_tf /= np.linalg.norm(vec_tf)
	return vec_tf


@hydra.main(version_base="1.3", config_path="./configs", config_name="ref_gnf_panda.yaml")
def eval(cfg: DictConfig):
	
	# read reference data
	dataset, seq_list = panda_funcs.read_pandaset(cfg.panda_root)
	
	# params
	pandaset_d = None  # distortion coefficients (not required)
	pandaset_wh = (1280, 720)
	
	# create output directories
	os.makedirs(cfg.output_dir, exist_ok=True)
	results_dir = os.path.join(cfg.output_dir, cfg.seq_num)
	os.makedirs(results_dir, exist_ok=True)
	og_vis_dir = os.path.join(results_dir, "og_vis")
	os.makedirs(og_vis_dir, exist_ok=True)
	eval_output_path = os.path.join(results_dir, f"eval_data_{cfg.seq_num}.pkl")


	cam_obj = panda_funcs.load_panda_seq_cam(dataset, seq_list, cfg.seq_num)
	gps_obj = panda_funcs.load_panda_seq_gps(dataset, seq_list, cfg.seq_num)
	rel_timestamps, abs_rot_mats, pitch_xyz, vels \
				  = panda_funcs.parse_panda_seq_poses(cam_obj, gps_obj)
	rel_rot_mats = panda_funcs.get_panda_seq_rel_rot_mats(abs_rot_mats)

	camera_K = panda_funcs.get_panda_seq_intrinsics(cam_obj)

	# can be sliced to get the actual images (numpy array)
	images = panda_funcs.get_panda_seq_images(cam_obj)

	print(
		f"Camera intrinsic matrix for PandaSet sequence {cfg.seq_num}:\n"
		f"{np.array2string(camera_K, formatter={'float_kind':lambda x: f'{x:.4f}'})}"
	)


	# comparison loading
	cam_poses = cam_obj.poses[:]
	lidar_obj = panda_funcs.load_panda_seq_lidar(dataset, seq_list, cfg.seq_num)
	lidar_poses = panda_funcs.parse_panda_seq_lidar_poses(lidar_obj, sensor=0)

	# import ground truth normal vectors
	gt_dir = os.path.join(cfg.gt_path, f"{cfg.seq_num}", "gt")
	gt_path = os.path.join(gt_dir, f"gt_data_{cfg.seq_num}.pkl")
	gt_data = data_utils.load_pickle_data(gt_path)
   
   
	dataset.unload(cfg.seq_num)


	# initialize objects
	pitch_vals = []
	pitch_vals_gt = []
	data_store = {}
	eval_data_store = {}


	gnf = GroundNormalFilterIEKF()
	vis = Visualization(K=camera_K, d=pandaset_d, input_wh=pandaset_wh)
	
	# main evaluation loop
	loop_length = len(rel_rot_mats)
	with tqdm(total=loop_length, desc="Processing Images", unit="img") as pbar:
		
		for idx in range(loop_length):

			# get camera to LiDAR transformation matrix (rotation only)
			cam_pose = cam_poses[idx]
			rot_cam_to_lidar = get_cam_tf_mat(cam_pose, lidar_poses[0])
			

			relative_so3 = rel_rot_mats[idx]
			# print(relative_so3)
			# break
			compensation_se3 = gnf.update(relative_so3)
			compensation_so3 = compensation_se3[:3, :3]


			np.set_printoptions(precision=3, suppress=True)


			# return (extract) ground normal vector
			ground_normal_vec = compensation_se3[:3, 1]


			# access gt data (current frame)
			proj_pts_2d_gt = np.array(gt_data[idx][0]["proj_pts_2d"])
			cam_pts_3d_gt = np.array(gt_data[idx][0]["cam_pts_3d"])
			plane_coeffs_gt = np.array(gt_data[idx][0]["plane_coefficients"])
			normal_gt = np.array(gt_data[idx][0]["normal"])
			centroid_gt = np.array(gt_data[idx][0]["centroid"])
			ref_frame = gt_data[idx][0]["ref_frame"]


			normal = rotate_vector(ground_normal_vec, rot_cam_to_lidar)
			# postprocess normal vector
			normal = linalg.align_normal_ref_pt(
	   			normal,
				np.array([0, 0, 1]),
				np.array([0, 0, 0])
		  	)
			pitch_from_norm = hg_funcs.calculate_pitch_from_normal(normal)
			pitch_from_norm_gt = hg_funcs.calculate_pitch_from_normal(normal_gt)
   
			# store results
			pitch_vals.append(pitch_from_norm)
			pitch_vals_gt.append(pitch_from_norm_gt)
   

			# evaluation for a single example
			signed_norm_angle_deg, pitch_error_deg = eval_norm.eval_example(
				normal, normal_gt,
				pitch_from_norm, pitch_from_norm_gt, 
				idx, print_results=True
			)


			# dummy data (calc or not needed)
			dummy_hg_mat = np.eye(3)

			data_store[idx] = {
				"homography_matrix": dummy_hg_mat,
				"normal": normal,
				"normal_gt": normal_gt,
				"pitch": pitch_from_norm,
				"pitch_gt": pitch_from_norm_gt,
				"signed_norm_angle_deg": signed_norm_angle_deg,
				"pitch_error_deg": pitch_error_deg
			}


			# store evaluation results
			eval_data_store[idx] = {
				"normal": normal,
				"normal_gt": normal_gt,
			}
   
			image = images[idx]
			combined_image = vis.get_frame(image, compensation_so3)
			output_path = os.path.join(og_vis_dir, f"{idx:06d}_{cfg.seq_num}.jpg")
			cv2.imwrite(output_path, combined_image)


			# save_vis = False
			# # comparison plotting
			# hg_vis.plot_complex_normal(
			# 	image, dummy_hg_mat, normal,
			# 	cam_pts_3d_gt, plane_coeffs_gt, centroid_gt, normal_gt,
			# 	signed_norm_angle_deg,
			# 	cfg.output_dir, cfg.seq_num, idx, False, save_vis,
			# 	normal_candidates=None
			# )

			# hg_vis.plot_complex_pitch(
			# 	image, dummy_hg_mat, pitch_vals,
			# 	pitch_vals_gt, signed_norm_angle_deg,
			# 	cfg.output_dir, cfg.seq_num, idx, False, save_vis
			# )

			pbar.update(1)

	timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	eval_norm.eval_sequence(data_store, results_dir, cfg.seq_num, timestamp, print_results=True)

	data_utils.save_pickle_data(eval_data_store, eval_output_path)


if __name__ == '__main__':
	eval()
