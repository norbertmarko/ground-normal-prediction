from pathlib import Path

from datetime import datetime
from omegaconf import DictConfig
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

import rich.table
from rich.console import Console

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import src.utils.data_utils as data_utils
import src.utils.linalg as linalg
import src.algos.homography.hg_funcs as hg_funcs
import src.eval.eval_norm as eval_norm
import src.eval.eval_vis as eval_vis


# TODO: add option to choose if you want to evaluate a single sequence or multiple sequences (glob for whole dataset)

def eval_exp_metrics(seq_num, gt_dir, exp_dir, ref_data_dirs, output_dir):
	print("Evaluating using experimental metrics...")
	(Path(output_dir) / seq_num).mkdir(parents=True, exist_ok=True)

	gt_data_path = Path(gt_dir) / seq_num / "gt" / f"gt_data_{seq_num}.pkl"
	exp_data_path = Path(exp_dir) / seq_num / f"eval_data_{seq_num}.pkl"
	ref_data_paths = [
		 Path(ref_data_dir) / seq_num / f"eval_data_{seq_num}.pkl" \
		 for ref_data_dir in ref_data_dirs
	]

	# load data
	gt_data = data_utils.load_pickle_data(gt_data_path)
	exp_data = data_utils.load_pickle_data(exp_data_path)
	# TODO: handle multiple reference data later
	ref_data_list = [data_utils.load_pickle_data(ref_data_path) for ref_data_path in ref_data_paths]
	ref_data = ref_data_list[0]

	# init data structures
	pitch_gt = []
	pitch_exp = []
	pitch_ref = []

	# metrics
	eval_seq_store_exp = {}
	eval_seq_store_ref = {}

	# cycle trough the sequence
	loop_length = len(exp_data.keys())
	with tqdm(total=loop_length, desc=f"Evaluating sequence {seq_num}", unit=" frame") as pbar:
		for i in range( len(exp_data.keys()) ):
		
			
			# get the ground truth data for the frame
			normal_gt = np.array(gt_data[i][0]["normal"])
			centroid_gt = np.array(gt_data[i][0]["centroid"])
			cam_pts_3d_gt = np.array(gt_data[i][0]["cam_pts_3d"])
			plane_coeffs_gt = np.array(gt_data[i][0]["plane_coefficients"])
			R_cam_to_lidar = np.array(gt_data[i][0]["rot_cam_to_lidar"])
		
			# get data for the frame
			normal_exp = exp_data[i]["normal"]
			normal_ref = ref_data[i]["normal"]

			# process the data

			# already rotated to LiDAR frame!
			# normal_gt = linalg.rotate_vector(normal_gt, R_cam_to_lidar)
			# normal_exp = linalg.rotate_vector(normal_exp, R_cam_to_lidar)
			# normal_ref = linalg.rotate_vector(normal_ref, R_cam_to_lidar)

			normal_exp = linalg.align_normal_ref_vec(normal_exp, normal_gt)
			normal_ref = linalg.align_normal_ref_vec(normal_ref, normal_gt)

			pitch_gt_curr = hg_funcs.calculate_pitch_from_normal(normal_gt)
			pitch_exp_curr = hg_funcs.calculate_pitch_from_normal(normal_exp)
			pitch_ref_curr = hg_funcs.calculate_pitch_from_normal(normal_ref)

			# save data for sequence evaluation (pitch)
			pitch_gt.append( pitch_gt_curr )
			pitch_exp.append( pitch_exp_curr )
			pitch_ref.append( pitch_ref_curr )

			eval_example_store_exp = {
				"idx": i,
				"normal": normal_exp,
				"normal_gt": normal_gt,
				"pitch": pitch_exp_curr,
				"pitch_gt": pitch_gt_curr,
			}
			eval_example_store_ref = {
				"idx": i,
				"normal": normal_ref,
				"normal_gt": normal_gt,
				"pitch": pitch_ref_curr,
				"pitch_gt": pitch_gt_curr,
			}

			results_exp_curr = eval_norm.eval_example_paper(
				eval_example_store_exp, print_results=False
			)
			results_ref_curr = eval_norm.eval_example_paper(
				eval_example_store_ref, print_results=False
			)

			# save data for sequence evaluation (metrics)
			eval_seq_store_exp[i] = {
				"norm_angle_deg": results_exp_curr["norm_angle_deg"],
				"signed_norm_angle_deg": results_exp_curr["signed_norm_angle_deg"],
				"pitch_error_deg": results_exp_curr["pitch_error_deg"],
				"signed_pitch_error_deg": results_exp_curr["signed_pitch_error_deg"],
			}
			eval_seq_store_ref[i] = {
				"norm_angle_deg": results_ref_curr["norm_angle_deg"],
				"signed_norm_angle_deg": results_ref_curr["signed_norm_angle_deg"],
				"pitch_error_deg": results_ref_curr["pitch_error_deg"],
				"signed_pitch_error_deg": results_ref_curr["signed_pitch_error_deg"],
			}

			# plot normals
			# eval_vis.plot_complex_normal(
			# 	cam_pts_3d_gt, plane_coeffs_gt, centroid_gt,
			# 	normal_exp, normal_gt, normal_ref,
			# 	output_dir, seq_num, i, vis=False, save_vis=False
			# )

			pbar.update(1)

	# eval sequence
	timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	results_exp = eval_norm.eval_sequence_paper(
		 eval_seq_store_exp, output_dir, seq_num, timestamp, name="exp", print_results=True
	)
	results_ref = eval_norm.eval_sequence_paper(
		eval_seq_store_ref, output_dir, seq_num, timestamp, name="ref", print_results=True
	)

	# print eval results
	console = Console()
	table = rich.table.Table(title="Sequence Evaluation Results", style="bold green")

	table.add_column("Model", justify="left", style="bold yellow", no_wrap=True)
	table.add_column("Avg Normal Error (°)", justify="center", style="magenta", no_wrap=True)
	table.add_column("Avg Signed Normal Error (°)", justify="center", style="cyan", no_wrap=True)
	table.add_column("Avg Pitch Error (°)", justify="center", style="green", no_wrap=True)
	table.add_column("Avg Signed Pitch Error (°)", justify="center", style="blue", no_wrap=True)

	table.add_row(
		"Ours",
		f"{results_exp['avg_norm_angle_deg']:.3f}",
		f"{results_exp['avg_signed_norm_angle_deg']:.3f}",
		f"{results_exp['avg_pitch_error_deg']:.3f}",
		f"{results_exp['avg_signed_pitch_error_deg']:.3f}",
	)
	table.add_row(
		"SOTA",
		f"{results_ref['avg_norm_angle_deg']:.3f}",
		f"{results_ref['avg_signed_norm_angle_deg']:.3f}",
		f"{results_ref['avg_pitch_error_deg']:.3f}",
		f"{results_ref['avg_signed_pitch_error_deg']:.3f}",
	)

	console.print(table)

	# save eval results
	df = pd.DataFrame({
		"Model": ["Ours", "SOTA"],
		"Average Normal Error (degrees)": [results_exp['avg_norm_angle_deg'], results_ref['avg_norm_angle_deg']],
		"Average Signed Normal Error (degrees)": [results_exp['avg_signed_norm_angle_deg'], results_ref['avg_signed_norm_angle_deg']],
		"Average Pitch Error (degrees)": [results_exp['avg_pitch_error_deg'], results_ref['avg_pitch_error_deg']],
		"Average Signed Pitch Error (degrees)": [results_exp['avg_signed_pitch_error_deg'], results_ref['avg_signed_pitch_error_deg']],
	})
	df.to_csv(
		Path(output_dir) / seq_num / f"eval_results_{seq_num}_{timestamp}.csv",
		index=False,
		float_format="%.3f"
	)
	df.to_latex(
		Path(output_dir) / seq_num / f"eval_results_{seq_num}_{timestamp}.tex",
		index=False,
		float_format="%.3f"
	)


	# plots (last, so they do not block the console)
	eval_vis.plot_pitch(pitch_gt, pitch_exp, pitch_ref, output_dir, seq_num, vis=False, save_vis=True)


def eval_paper_metrics(seq_num, gt_dir, exp_dir, ref_data_dirs, output_dir):
	print("Evaluating using reference papers' metrics...")

	gt_data_path = Path(gt_dir) / seq_num / "gt" / f"gt_data_{seq_num}.pkl"
	exp_data_path = Path(exp_dir) / seq_num / f"eval_data_{seq_num}.pkl"
	
	ref_data_paths = [
		 Path(ref_data_dir) / seq_num / f"eval_data_{seq_num}.pkl" \
		 for ref_data_dir in ref_data_dirs
	]

	# load data

	# calculate metrics and generate plots

	# print eval results

	# save eval results


@hydra.main(version_base="1.3", config_path="../configs/eval", config_name="eval_panda.yaml")
def run(cfg: DictConfig):
	"""
	Main entry point for evaluation.
	"""
	print(f"Running evaluation for sequence {cfg.seq_num} ...")
	if 'exp' in cfg.eval_types:
		 eval_exp_metrics(cfg.seq_num, cfg.gt_dir, cfg.exp_dir, cfg.ref_data_dirs, cfg.output_dir)
	if 'paper' in cfg.eval_types:
		eval_paper_metrics(cfg.seq_num, cfg.gt_dir, cfg.exp_dir, cfg.ref_data_dirs, cfg.output_dir)

if __name__ == '__main__':
	run()