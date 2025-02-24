import os
import pickle
import rich.table
from rich.console import Console
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import src.eval.metrics.metrics_norm as metrics_norm
import numpy as np

def eval_example_paper(data_store, print_results=True):
    # Calculate the absolute normal vector error
    norm_angle_deg = metrics_norm.calc_norm_vec_error(
        data_store["normal"], data_store["normal_gt"]
    )
    # Calculate the signed normal vector error (using the reference normal internally)
    signed_norm_angle_deg = metrics_norm.calc_signed_norm_vec_error(
        data_store["normal"], data_store["normal_gt"], normal_ref=[1, 0, 0]
    )
    # Calculate the pitch error (absolute)
    pitch_error_deg = metrics_norm.calc_pitch_error(
        data_store["pitch"], data_store["pitch_gt"], signed=False
    )
    # Calculate the signed pitch error
    signed_pitch_error_deg = metrics_norm.calc_pitch_error(
        data_store["pitch"], data_store["pitch_gt"], signed=True
    )

    results = {
        "norm_angle_deg": norm_angle_deg,
        "signed_norm_angle_deg": signed_norm_angle_deg,
        "pitch_error_deg": pitch_error_deg,
        "signed_pitch_error_deg": signed_pitch_error_deg,
    }
    return results


def eval_sequence_paper(data_store, save_path, seq_num, timestamp, name, print_results=True):
    print(f"Evaluating sequence (for {name} metrics)...")
    
    # Collect per-frame metrics (if present)
    norm_angles = [
        d["norm_angle_deg"] for d in data_store.values() if "norm_angle_deg" in d
    ]
    signed_norm_angles = [
        d["signed_norm_angle_deg"] for d in data_store.values() if "signed_norm_angle_deg" in d
    ]
    pitch_errors = [
        d["pitch_error_deg"] for d in data_store.values() if "pitch_error_deg" in d
    ]
    signed_pitch_errors = [
        d["signed_pitch_error_deg"] for d in data_store.values() if "signed_pitch_error_deg" in d
    ]
    
    if not norm_angles or not pitch_errors:
        print("Required metric values not found. Evaluation skipped.")
        return

    # Compute mean values for each metric
    avg_norm_angle = sum(norm_angles) / len(norm_angles)
    avg_signed_norm_angle = sum(signed_norm_angles) / len(signed_norm_angles)
    avg_pitch_error = sum(pitch_errors) / len(pitch_errors)
    avg_signed_pitch_error = sum(signed_pitch_errors) / len(signed_pitch_errors)

    results = {
        "avg_norm_angle_deg": avg_norm_angle,
        "avg_signed_norm_angle_deg": avg_signed_norm_angle,
        "avg_pitch_error_deg": avg_pitch_error,
        "avg_signed_pitch_error_deg": avg_signed_pitch_error,
    }
    return results



def eval_example(normal, ref_normal, pitch, ref_pitch, i, print_results=True):
	signed_norm_angle_deg = metrics_norm.calc_signed_norm_vec_error(normal, ref_normal, normal_ref=[1, 0, 0])
	pitch_error_deg = metrics_norm.calc_pitch_error(pitch, ref_pitch, signed=False)
	if print_results:
		show_eval_example_results(
			i, normal, ref_normal, signed_norm_angle_deg, 
			pitch, ref_pitch, pitch_error_deg
		)
	return signed_norm_angle_deg, pitch_error_deg


def eval_sequence(eval_data_store, save_path, seq_num, timestamp, print_results=True):
	print("Evaluating sequence...")

	signed_angles = [eval_data["signed_norm_angle_deg"]
		for eval_data in eval_data_store.values()
		if "signed_norm_angle_deg" in eval_data
	]
	if not signed_angles:
		print("No signed_norm_angle_deg values found. Evaluation skipped.")
		return
	pitch_errors = [eval_data["pitch_error_deg"]
		for eval_data in eval_data_store.values()
		if "pitch_error_deg" in eval_data
	]
	if not pitch_errors:
		print("No pitch_error_deg values found. Evaluation skipped.")
		return
	
	avg_signed_angle = sum(signed_angles) / len(signed_angles)
	avg_pitch_error = sum(pitch_errors) / len(pitch_errors)
	ref_pitch = np.array([data["pitch_gt"] for data in eval_data_store.values()])
	homography_params = np.array([data["homography_matrix"].flatten() for data in eval_data_store.values()])

	correlations = metrics_norm.calc_correlation(ref_pitch, homography_params)
	dtw_distances = metrics_norm.calc_dtw(ref_pitch, homography_params)

	print("Correlations:", correlations)
	print("DTW Distances:", dtw_distances)

	if print_results:
		show_eval_sequence_results(
			avg_signed_angle, avg_pitch_error, correlations, dtw_distances
		)
	save_metric_results(eval_data_store, save_path, seq_num, timestamp)


def save_metric_results(data, save_dir, seq_num, timestamp):
	"""
	Save metric results to disk in pickle format.

	:param data: Dictionary containing metric results.
	:param save_path: Path to save the pickle file.
	"""
	save_path = os.path.join(save_dir, seq_num, "eval", f"eval_data_{timestamp}.pkl")
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	with open(save_path, "wb") as f:
		pickle.dump(data, f)
	print(f"Metric results saved to {save_path}")


def load_metric_results(file_path):
	"""
	Load metric results from a pickle file.

	:param file_path: Path to the pickle file.
	:return: Dictionary containing metric results.
	"""
	with open(file_path, "rb") as f:
		data = pickle.load(f)
	print(f"Metric results loaded from {file_path}")
	return data


def show_eval_sequence_results(avg_signed_angle, avg_pitch_error, correlations, dtw_distances):
	"""
	Display the evaluation results in a styled table.
	"""
	console = Console()
	table = rich.table.Table(title="Sequence Evaluation Results", style="bold green")
	
	table.add_column("Parameter", justify="left", style="cyan", no_wrap=True)
	table.add_column("Value", justify="center", style="magenta")
	
	table.add_row("Average Signed Angle (degrees)", f"{avg_signed_angle:.2f}")
	table.add_row("Average Pitch Error (degrees)", f"{avg_pitch_error:.2f}")
	
	console.print(table)

	# Table for Correlation and DTW distances for each parameter (h1 to h9)
	param_table = rich.table.Table(title="Parameter-wise Correlation and DTW", style="bold green")
	param_table.add_column("Parameter", justify="left", style="cyan", no_wrap=True)
	param_table.add_column("Correlation", justify="center", style="magenta")
	param_table.add_column("DTW Distance", justify="center", style="magenta")
	
	for i in range(9):  # For parameters h1 to h9 (index 0 to 8)
		param_table.add_row(f"h{i+1}", f"{correlations[i]:.3f}", f"{dtw_distances[i]:.3f}")
	
	console.print(param_table)


def show_eval_example_results(
		i, normal, ref_normal, signed_norm_angle_deg,
		pitch, ref_pitch, pitch_error_deg
):
	"""
	Display the evaluation results in a styled table.
	"""
	console = Console()
	table = rich.table.Table(title=f"Example {i} - {i+1}", style="bold green")
	
	table.add_column("Parameter", justify="left", style="cyan", no_wrap=True)
	table.add_column("Value", justify="center", style="magenta")
	
	table.add_row("Normal Vector", f"{normal}")
	table.add_row("Reference Normal Vector", f"{ref_normal}")
	table.add_row("Signed Angle (degrees)", f"{signed_norm_angle_deg:.2f}")

	table.add_row("", "")
	
	table.add_row("Pitch Angle (degrees)", f"{pitch:.2f}")
	table.add_row("Reference Pitch Angle (degrees)", f"{ref_pitch:.2f}")
	table.add_row("Pitch Error (degrees)", f"{pitch_error_deg:.2f}")
	
	console.print(table)
