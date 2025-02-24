import os

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['ieee', 'grid'])

# 0.35 scale factor for IEEE double column (multiply parameters by 0.35) -> (3.5, 2.625)
plt.rcParams.update({
	# Figure resolution
	"figure.dpi": 1200,
	"savefig.dpi": 1200,

	# Scaled font sizes for IEEE small figures
	"font.family": "Arial",
	"font.size": 8.4,
	"axes.labelsize": 10.5,
	"axes.titlesize": 10.5,
	"xtick.labelsize": 8.82,
	"ytick.labelsize": 8.82,
	"legend.fontsize": 8.82,

	# Grid settings remain the same
	"grid.linewidth": 0.28,  
	"grid.linestyle": "--",  
	"grid.alpha": 0.245,  

	# Scaled lines and markers
	"lines.linewidth": 1.0,  # 1.4,
	"lines.markersize": 2.8,  
	"lines.markeredgewidth": 0.7,
})

# Update rcParams for 1200 dpi, scaling the text and line sizes by 0.5 (Springer LNCS)
plt.rcParams.update({
    # Figure resolution
    "figure.dpi": 1200,
    "savefig.dpi": 1200,

    # Scaled font sizes (multiplied by 0.5 compared to original IEEE settings)
    "font.family": "Arial",
    "font.size": 5.0,           # was 8.4
    "axes.labelsize": 6.05,       # was 10.5
    "axes.titlesize": 6.05,       # was 10.5
    "xtick.labelsize": 5.21,      # was 8.82
    "ytick.labelsize": 5.21,      # was 8.82
    "legend.fontsize": 5.0,      # was 8.82

    # Grid settings (you may adjust these if necessary)
    "grid.linewidth": 0.28,
    "grid.linestyle": "--",
    "grid.alpha": 0.245,

    # Scaled lines and markers (multiplied by 0.5)
    "lines.linewidth": 0.6,     # was 1.0
    "lines.markersize": 1.4,      # was 2.8
    "lines.markeredgewidth": 0.35,  # was 0.7

    # Scale down the axes frame (spines) width and tick widths similarly
    "axes.linewidth": 0.5,      # Adjusted from the default value
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
})


def plot_pitch(
	pitch_vals_gt,
	pitch_vals_exp,
	pitch_vals_ref,
	root,       # root directory used for saving the visualization
	seq_num,    # sequence number (used in the output path)
	vis=True,   # if True, show the plot interactively
	save_vis=True  # if True, save the plot as an image
):
	"""
	Plot a pitch vs. frame number plot for three pitch lists.
	
	Parameters:
		pitch_vals_gt (list of float): Pitch values from method 1.
		pitch_vals_exp (list of float): Pitch values from method 2.
		pitch_vals_ref (list of float): Pitch values from method 3.
		root (str): Root directory to save the output image.
		seq_num (str): Sequence identifier.
		i (int): Current frame index (used for naming the output file).
		vis (bool): Whether to show the plot.
		save_vis (bool): Whether to save the plot image.
	"""
	# create a new figure and axis
	# fig, ax = plt.subplots(figsize=(3.5, 2.625))
	fig, ax = plt.subplots(figsize=(3.27, 1.35))
	
	# x-axis: frame numbers (starting at 1)
	frames = range(1, len(pitch_vals_gt) + 1)
	
	# plot the three pitch lists with different colors and markers
	ax.plot(frames, pitch_vals_exp, label='Ours')  # color='green' linestyle='-'
	ax.plot(frames, pitch_vals_ref, label='SOTA Method')  # color='orange' linestyle='-.'
	ax.plot(frames, pitch_vals_gt, label='Ground Truth', alpha=0.7)  # color='purple' marker='x' linestyle='-' alpha=0.7
	
	# # highlight the current frame with a vertical line
	# ax.axvline(x=len(pitch_vals_gt), color='grey', linestyle=':', label='Current Frame')
	
	# set titles and labels
	# ax.set_title('Pitch Values Comparison')
	ax.set_xlabel('Frame')
	ax.set_ylabel('Pitch (degrees)')
	ax.legend()
	ax.grid(True)
	
	# optionally adjust x and y limits
	ax.set_xlim(1, max(80, len(pitch_vals_gt) + 1))
	all_pitches = pitch_vals_gt + pitch_vals_exp + pitch_vals_ref
	if all_pitches:
		y_margin = 5
		ax.set_ylim(min(all_pitches) - y_margin, max(all_pitches) + y_margin)
	
	
	for line in ax.get_lines():
		line.set_alpha(0.7)
	ax.get_lines()[-1].set_alpha(1.0)  # ground truth line (put it last)
 
	plt.tight_layout(pad=0.2)
	

	if save_vis:
		output_path = os.path.join(root, seq_num, f"pitch_plot.pdf")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=600, format='pdf', bbox_inches='tight')
	if vis:
		plt.show()
	plt.close()


def plot_complex_normal(
		cam_pts_3d, plane_coefficients, centroid,
		calculated_normal, gt_normal, ref_normal,
		root, seq_num, i, vis, save_vis,
		normal_candidates=None
):
	"""
	A complex plot with two components:
	  1. Top: 3D Plane with Normals (Three Normals: Calculated, Ground Truth, and SOTA)
	  2. Bottom: A text panel displaying vector values and the angles between normals.
	
	Parameters:
		cam_pts_3d (np.ndarray): 3D points of the plane.
		plane_coefficients (tuple/list): Plane parameters (a, b, c, d) from ax+by+cz+d=0.
		centroid (np.ndarray): 3D centroid of the plane.
		calculated_normal (np.ndarray): Our (predicted) normal vector.
		gt_normal (np.ndarray): Ground truth normal vector.
		ref_normal (np.ndarray): SOTA (reference) normal vector.
		root (str): Root directory for saving the visualization.
		seq_num (str): Sequence identifier.
		i (int): Current index used in the output filename.
		vis (bool): If True, show the plot.
		save_vis (bool): If True, save the plot to disk.
		normal_candidates (list, optional): List of candidate normals to plot.
	"""
	# Create a figure with 2 rows: top for 3D plot, bottom for text
	fig = plt.figure(figsize=(14, 10))
	gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1])
	
	#################################
	# Subplot 1 (Top): 3D Plane Plot#
	#################################
	ax1 = fig.add_subplot(gs[0], projection='3d')
	
	# Unpack plane coefficients (ax+by+cz+d=0)
	a, b, c, d = plane_coefficients
	# Create a grid for the plane
	x = np.linspace(np.min(cam_pts_3d[:, 0]), np.max(cam_pts_3d[:, 0]), 10)
	y = np.linspace(np.min(cam_pts_3d[:, 1]), np.max(cam_pts_3d[:, 1]), 10)
	X, Y = np.meshgrid(x, y)
	# Avoid division by zero if c == 0:
	Z = (-a * X - b * Y - d) / c if c != 0 else np.zeros_like(X)
	# Clip Z values to the range of the 3D points
	z_min, z_max = np.min(cam_pts_3d[:, 2]), np.max(cam_pts_3d[:, 2])
	Z = np.clip(Z, z_min, z_max)
	
	# Determine arrow scaling
	scale_factor = 0.25
	max_range = np.ptp(cam_pts_3d, axis=0).max()
	arrow_length = max_range * scale_factor
	
	# Scale normals
	scaled_calc = calculated_normal * arrow_length
	scaled_gt = gt_normal * arrow_length
	scaled_ref = ref_normal * arrow_length
	
	# Plot 3D points and plane surface
	ax1.scatter(cam_pts_3d[:, 0], cam_pts_3d[:, 1], cam_pts_3d[:, 2],
				c='blue', marker='o', s=10, label="3D Points", alpha=0.5)
	ax1.plot_surface(X, Y, Z, color='orange', alpha=0.3)
	ax1.scatter(centroid[0], centroid[1], centroid[2],
				color='red', s=50, label="Plane Centroid")
	
	# Plot the three normals with quiver
	ax1.quiver(
		centroid[0], centroid[1], centroid[2],
		scaled_calc[0], scaled_calc[1], scaled_calc[2],
		color='green', linewidth=2, label="Our Normal"
	)
	ax1.quiver(
		centroid[0], centroid[1], centroid[2],
		scaled_gt[0], scaled_gt[1], scaled_gt[2],
		color='purple', linewidth=2, label="GT Normal", alpha=0.7
	)
	ax1.quiver(
		centroid[0], centroid[1], centroid[2],
		scaled_ref[0], scaled_ref[1], scaled_ref[2],
		color='orange', linewidth=2, label="SOTA Normal", alpha=0.7
	)
	
	# Optionally plot normal candidates if provided
	if normal_candidates:
		for idx, candidate in enumerate(normal_candidates):
			scaled_candidate = candidate * arrow_length
			ax1.quiver(
				centroid[0], centroid[1], centroid[2],
				scaled_candidate[0], scaled_candidate[1], scaled_candidate[2],
				color='cyan', linewidth=1, alpha=0.5,
				label="Normal Candidate" if idx == 0 else ""
			)
	
	# Ensure equal aspect ratio for the 3D plot
	limits = np.array([
		[np.min(cam_pts_3d[:, 0]), np.max(cam_pts_3d[:, 0])],
		[np.min(cam_pts_3d[:, 1]), np.max(cam_pts_3d[:, 1])],
		[np.min(cam_pts_3d[:, 2]), np.max(cam_pts_3d[:, 2])]
	])
	range_max = np.ptp(limits, axis=1).max()
	centers = np.mean(limits, axis=1)
	ax1.set_xlim(centers[0] - range_max / 2, centers[0] + range_max / 2)
	ax1.set_ylim(centers[1] - range_max / 2, centers[1] + range_max / 2)
	ax1.set_zlim(centers[2] - range_max / 2, centers[2] + range_max / 2)
	
	ax1.set_title("3D Plane with Normals")
	ax1.set_xlabel("X")
	ax1.set_ylabel("Y")
	ax1.set_zlabel("Z")
	ax1.legend()
	ax1.view_init(elev=23, azim=29)
	
	##############################################
	# Subplot 2 (Bottom): Text Panel (Vector Data)#
	##############################################
	ax2 = fig.add_subplot(gs[1])
	ax2.axis('off')
	
	# Function to compute the angle (in degrees) between two vectors.
	def calc_angle(v1, v2):
		dot = np.dot(v1, v2)
		norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
		cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
		return np.degrees(np.arccos(cos_angle))
	
	# Compute angles between the normals
	angle_calc = calc_angle(calculated_normal, gt_normal)
	angle_calc_ref = calc_angle(ref_normal, gt_normal)
	
	# Build text strings for each vector and the calculated angles.
	calc_text = (f"Our Normal:\n"
				 f"  x: {calculated_normal[0]:.2f}"
				 f"  y: {calculated_normal[1]:.2f}"
				 f"  z: {calculated_normal[2]:.2f}")
	
	gt_text = (f"GT Normal:\n"
			   f"  x: {gt_normal[0]:.2f}"
			   f"  y: {gt_normal[1]:.2f}"
			   f"  z: {gt_normal[2]:.2f}")
	
	ref_text = (f"SOTA Normal:\n"
				f"  x: {ref_normal[0]:.2f}"
				f"  y: {ref_normal[1]:.2f}"
				f"  z: {ref_normal[2]:.2f}")
	
	angles_calc_text = (f"Angle (Our vs GT): {angle_calc:.2f}°")
	angles_ref_text = (f"Angle (SOTA vs GT): {angle_calc_ref:.2f}°")
	
	# Place the text with colors as in the original.
	ax2.text(0.05, 0.85, calc_text, fontsize=12, ha='left', va='center', color='green')
	ax2.text(0.05, 0.55, gt_text, fontsize=12, ha='left', va='center', color='purple')
	ax2.text(0.05, 0.25, ref_text, fontsize=12, ha='left', va='center', color='orange')
	ax2.text(0.3, 0.85, angles_calc_text, fontsize=12, ha='left', va='center', color='green', weight='bold')
	ax2.text(0.3, 0.25, angles_ref_text, fontsize=12, ha='left', va='center', color='orange', weight='bold')
	# ax2.text(0.55, 0.55, angles_text, fontsize=12, ha='left', va='center', color='black', weight='bold')
	

	plt.tight_layout()
	
	if save_vis:
		output_path = os.path.join(root, seq_num, "complex_normal", f"complex_normal_{i}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=300, format='jpg')
	if vis:
		plt.show()
	plt.close()
