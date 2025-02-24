import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import src.algos.homography.hg_funcs as hg_funcs


plt.rcParams["font.serif"] = "cmr10"


def plot_complex_normal(
	blended_image, homography_matrix, normal, 
	cam_pts_3d, plane_coefficients, centroid, ref_normal, 
	signed_norm_angle_deg,
	root, seq_num, i, vis, save_vis,
	normal_candidates=None
):
	"""
	A complex plot with four components:
		1. Blended Image
		2. Calculated Homography Matrix
		3. 3D Plane with Normals (Predicted and Ground Truth)
		4. Calculated Angle between Normals (Predicted and Ground Truth)
		5. (Optional) Normal Candidates
	"""
	fig = plt.figure(figsize=(16, 8))
	gs = GridSpec(4, 2, figure=fig)

	# subplot 1 (upper left): Blended Image
	ax1 = fig.add_subplot(gs[0:3, 0])
	ax1.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
	ax1.set_title('Blended Image')
	ax1.axis('off')

	# subplot 2 (lower left): Homography Matrix
	ax2 = fig.add_subplot(gs[3, 0])
	homography_str = '\n'.join([' '.join([f"{item:8.4f}" for item in row]) for row in homography_matrix])
	ax2.axis('off')
	ax2.text(0.5, 0.5, homography_str, fontsize=12, ha='center', va='center', family='monospace')
	ax2.set_title('Homography Matrix')

	# subplot 3 (upper right): 3D Plane with Normals
	ax3 = fig.add_subplot(gs[0:3, 1], projection='3d')

	a, b, c, d = plane_coefficients
	# calculate plane grid
	x = np.linspace(np.min(cam_pts_3d[:, 0]), np.max(cam_pts_3d[:, 0]), 10)
	y = np.linspace(np.min(cam_pts_3d[:, 1]), np.max(cam_pts_3d[:, 1]), 10)
	X, Y = np.meshgrid(x, y)
	Z = (-a * X - b * Y - d) / c if c != 0 else np.zeros_like(X)
	z_min, z_max = np.min(cam_pts_3d[:, 2]), np.max(cam_pts_3d[:, 2])
	Z = np.clip(Z, z_min, z_max)  # clip Z values to the range of the 3D points

	# scale the arrow length relative to the bounding box
	scale_factor = 0.25
	max_range = np.ptp(cam_pts_3d, axis=0).max()  # max range in any axis
	arrow_length = max_range * scale_factor

	scaled_normal = normal * arrow_length
	scaled_ref_normal = ref_normal * arrow_length

	# plot the 3D points
	ax3.scatter(cam_pts_3d[:, 0], cam_pts_3d[:, 1], cam_pts_3d[:, 2],
				c='blue', marker='o', s=10, label="3D Points", alpha=0.5)

	# plot the plane
	ax3.plot_surface(X, Y, Z, color='orange', alpha=0.3)

	# plot the center of the plane
	ax3.scatter(centroid[0], centroid[1], centroid[2],
				color='red', s=50, label="Plane Centroid")

	# plot the calc normal vector
	ax3.quiver(
		centroid[0], centroid[1], centroid[2],
		scaled_normal[0], scaled_normal[1], scaled_normal[2],
		length=1.0, color='green', linewidth=2, label="Calculated Normal Vector"
	)

	# plot the ref. normal vector
	ax3.quiver(
		centroid[0], centroid[1], centroid[2],
		scaled_ref_normal[0], scaled_ref_normal[1], scaled_ref_normal[2],
		length=1.0, color='purple', linewidth=2, label="Reference Normal Vector"
	)

	# (debug) plot normal candidates if provided
	if normal_candidates:
		for idx, candidate in enumerate(normal_candidates):
			scaled_candidate = candidate * arrow_length
			ax3.quiver(
				centroid[0], centroid[1], centroid[2],
				scaled_candidate[0], scaled_candidate[1], scaled_candidate[2],
				length=1.0, color='cyan', linewidth=1, alpha=0.5, 
				label="Normal Candidate" if idx == 0 else ""
			)

	# Ensure equal aspect ratio
	limits = np.array([
		[np.min(cam_pts_3d[:, 0]), np.max(cam_pts_3d[:, 0])],
		[np.min(cam_pts_3d[:, 1]), np.max(cam_pts_3d[:, 1])],
		[np.min(cam_pts_3d[:, 2]), np.max(cam_pts_3d[:, 2])]
	])
	range_max = np.ptp(limits, axis=1).max()
	centers = np.mean(limits, axis=1)
	ax3.set_xlim(centers[0] - range_max / 2, centers[0] + range_max / 2)
	ax3.set_ylim(centers[1] - range_max / 2, centers[1] + range_max / 2)
	ax3.set_zlim(centers[2] - range_max / 2, centers[2] + range_max / 2)

	ax3.set_title("3D Plane with Normals (Predicted and Ground Truth)")
	ax3.set_xlabel("X")
	ax3.set_ylabel("Y")
	ax3.set_zlabel("Z")
	ax3.legend()
	ax3.view_init(elev=23, azim=29)

	# subplot 4 (lower right): angle between normals
	ax4 = fig.add_subplot(gs[3, 1])
	ax4.axis('off')

	ref_text = (f"Reference Normal: (x: {ref_normal[0]:.2f}, y: {ref_normal[1]:.2f}, z: {ref_normal[2]:.2f})")
	ax4.text(0.5, 0.7, ref_text, fontsize=10, ha='center', va='center', color='purple')
	calc_text = (f"Calculated Normal: (x: {normal[0]:.2f}, y: {normal[1]:.2f}, z: {normal[2]:.2f})")
	ax4.text(0.5, 0.5, calc_text, fontsize=10, ha='center', va='center', color='green')
	angle_text = f"Signed Angle between Normals: {signed_norm_angle_deg:.2f}°"
	ax4.text(0.5, 0.3, angle_text, fontsize=10, ha='center', va='center', color='black', weight='bold')
	if normal_candidates is not None:
		candidates_text = f"Number of Normal Candidates: {len(normal_candidates)}"
		ax4.text(0.5, 0.1, candidates_text, fontsize=10, ha='center', va='center', color='red', weight='bold')

	plt.tight_layout()

	if save_vis:
		output_path = os.path.join(root, seq_num, "complex_normal", f"complex_normal_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=300, format='jpg')
	if vis:
		plt.show()
	plt.close()


def plot_complex_pitch(
	blended_image, homography_matrix, pitch_vals,
	ref_pitch_vals, pitch_error,
	root, seq_num, i, vis, save_vis
):
	"""
	A complex plot with four components:
		1. Blended Image
		2. Calculated Homography Matrix
		3. Pitch Values (Predicted and Ground Truth)
		4. Calculated Difference Between Normals /deg/ (Predicted and Ground Truth)
	"""
	fig = plt.figure(figsize=(16, 8))
	gs = GridSpec(4, 2, figure=fig)

	# subplot 1 (upper left): Blended Image
	ax1 = fig.add_subplot(gs[0:3, 0])
	ax1.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
	ax1.set_title('Blended Image')
	ax1.axis('off')

	# subplot 2 (lower left): Homography Matrix
	ax2 = fig.add_subplot(gs[3, 0])
	homography_str = '\n'.join([' '.join([f"{item:8.4f}" for item in row]) for row in homography_matrix])
	ax2.axis('off')
	ax2.text(0.5, 0.5, homography_str, fontsize=12, ha='center', va='center', family='monospace')
	ax2.set_title('Homography Matrix')

	# subplot 3 (upper right): Pitch Values (Predicted and Ground Truth)
	ax3 = fig.add_subplot(gs[0:3, 1])

	# x-axis: frame numbers (1-based indexing)
	frames = range(1, len(pitch_vals) + 1)
	ax3.plot(frames, pitch_vals, label='Predicted Pitch', color='green', marker='o', linestyle='-')
	ax3.plot(frames, ref_pitch_vals, label='Ground Truth Pitch', color='purple', marker='x', linestyle='--')
	
	# highlight the current frame with a vertical line
	ax3.axvline(x=len(pitch_vals), color='green', linestyle=':', label='Current Frame')
	
	# Set plot titles and labels
	ax3.set_title('Pitch Values (Predicted and Ground Truth)')
	ax3.set_xlabel('Frame')
	ax3.set_ylabel('Pitch (degrees)')
	ax3.legend()
	
	# set consistent x and y limits for better video visualization
	ax3.set_xlim(1, max(80, len(pitch_vals) + 1))  # adjust 10 to a suitable number based on your data
	# determine y-axis limits based on current data with some margin
	all_pitches = pitch_vals + ref_pitch_vals
	if all_pitches:
		y_min = min(all_pitches) - 5  # margin of 5 degrees
		y_max = max(all_pitches) + 5
		ax3.set_ylim(y_min, y_max)
	
	ax3.grid(True)

	# subplot 4 (lower right): Calculated Difference Between Pitch Values /deg/
	ax4 = fig.add_subplot(gs[3, 1])
	ax4.axis('off')

	pitch_text = f"Pitch Error: {pitch_error:.2f}°"
	ax4.text(0.5, 0.3, pitch_text, fontsize=10, ha='center', va='center', color='black', weight='bold')

	plt.tight_layout()

	if save_vis:
		output_path = os.path.join(root, seq_num, "complex_pitch", f"complex_pitch_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=300, format='jpg')
	if vis:
		plt.show()
	plt.close()


def vis_matches(img1, img2, keypoints1, keypoints2, matches, roi_pts, root, seq_num, i, vis, save_vis):
	"""
	Visualize matches between two images.

	Parameters:
		img1, img2 (numpy.ndarray): Images to visualize matches between.
		keypoints1, keypoints2 (list of cv2.KeyPoint): Keypoints from both images.
		matches (list of cv2.DMatch): Matched keypoints.
		roi_pts (list of tuples): ROI points for drawing.
		output_path (str): Directory where the visualization will be saved.
		vis (bool): Whether to display the visualization.
		save_vis (bool): Whether to save the visualization to disk.
	"""
	roi_polygon = np.array(roi_pts, dtype=np.int32)
	img1_vis = img1.copy()
	img2_vis = img2.copy()
	cv2.polylines(img1_vis, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
	cv2.polylines(img2_vis, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
	
	img_with_matches = hg_funcs.visualize_matches(img1_vis, img2_vis, keypoints1, keypoints2, matches)

	if save_vis:
		output_path = os.path.join(root, seq_num, "matches", f"matches_img_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		cv2.imwrite(output_path, img_with_matches)
	if vis:
		hg_funcs.show_img(img_with_matches, title=f"Matches")


def vis_ransac(img1, img2, keypoints1, keypoints2, matches, mask, root, seq_num, i, vis, save_vis):
	"""
	Visualize RANSAC inliers and outliers.

	Parameters:
		img1, img2 (numpy.ndarray): Images to vis matches between.
		keypoints1, keypoints2 (list of cv2.KeyPoint): Keypoints from both images.
		matches (list of cv2.DMatch): Matched keypoints.
		mask (numpy.ndarray): Mask indicating inliers.
		output_path (str): If provided, save the visualization to this path.
		vis (bool): Whether to display the visualization.
		save_vis (bool): Whether to save the visualization to disk.
	"""
	img_ransac = hg_funcs.visualize_ransac_matches_combined(img1, img2, keypoints1, keypoints2, matches, mask)
	
	if save_vis:
		output_path = os.path.join(root, seq_num, "ransac", f"ransac_img_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		cv2.imwrite(output_path, img_ransac)
	if vis:
		hg_funcs.show_img(img_ransac, title="RANSAC Inliers and Outliers")


def vis_warp_if(
		img1, img2, homography_matrix, invert_direction,
		root, seq_num, i, vis, save_vis
):
	"""
	Warp src_img to the plane of dst_img using the homography matrix and visualize.

	Parameters:
		img1, img2 (numpy.ndarray): Images to warp between.
		homography_matrix (numpy.ndarray): Homography matrix.
		output_path (str): Directory where the warped image will be saved.
		vis (bool): Whether to display the warped image.
		save_vis (bool): Whether to save the warped image to disk.
	"""
	if not invert_direction:
		src_img, dst_img = img1, img2
	else:
		src_img, dst_img = img2, img1

	# warp the second image using the homography matrix
	img_warped = hg_funcs.warp_img(src_img, homography_matrix)

	if save_vis:
		output_path = os.path.join(root, seq_num, "warp_if", f"warp_if_img_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		cv2.imwrite(output_path, img_warped)
	if vis:
		hg_funcs.show_img(img_warped, title="Warped Image 2 to Image 1 Plane")
	
	return img_warped


def vis_surface_normal(img, normal, roi_pts, root, seq_num, i, vis, save_vis):
	"""
	Visualize the surface normal on the image by drawing an arrow.

	Parameters:
		img (numpy.ndarray): Original image to draw on.
		normal (numpy.ndarray): Surface normal vector (3,).
		roi_pts (list of tuples): ROI points for reference or origin.
		output_path (str): Directory where the visualization will be saved.
		vis (bool): Whether to display the visualization.
		save_vis (bool): Whether to save the visualization to disk.
	"""
	# Define the origin for the normal arrow (centroid of ROI)
	roi_polygon = np.array(roi_pts, dtype=np.int32)
	M = cv2.moments(roi_polygon)
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = roi_pts[0]  # Fallback to first point if moment is zero

	origin = (cX, cY)

	# Define the end point based on the normal vector
	# Project the 3D normal to 2D (assuming Z is forward, project to X and Y)
	normal_2d = normal[:2].flatten()  # Ensure it's a 1D array
	if np.linalg.norm(normal_2d) == 0:
		print("Normal vector has zero length in X and Y components; skipping visualization.")
		return
	normal_2d /= np.linalg.norm(normal_2d)  # Normalize
	scale = 100  # Length of the arrow in pixels

	# Ensure normal_2d components are scalars
	end_x = int(origin[0] + normal_2d[0] * scale)
	end_y = int(origin[1] + normal_2d[1] * scale)
	end_point = (end_x, end_y)

	# Draw the arrow
	img_with_normal = img.copy()
	cv2.arrowedLine(img_with_normal, origin, end_point, (255, 0, 0), 2, tipLength=0.3)

	# Optionally, add text to indicate normal vector
	cv2.putText(img_with_normal, 'Normal', end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
	cv2.polylines(img_with_normal, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

	if save_vis:
		output_path = os.path.join(root, seq_num, "normal", f"normal_img_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		cv2.imwrite(output_path, img_with_normal)
	if vis:
		hg_funcs.show_img(img_with_normal, title="Surface Normal")


def plot_pitch(pitch_values, root, seq_num, i, vis, save_vis):
	"""
	Plot pitch over time for each frame and save the plot.
	
	Parameters:
		pitch_values (list): List of pitch values for each frame.
		current_frame (int): Index of the current frame.
		output_dir (str): Directory to save the pitch plot images.
	"""
	fig, ax1 = plt.subplots(figsize=(4.8, 2.56), dpi=300)
	# TODO: fix plot limit + plot value problem
	ax1.set_ylim([-2, 2])
	ax1.set_xlabel('frame')
	ax1.set_ylabel('pitch (degree)')
	
	# Generate timestamps for frames
	timestamps = np.arange(len(pitch_values))
	
	# Plot vertical lines and scatter points to match the style
	ax1.plot((timestamps, timestamps), ([0] * len(timestamps), pitch_values), c='black')
	ax1.scatter(timestamps, pitch_values, s=10, c='red')
	
	# Save plot to output directory
	plt.tight_layout()
	
	if save_vis:
		output_path = os.path.join(root, seq_num, "pitch", f"pitch_plot_frame_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=300, format='jpg')
	plt.close(fig)


def vis_pitch(pitch, pitch_values, frame_idx, output_path, save_vis=False):
	"""
	Append the current pitch to the pitch values list and generate a pitch-over-time plot.

	Parameters:
		pitch (float): Current pitch angle.
		pitch_values (list): List storing all pitch angles so far.
		frame_idx (int): Current frame index.
		pitch_plot_dir (str): Directory to save the pitch-over-time plots.
		save_vis (bool): Whether to save the visualization to disk.
	"""
	# Append the current pitch to the list
	pitch_values.append(pitch)
	
	# Plot and save the pitch-over-time visualization
	plot_pitch(pitch_values, frame_idx, output_path, save_vis=save_vis)


def vis_blend(
		img1, img2, homography_matrix, keypoints1, keypoints2, matches, roi_pts,
		invert_direction, alpha,
		root, seq_num, i, vis, save_vis
):
	"""
	Visualize the alignment of img2 warped to img1 with keypoints overlaid.

	Parameters:
		img1 (numpy.ndarray): The first image (reference).
		img2 (numpy.ndarray): The second image to warp to img1's plane.
		homography_matrix (numpy.ndarray): The calculated homography matrix.
		keypoints1 (list of cv2.KeyPoint): Keypoints from img1.
		keypoints2 (list of cv2.KeyPoint): Keypoints from img2.
		matches (list of cv2.DMatch): Matched keypoints.
		alpha (float): Transparency factor for blending (0.0 to 1.0).
	"""
	alpha = 0.5 if alpha is None else alpha
	roi_polygon = np.array(roi_pts, dtype=np.int32)
	
	if not invert_direction:
		img_src, img_dst = img1, img2
	else:
		img_src, img_dst = img2, img1

	#  the ROI is defined on the source image before warping
	overlay = img_src.copy()
	cv2.polylines(overlay, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
	img_src = cv2.addWeighted(overlay, 0.7, img_src, 0.3, 0)

	img_warped = cv2.warpPerspective(img_src, homography_matrix, (img_dst.shape[1], img_dst.shape[0]))

	# Convert images to RGB if they are grayscale
	if len(img_dst.shape) == 2:
		img_dst = cv2.cvtColor(img_dst, cv2.COLOR_GRAY2RGB)
	if len(img_warped.shape) == 2:
		img_warped = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2RGB)

	# Blend images using alpha for transparency
	blended = cv2.addWeighted(img_dst, alpha, img_warped, 1 - alpha, 0)

	# Draw matched keypoints on the blended image
	for match in matches:
		pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
		pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype(int))
		color = (0, 255, 0)  # Green for keypoints
		cv2.circle(blended, pt1, 5, color, -1)  # Keypoint from img1
		cv2.circle(blended, pt2, 5, (0, 0, 255), -1)  # Keypoint from warped img2

		cv2.line(blended, pt1, pt2, (255, 128, 128), 1)

	if save_vis:
		output_path = os.path.join(root, seq_num, "blend", f"blend_img_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		cv2.imwrite(output_path, blended)
	if vis:
		hg_funcs.show_img(blended, title="Aligned Images with Keypoints Overlay")

	return blended


def vis_surface_normal_3d(img, normal, roi_pts, output_path, vis=False, save_vis=False):
	"""
	Visualize the surface normal and ROI points in 3D, aligned to OpenCV's right-handed Z-forward coordinate system.

	Parameters:
		img (numpy.ndarray): Image (not used here but part of the interface).
		normal (numpy.ndarray): Surface normal vector (3,).
		roi_pts (list of tuples): ROI points for reference or origin.
		output_path (str): Directory where the visualization will be saved.
		vis (bool): Whether to show the plot.
		save_vis (bool): Whether to save the visualization to disk.
	"""
	# Add a middle dimension (Y = 0) to make it 3D
	roi_pts_np = np.array(roi_pts)
	roi_pts_3d = np.column_stack(
		(roi_pts_np[:, 0], np.zeros(len(roi_pts_np)), roi_pts_np[:, 1]))
	
	# Apply the transform to swap axes
	tf = np.array([[1, 0, 0],    # New X = Old X
				   [0, 0, 1],    # New Y = Old Z
				   [0, 1, 0]])  # New Z = Old Y
	pts_tf = roi_pts_3d @ tf.T
	normal_tf = normal @ tf.T  # Transform the normal vector

	# Compute the centroid of the transformed ROI points
	roi_center = np.mean(pts_tf, axis=0)

	# Calculate the endpoint of the normal vector for visualization
	normal_end = roi_center + (normal_tf / np.linalg.norm(normal_tf)) * 50  # Scale for visibility

	# Set up plot
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	
	# Add labels to axes
	ax.set_xlabel('X-axis (+ right)')
	ax.set_ylabel('Z-axis (+ forward)')
	ax.set_zlabel('Y-axis (+ down)')  # y is positive downwards

	# Plot transformed ROI points
	pts_tf_poli = np.vstack((pts_tf, pts_tf[0]))  # Close the polygon
	ax.plot(pts_tf_poli[:, 0],  # X-coordinates
			pts_tf_poli[:, 1],  # Y-coordinates
			pts_tf_poli[:, 2],  # Z-coordinates
			'g-', label="ROI Polygon")

	# Plot the centroid
	ax.scatter(roi_center[0], roi_center[1], roi_center[2], color='b', s=50, label="ROI Centroid")

	# Plot the normal vector
	ax.quiver(
		roi_center[0], roi_center[1], roi_center[2],  # Origin
		normal_tf[0], normal_tf[1], normal_tf[2],     # Normal direction
		length=50, color='r', normalize=True, label="Surface Normal"
	)

	# Adjust the view (for better perspective)
	ax.view_init(elev=20, azim=-35, roll=0)

	# Set limits for better visualization (dynamic)
	all_points = np.vstack((pts_tf, roi_center, normal_end))
	ax.set_xlim([np.min(all_points[:, 0]) - 10, np.max(all_points[:, 0]) + 10])
	ax.set_ylim([np.min(all_points[:, 1]) - 10, np.max(all_points[:, 1]) + 10])
	ax.set_zlim([np.min(all_points[:, 2]) - 10, np.max(all_points[:, 2]) + 10])
	
	# Invert current Z-axis (to make Y positive downwards)
	ax.invert_zaxis()

	# Add legend
	ax.legend()

	# Save or display the plot
	if save_vis:
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		output_3d_path = output_path.replace(".png", "_3d.png")
		plt.savefig(output_3d_path, dpi=300, bbox_inches='tight', format='jpg')
		print(f"3D visualization saved to {output_3d_path}")

	if vis:
		plt.show()

	plt.close(fig)


def plot_hg_param_sensitivity(
	pitch_vals, 
	ref_pitch_vals, 
	hg_mats,
	root,
	seq_num,
	vis,
	save_vis
):
	"""
	Plot pitch values (calculated and ground truth) and homography matrix elements in a 3-column grid.

	Each column contains:
		- Top: Pitch plot with both calculated and ground truth values.
		- Middle: Three homography parameters.

	Parameters:
		pitch_vals (list of float): Calculated pitch values.
		ref_pitch_vals (list of float): Ground truth pitch values.
		hg_mats (list of np.ndarray): List of 3x3 homography matrices.
	"""
	if len(hg_mats) != len(pitch_vals):
		raise ValueError(
			"Length of homography_matrices must match length of pitch_values."
		)
	if len(ref_pitch_vals) != len(pitch_vals):
		raise ValueError(
			"Length of pitch_values_gt must match length of pitch_values."
		)
	
	# extract homography matrix elements (shape: (num_frames, 9))
	hg_elements = np.array([hg.flatten() for hg in hg_mats])
	
	# define subplot grid: 4 rows x 3 columns
	# row 0: pitch plots
	# rows 1-3: homography parameters
	fig = plt.figure(figsize=(18, 16), dpi=300)
	gs = GridSpec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3)
	
	# define groups for homography parameters
	hg_groups = {
		0: ['h1', 'h2', 'h3'],
		1: ['h4', 'h5', 'h6'],
		2: ['h7', 'h8', 'h9']
	}
	
	# define colors for homography parameters within each group
	# 3 distinct colors per group
	color_palette = plt.cm.tab10(np.linspace(0, 1, 3))
	
	frame_idxs = np.arange(len(pitch_vals))
	for col in range(3):
		# --- Pitch Subplot ---
		ax_pitch = fig.add_subplot(gs[0, col])
		ax_pitch.plot(frame_idxs, pitch_vals, label="Calculated Pitch",
				color="blue", linewidth=2, alpha=0.7
		)
		ax_pitch.plot(
			frame_idxs, ref_pitch_vals, label="Ground Truth Pitch", color="red",
			linestyle='--', linewidth=2, alpha=0.7
		)
		ax_pitch.set_ylabel("Pitch (degrees)")
		ax_pitch.set_title(f"Pitch Sensitivity - Column {col+1}")
		ax_pitch.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
		ax_pitch.legend(loc="upper right", fontsize='small')
		
		# --- Homography Parameter Subplots ---
		for row in range(1, 4):
			param_idx = col * 3 + (row - 1)
			param_name = hg_groups[col][row - 1]
			ax = fig.add_subplot(gs[row, col])
			ax.plot(
				frame_idxs, hg_elements[:, param_idx], 
				label=f"{param_name}", color=color_palette[row-1], 
				linewidth=1.5
			)
			ax.set_ylabel(param_name)
			ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
			
			# set x-labels only on the bottom row (hide for upper subplots)
			if row == 3:
				ax.set_xlabel("Frame")
			else:
				ax.set_xticklabels([])
	
	fig.suptitle("Pitch and Homography Matrix Sensitivity", fontsize=16, y=0.95)
	
	# prevent overlapping (leave space for the suptitle)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	
	if save_vis:
		output_path = os.path.join(root, seq_num, f"hg_param_sensitivity_pitch.jpg")
		plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
		print(f"HG sensitivity plot saved to {output_path}")
	if vis:
		plt.show()
	plt.close(fig)


def plot_complex_imu_pitch(
	blended_image, homography_matrix, pitch_vals, imu_vals, imu_ts,
	root, seq_name, i, vis, save_vis
):
	"""
	A complex plot with four components:
		1. Blended Image
		2. Calculated Homography Matrix
		3. Pitch Values (Predicted and Ground Truth)
		4. IMU values
	"""
	fig = plt.figure(figsize=(16, 8))
	gs = GridSpec(4, 2, figure=fig)

	# subplot 1 (upper left): Blended Image
	ax1 = fig.add_subplot(gs[0:3, 0])
	ax1.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
	ax1.set_title('Blended Image')
	ax1.axis('off')

	# subplot 2 (lower left): Homography Matrix
	ax2 = fig.add_subplot(gs[3, 0])
	homography_str = '\n'.join([' '.join([f"{item:8.4f}" for item in row]) for row in homography_matrix])
	ax2.axis('off')
	ax2.text(0.5, 0.5, homography_str, fontsize=12, ha='center', va='center', family='monospace')
	ax2.set_title('Homography Matrix')

	# subplot 3 (upper right): Pitch Values (Predicted and Ground Truth)
	ax3 = fig.add_subplot(gs[0:2, 1])

	# x-axis: frame numbers (1-based indexing)
	frames = range(1, len(pitch_vals) + 1)
	ax3.plot(frames, pitch_vals, label='Predicted Pitch', color='green', marker='o', linestyle='-')
	# ax3.plot(frames, ref_pitch_vals, label='Ground Truth Pitch', color='purple', marker='x', linestyle='--')
	
	# highlight the current frame with a vertical line
	ax3.axvline(x=len(pitch_vals), color='green', linestyle=':', label='Current Frame')
	
	# Set plot titles and labels
	ax3.set_title('Pitch Values (Predicted)') # and Ground Truth)')
	ax3.set_xlabel('Frame')
	ax3.set_ylabel('Pitch (degrees)')
	ax3.legend()
	
	# set consistent x and y limits for better video visualization
	ax3.set_xlim(1, max(80, len(pitch_vals) + 1))  # adjust 10 to a suitable number based on your data
	# determine y-axis limits based on current data with some margin
	all_pitches = pitch_vals  # + ref_pitch_vals
	if all_pitches:
		y_min = min(all_pitches) - 5  # margin of 5 degrees
		y_max = max(all_pitches) + 5
		ax3.set_ylim(y_min, y_max)
	
	ax3.grid(True)

	# subplot 4 (lower right): Calculated Difference Between Pitch Values /deg/
	ax4 = fig.add_subplot(gs[2:, 1])

	ax4.plot(imu_ts, imu_vals, label='IMU Values', color='blue', linestyle='-', linewidth=0.5)

	# Set plot titles and labels
	ax4.set_title('IMU Values (Predicted)') # and Ground Truth)')
	ax4.set_xlabel('Frame')
	ax4.set_ylabel('IMU Values (m/s^2)')
	ax4.legend()

	plt.tight_layout()

	if save_vis:
		output_path = os.path.join(root, seq_name, "complex_imu", f"complex_imu_{i}_{i+1}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=300, format='jpg')
	if vis:
		plt.show()
	plt.close()
