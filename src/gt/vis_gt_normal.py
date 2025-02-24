from typing import List, Tuple
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import cv2
import open3d as o3d


def visualize_point_clouds_with_open3d(raw_points, ego_points, frame_index=None, window_name="Point Cloud Visualization"):
    """
    Visualize raw and ego-frame LiDAR points using Open3D.

    :param raw_points: Nx3 numpy array of raw LiDAR points.
    :param ego_points: Nx3 numpy array of transformed ego-frame points.
    :param frame_index: Frame index (optional, for display in the window title).
    :param window_name: Window name for the Open3D visualization.
    """
    # Create Open3D Point Clouds
    raw_pc = o3d.geometry.PointCloud()
    ego_pc = o3d.geometry.PointCloud()

    # Assign points to the point clouds
    raw_pc.points = o3d.utility.Vector3dVector(raw_points)
    ego_pc.points = o3d.utility.Vector3dVector(ego_points)

    # Optionally, assign colors for differentiation
    raw_pc.paint_uniform_color([0, 0, 1])  # Blue for raw LiDAR points
    ego_pc.paint_uniform_color([1, 0, 0])  # Red for ego-frame points

    # Create coordinate frame
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0]
    )

    # Visualize the point clouds with axes
    full_window_name = f"{window_name} (Frame {frame_index})" if frame_index is not None else window_name
    o3d.visualization.draw_geometries(
        [raw_pc, ego_pc, axes], 
        window_name=full_window_name, 
        width=800, 
        height=600
    )


def visualize_roi(roi: List[Tuple[int, int]], size: Tuple[int, int] = (1280, 720)) -> None:
	"""
	Visualizes the ROI on a blank canvas of the given size.

	:param roi: List of (x, y) points defining the ROI.
	:param size: Tuple of (width, height) for the visualization canvas.
	"""
	# create blank image
	img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

	# draw ROI polygon
	polygon = np.array(roi, np.int32).reshape((-1, 1, 2))
	img = cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)

	# display the image
	plt.imshow(img)
	plt.axis("off")


def visualize_filtered_projection(img, proj_pts_2d, cam_pts_3d, root, seq_num, i, vis, save_vis, show_img=True):
	if show_img:
		plt.imshow(img)
	distances = np.sqrt(np.sum(np.square(cam_pts_3d), axis=-1))
	colors = cm.jet(distances / np.max(distances))
	plt.gca().scatter(proj_pts_2d[:, 0], proj_pts_2d[:, 1], color=colors, s=1)

	if save_vis:
		output_path = os.path.join(root, seq_num, "proj_filtered", f"proj_{i}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=300, format='jpg')
	if vis:
		plt.show()
	plt.close()


def plot_filtered_3d_points(points3d, root, seq_num, i, vis, save_vis):
	"""
	Plots the filtered 3D points in a simple 3D scatter plot with proper scaling.

	:param points3d: Nx3 array of 3D points.
	:param title: Title for the plot.
	"""
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')
	
	title=f"Filtered 3D Points for Image {i}"

	# Scatter 3D points
	ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c='blue', marker='o', s=2)

	# Set labels and title
	ax.set_title(title)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	# Adjust axes limits for proper scaling
	max_range = np.ptp(points3d, axis=0).max()  # Max range across any axis
	mid_x = (np.max(points3d[:, 0]) + np.min(points3d[:, 0])) * 0.5
	mid_y = (np.max(points3d[:, 1]) + np.min(points3d[:, 1])) * 0.5
	mid_z = (np.max(points3d[:, 2]) + np.min(points3d[:, 2])) * 0.5

	ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
	ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
	ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
 
	ax.view_init(elev=23, azim=29, roll=0)

	if save_vis:
		output_path = os.path.join(root, seq_num, "pc_filtered_3d", f"pc_filtered_{i}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)		
		plt.savefig(output_path, dpi=300, format='jpg')
	if vis:
		plt.show()
	plt.close()


def plot_plane_with_normal(points3d, plane_coefficients, normal, root, seq_num, i, vis, save_vis):
	"""
	Plots 3D points, the plane calculated by RANSAC, and the normal vector.

	:param points3d: Nx3 array of 3D points.
	:param plane_coefficients: Plane coefficients (a, b, c, d) from RANSAC.
	:param title: Title for the plot.
	"""
	a, b, c, d = plane_coefficients

	# Calculate plane grid
	x = np.linspace(np.min(points3d[:, 0]), np.max(points3d[:, 0]), 10)
	y = np.linspace(np.min(points3d[:, 1]), np.max(points3d[:, 1]), 10)
	X, Y = np.meshgrid(x, y)
	Z = (-a * X - b * Y - d) / c  # Plane equation: ax + by + cz + d = 0
 
	# dynamic limit for Z values
	z_min, z_max = np.min(points3d[:, 2]), np.max(points3d[:, 2])
	Z = np.clip(Z, z_min, z_max)  # Limit Z values to the range of the 3D points

	# Calculate center of the points
	center = np.mean(points3d, axis=0)

	# Scale the arrow length relative to the bounding box
	max_range = np.ptp(points3d, axis=0).max()  # Max range in any axis
	arrow_length = max_range * 0.15  # Adjust scale factor as needed
	scaled_normal = normal * arrow_length

	# Plot
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection='3d')

	title= f"3D Points with Plane and Normal {i}"

	# Scatter 3D points
	scatter = ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c='blue', marker='o', s=10, label="3D Points")

	# Plot the plane
	plane_surface = ax.plot_surface(X, Y, Z, color='orange', alpha=0.5)

	# Plot the center
	center_point = ax.scatter(center[0], center[1], center[2], color='red', s=50, label="Plane Center")

	# Plot the normal vector as an arrow
	normal_arrow = ax.quiver(
		center[0], center[1], center[2],  # Start point
		scaled_normal[0], scaled_normal[1], scaled_normal[2],  # Direction and length
		length=1.0, color='green', linewidth=2, label="Normal Vector"
	)

	# Add text for the normal vector components
	text_position = (
		center[0] + scaled_normal[0] * 1.1,  	   # x
		center[1] + scaled_normal[1] * 1.1,  	   # y
		center[2] + scaled_normal[2] * 1.1 + 0.1,  # z
	)
	ax.text(
		text_position[0], text_position[1], text_position[2],
		f"(x: {normal[0]:.2f}, y: {normal[1]:.2f}, z: {normal[2]:.2f})",
		color='black', fontsize=10, weight='bold'
	)

	# Scale plot uniformly
	max_range = np.ptp(points3d, axis=0).max()
	mid_x = (np.max(points3d[:, 0]) + np.min(points3d[:, 0])) * 0.5
	mid_y = (np.max(points3d[:, 1]) + np.min(points3d[:, 1])) * 0.5
	mid_z = (np.max(points3d[:, 2]) + np.min(points3d[:, 2])) * 0.5

	ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
	ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
	ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

	# Custom legend
	ax.legend([scatter, center_point, normal_arrow], ["3D Points", "Plane Center", "Normal Vector"])

	# Labels and title
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title(title)

	ax.view_init(elev=23, azim=29, roll=0)

	if save_vis:
		output_path = os.path.join(root, seq_num, "plane_with_normal", f"plane_normal_{i}.jpg")
		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		plt.savefig(output_path, dpi=300, format='jpg')
	if vis:
		plt.show()
	plt.close()



# combines the 2D projection and 3D plane with normal in a single figure

def plot_complex_visualization(
    img, proj_pts_2d, cam_pts_3d, plane_coefficients,
	normal, centroid, root, seq_num, i, vis, save_vis, show_img=True
):
    """
    Combines the 2D projection plot and the 3D plane with normal plot into a single figure with two subplots.

    :param img: Original image for 2D projection.
    :param proj_pts_2d: Projected 2D points.
    :param cam_pts_3d: Corresponding 3D points.
    :param plane_coefficients: Plane coefficients (a, b, c, d) for the plane plot.
    :param normal: Normal vector of the plane.
    :param root: Root path for saving the visualization.
    :param seq_num: Sequence number.
    :param i: Frame index.
    :param vis: Whether to display the visualization.
    :param save_vis: Whether to save the visualization to a file.
    :param show_img: Whether to show the image for the 2D projection plot.
    """
    # Create the figure and subplots
    fig = plt.figure(figsize=(16, 8))

    # Subplot 1: 2D Projection
    ax1 = fig.add_subplot(1, 2, 1)
    if show_img:
        ax1.imshow(img)
    distances = np.sqrt(np.sum(np.square(cam_pts_3d), axis=-1))
    colors = cm.jet(distances / np.max(distances))
    ax1.scatter(proj_pts_2d[:, 0], proj_pts_2d[:, 1], color=colors, s=1)
    ax1.set_title("2D Projection")
    ax1.axis("off")

    # Subplot 2: 3D Plane with Normal
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    a, b, c, d = plane_coefficients
    # Calculate plane grid
    x = np.linspace(np.min(cam_pts_3d[:, 0]), np.max(cam_pts_3d[:, 0]), 10)
    y = np.linspace(np.min(cam_pts_3d[:, 1]), np.max(cam_pts_3d[:, 1]), 10)
    X, Y = np.meshgrid(x, y)
    Z = (-a * X - b * Y - d) / c if c != 0 else np.zeros_like(X)
    z_min, z_max = np.min(cam_pts_3d[:, 2]), np.max(cam_pts_3d[:, 2])
    Z = np.clip(Z, z_min, z_max)  # Clip Z values to the range of the 3D points

    # Scale the arrow length relative to the bounding box
    max_range = np.ptp(cam_pts_3d, axis=0).max()  # Max range in any axis
    arrow_length = max_range * 0.15  # Adjust scale factor as needed
    scaled_normal = normal * arrow_length

    # Plot the 3D points
    ax2.scatter(cam_pts_3d[:, 0], cam_pts_3d[:, 1], cam_pts_3d[:, 2], c='blue', marker='o', s=10, label="3D Points")

    # Plot the plane
    ax2.plot_surface(X, Y, Z, color='orange', alpha=0.5)

    # Plot the center of the plane
    ax2.scatter(centroid[0], centroid[1], centroid[2], color='red', s=50, label="Plane Center")

    # Plot the normal vector
    ax2.quiver(
        centroid[0], centroid[1], centroid[2],
        scaled_normal[0], scaled_normal[1], scaled_normal[2],
        length=1.0, color='green', linewidth=2, label="Normal Vector"
    )

    # Add text for the normal vector
    text_position = (
        centroid[0] + scaled_normal[0] * 1.1,
        centroid[1] + scaled_normal[1] * 1.1,
        centroid[2] + scaled_normal[2] * 1.1 + 0.1,
    )
    ax2.text(
        text_position[0], text_position[1], text_position[2],
        f"(x: {normal[0]:.2f}, y: {normal[1]:.2f}, z: {normal[2]:.2f})",
        color='black', fontsize=10, weight='bold'
    )

    # Ensure equal aspect ratio
    limits = np.array([
        [np.min(cam_pts_3d[:, 0]), np.max(cam_pts_3d[:, 0])],
        [np.min(cam_pts_3d[:, 1]), np.max(cam_pts_3d[:, 1])],
        [np.min(cam_pts_3d[:, 2]), np.max(cam_pts_3d[:, 2])]
    ])
    range_max = np.ptp(limits, axis=1).max()
    centers = np.mean(limits, axis=1)
    ax2.set_xlim(centers[0] - range_max / 2, centers[0] + range_max / 2)
    ax2.set_ylim(centers[1] - range_max / 2, centers[1] + range_max / 2)
    ax2.set_zlim(centers[2] - range_max / 2, centers[2] + range_max / 2)

    # Set up the 3D plot
    ax2.set_title("3D Plane with Normal")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()
    ax2.view_init(elev=23, azim=29)

    # Save or show the combined plot
    if save_vis:
        output_path = os.path.join(root, seq_num, "complex", f"complex_{i}.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, format='jpg')
    if vis:
        plt.show()
    plt.close()
