import warnings

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_homography(hg_mat, intrinsics):
	"""
	Normalize a homography matrix by:
	1. Removing the effect of the camera intrinsic matrix.
	2. Scaling the homography so that H[2, 2] = 1.

	Parameters:
		homography (numpy.ndarray): 3x3 homography matrix.
		intrinsics (numpy.ndarray): 3x3 camera intrinsic matrix.

	Returns:
		numpy.ndarray: Fully normalized homography matrix.
	"""
	# normalize by camera intrinsic matrix
	hg_mat_intrinsics_norm = np.linalg.inv(intrinsics) @ hg_mat @ intrinsics
	# scale the homography so that H[2, 2] = 1
	hg_mat_final_norm = hg_mat_intrinsics_norm / hg_mat_intrinsics_norm[2, 2]
	return hg_mat_final_norm


def decompose_homography(hg_mat, intrinsics, last_n_norms, handle_outliers=False, debug=False):
	"""
	Decomposes a homography matrix into rotation, translation, and normal vectors.
	Also selects the correct solution based on the chosen criteria.
	"""
	hg_mat_norm = normalize_homography(hg_mat, intrinsics)
	retval, rotations, translations, normals = cv2.decomposeHomographyMat(hg_mat_norm, np.eye(3))
	
	sols = []
	correct_sol = None
	ref_dev_vec = np.array([0, -1, 0])
	
	for i in range(retval):
		rot = rotations[i]
		t = translations[i].flatten()
		n = normals[i].flatten()        

		sols.append({
			"rotation": rot,
			"translation": t,
			"normal": n
		})

		if debug:
			print(f"Translation: {t}")
			print(f"Normal: {n}")

	correct_sol = select_correct_sol(sols, last_n_norms, handle_outliers, ref_dev_vec)
	
	if correct_sol is None:
		warnings.warn("No valid decomposition solution found. Choosing fallback solution.", UserWarning)
		correct_sol = select_fallback_sol(sols, ref_dev_vec)
		if correct_sol is None:
			warnings.warn("No valid decomposition solution found. Choosing first solution.", UserWarning)
			correct_sol = sols[0]
	if debug:
		print(f"Selected solution: {correct_sol}")
	return correct_sol, sols


def select_correct_sol(
	sols,
	last_n_norms,
	handle_outliers,
	ref_dev_vec=None,
	max_dev_deg=30, # maximum deviation from 0°
	x_max=0.1,
	y_max_neg=-0.9,
	z_max_pos=0.3,  # arcsin(0.3) ≈ 17.46°
	z_max_neg=-0.3, # arcsin(-0.3) ≈ -17.46°
):
	"""
 	Selects the correct decomposition solution based on the chosen criteria.
  	"""
	max_dev_rad = np.deg2rad(max_dev_deg)

	valid_sols = []
	for sol in sols:
		t = sol["translation"]
		n = sol["normal"]
		
		if t[2] <= 0:
			continue
		if abs(n[0]) >= x_max:
			continue
		if n[1] >= y_max_neg:
			continue
		
		# exclude normals outside slight incline/decline range
		if not (z_max_neg <= n[2] <= z_max_pos):
			continue  
		
		if ref_dev_vec is not None:
			# angle between the normal and ref. deviation vector
			cos_angle = np.dot(
				(n / np.linalg.norm(n)),
				(ref_dev_vec / np.linalg.norm(ref_dev_vec))
			)
			angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
			dev = abs(angle)  # deviation from 0°
			if dev > max_dev_rad:
				continue
		
		valid_sols.append(sol)
	
	best_sol = filter_sols(valid_sols, last_n_norms, handle_outliers)
	return best_sol


def filter_sols(
	valid_sols,
	last_n_norms,
	handle_outliers,
	outlier_thresh_deg=30,
):
	"""
	Filter the valid solutions based on the last n normals. 
	"""
	if not last_n_norms["normals"]:  # no history available
		if not valid_sols:
			return None
		else:
			return valid_sols[0]  # TODO: add some condition here

	norms = np.array(last_n_norms["normals"])
	rots = np.array(last_n_norms["rotations"])
	trls = np.array(last_n_norms["translations"])
 
	norms /= np.linalg.norm(norms, axis=1, keepdims=True)
	norm_avg = np.mean(norms, axis=0)
	norm_avg /= np.linalg.norm(norm_avg)

	# angle between norm_last and norm_avg
	norm_last = norms[-1]
	cos_angle = np.dot(norm_last, norm_avg)
	angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
	deviation_deg = np.rad2deg(angle)
 
	if handle_outliers:
		if deviation_deg > outlier_thresh_deg:
			is_outlier = True
			ref_norm = norm_avg
			rot_mat = calc_vec_rot_mat(ref_norm, [0, -1, 0])
		else:
			is_outlier = False
			ref_norm = norm_last

		if is_outlier and not valid_sols:
			ref_norm = norm_avg
			sub_sol = {
				"normal": norm_avg,
				"rotation": rot_mat,
				"translation": trls[-1],
			}
			return sub_sol
		elif not is_outlier and not valid_sols:
			sub_sol = {
				"normal": norm_last,
				"rotation": rots[-1],
				"translation": trls[-1],
			}
			return sub_sol
	else:
		# when outlier handling is disabled, always use the last normal
		ref_norm = norm_last
		is_outlier = False  # not used in this case
  
	if not valid_sols:
		if is_outlier:
			ref_norm = norm_avg
			sub_sol = {
				"normal": norm_avg,
				"rotation": rot_mat,
				"translation": trls[-1],
			}
		else:
			sub_sol = {
				"normal": norm_last,
				"rotation": rots[-1],
				"translation": trls[-1],
			}
		return sub_sol

	# select the solution whose normal is closest to ref_norm
	min_angle_diff = float('inf')
	best_sol = None
	for sol in valid_sols:
		sol_norm = sol["normal"] / np.linalg.norm(sol["normal"])
		cos_angle = np.dot(sol_norm, ref_norm)
		angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
		if angle < min_angle_diff:
			min_angle_diff = angle
			best_sol = sol
	return best_sol


def select_fallback_sol(
	sols,
	ref_dev_vec,
	x_max=0.1,
	y_max_neg=-0.9,
	max_dev_deg=30
 ):
	"""
	Implements the fallback criteria to select a solution when primary criteria fail.
	"""
	max_dev_rad = np.deg2rad(max_dev_deg)

	for sol in sols:
		t = sol["translation"]
		n = sol["normal"]
		
		if t[2] <= 0:
			continue
		if abs(n[0]) >= x_max:
			continue
		if n[1] >= y_max_neg:
			continue

		if ref_dev_vec is not None:
			cos_angle = np.dot(
				(n / np.linalg.norm(n)),
				(ref_dev_vec / np.linalg.norm(ref_dev_vec))
			)
			angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
			dev = abs(angle)  # deviation from 0°
			if dev > max_dev_rad:
				continue
		return sol
	return None


def tf_rot_mat_to_euler(rot_mat, tf_order='zyx', deg=True, debug=False):
	"""
	Convert a rotation matrix to Euler angles in the specified order.
	When debug is True, prints the roll, pitch, and yaw angles in the 
	correct unpack order.
	Default:
		The rotation matrix is assumed to be in the camera frame 
		(extrinsic, right-handed, Z-forward, X-right, Y-down).
	"""
	r = R.from_matrix(rot_mat)
	angles = r.as_euler(tf_order, degrees=deg)
	if debug:
		axis_labels = {'x': 'Roll', 'y': 'Pitch', 'z': 'Yaw'}
		labels = [axis_labels[axis] for axis in tf_order]
		print(
			f"Euler angles ({tf_order}):   "
			f"{labels[0]}={angles[0]:.2f}, "
			f"{labels[1]}={angles[1]:.2f}, "
			f"{labels[2]}={angles[2]:.2f}  "
		)
	return tuple(angles)


def normalize_rot_mat(rot_mat):
	U, S, Vt = np.linalg.svd(rot_mat)
	return np.dot(U, Vt)


def is_rot_mat(rot_mat, debug=True):
	should_be_identity = np.dot(rot_mat.T, rot_mat)
	I = np.identity(3, dtype=rot_mat.dtype)
	is_valid_rot_mat = (
		np.allclose(should_be_identity, I) and
		np.isclose(np.linalg.det(rot_mat), 1)
	)
	if debug:
		if not is_valid_rot_mat:
			print("[WARN] The matrix is not a valid rotation matrix.")
			print("Determinant:", np.linalg.det(rot_mat))
			print("Matrix product:", should_be_identity)
		else:
			print("[INFO] The matrix is a valid rotation matrix.")
	return is_valid_rot_mat


def calc_vec_rot_mat(normal, reference=[0, -1, 0]):
	"""
	Computes the rotation matrix for a given normal vector 
 	and reference direction.
	"""
	normal = np.array(normal)
	reference = np.array(reference)
	
	normal = normal / np.linalg.norm(normal)
	reference = reference / np.linalg.norm(reference)
	
	# compute rotation axis and angle
	axis = np.cross(normal, reference)
	angle = np.arccos(np.clip(np.dot(normal, reference), -1.0, 1.0))
	
	if np.linalg.norm(axis) < 1e-6:  # if the vectors are aligned
		return np.eye(3)
	
	axis /= np.linalg.norm(axis)  # normalize rotation axis
	
 	# compute the skew-symmetric matrix for the axis
	K = np.array([
		[       0, -axis[2],  axis[1]],
		[ axis[2],        0, -axis[0]],
		[-axis[1],  axis[0],        0]
	])
	# rotation matrix using Rodrigues' formula
	R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
	return R


if __name__ == '__main__':

	debug = False

	rotation_matrix = np.array(
		[[0.36, -0.8, 0.48],
		 [0.8,   0.6,    0],
		 [-0.48, 0.36, 0.8]]
	)
	tf_order = 'zyx'
	deg = True

	# Validate and normalize
	if not is_rot_mat(rotation_matrix, debug=debug):
		rotation_matrix = normalize_rot_mat(rotation_matrix)
		is_rot_mat(rotation_matrix, debug=debug)

	# SciPy Euler angle calculation
	yaw, pitch, roll = tf_rot_mat_to_euler(rotation_matrix, tf_order=tf_order, deg=deg, debug=True)
	print(yaw, pitch, roll)