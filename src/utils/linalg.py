import numpy as np


def align_normal_ref_vec(normal_calc, normal_ref):
	"""
	Aligns the calculated normal to point towards the reference normal.

	:param normal_calc: Calculated normal vector (numpy array).
	:param normal_ref: Ground truth normal vector (numpy array).
	:return: Aligned Calculated normal vector (numpy array).
	"""
	if np.dot(normal_calc, normal_ref) < 0:
		normal_calc = -normal_calc
	return normal_calc


def align_normal_ref_pt(normal_calc, ref_pt=None, plane_ctr=None):
	"""
	Aligns the calculated normal vector to point towards a reference direction.
	
	:param normal_calc: Calculated normal vector (numpy array).
	:param ref_pt: A reference point for alignment (e.g., camera position).
	:param plane_ctr: Center of the plane (used for consistency).
	:return: Aligned Calculated normal vector (numpy array).

	Example:
	--------
	>>> normal_calc = np.array([0, 0, -1])
	>>> ref_pt = np.array([0, 0, 1])    # camera position in right-handed +y-forward frame
	>>> plane_ctr = np.array([0, 0, 0])  # center of the plane in the same frame
	>>> align_normal_ref_pt(normal_calc, ref_pt)
	res: array([ 0.,  0.,  1.])
	"""
	if ref_pt is None:
		ref_pt = np.array([0, 0, 0])
	if plane_ctr is None:
		plane_ctr = np.array([0, 0, 0])

	# create a vector from the plane center to the reference point
	normal_ref = ref_pt - plane_ctr

	# if the dot product is negative, flip the normal
	if np.dot(normal_calc, normal_ref) < 0:
		normal_calc = -normal_calc
	return normal_calc


def rotate_vector(vec, R):
	"""Rotate a vector using a rotation matrix R, then normalize it."""
	vec_tf = R @ vec
	vec_tf /= np.linalg.norm(vec_tf)
	return vec_tf
