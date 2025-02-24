import numpy as np
from scipy.stats import pearsonr
from fastdtw import fastdtw


# normal vector error
def calc_norm_vec_error(normal, normal_gt):
    """
    Computes the absolute normal vector error in degrees between a single estimated 
    and ground truth normal vector.

    This follows the metric used in the paper:
    
    E_normal = degrees(arccos(N_est . N_gt))

    :param normal: Estimated normal vector (3D numpy array).
    :param normal_gt: Ground truth normal vector (3D numpy array).
    :return: Normal vector error in degrees.
    """
    normal = normal / np.linalg.norm(normal)
    normal_gt = normal_gt / np.linalg.norm(normal_gt)

    # compute dot product and clamp
    dot_product = np.dot(normal, normal_gt)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # compute angular error
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# signed normal vector error
def calc_signed_norm_vec_error(normal, normal_gt, normal_ref=[0, 0, 1]):
    """
    Computes the signed angular error in degrees between two normal vectors 
    using a reference normal to determine directionality.

    The angular error is computed as:
    
        E_signed = sign((N_est x N_gt) ⋅ N_ref) * degrees(arccos(N_est ⋅ N_gt))

    where:
    - N_est is the estimated normal vector (or first normal vector).
    - N_gt is the ground truth normal vector (or second normal vector).
    - N_ref is the reference normal vector, which defines the reference plane for sign determination.
    - 'x' denotes the cross product.
    - '⋅' denotes the dot product.
    - The sign of the error is determined by the dot product of the cross product 
      (N_est x N_gt) with the reference normal N_ref.

    :param normal1: First normal vector (numpy array).
    :param normal2: Second normal vector (numpy array).
    :param reference_normal: Normal vector defining the reference plane. Defaults to Z-axis ([0, 0, 1]).
    :return: Signed angular error in degrees.
    """
    normal = normal / np.linalg.norm(normal)
    normal_gt = normal_gt / np.linalg.norm(normal_gt)
    normal_ref = np.array(normal_ref) / np.linalg.norm(np.array(normal_ref))

    # compute dot product and clamp
    dot_product = np.dot(normal, normal_gt)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # compute angular error
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    # determine the directionality of the error
    cross_product = np.cross(normal, normal_gt)
    sign = np.sign(np.dot(cross_product, normal_ref))

    # handle cases where the cross product is orthogonal to the reference normal
    if sign == 0 and (np.isclose(angle_deg, 90.0) or np.isclose(angle_deg, 180.0)):
        signed_angle_deg = angle_deg
    else:
        signed_angle_deg = angle_deg * sign

    return signed_angle_deg


# pitch error
def calc_pitch_error(pitch, ref_pitch, signed=False):
    """
    Calculate the pitch error between two pitch angles.

    :param pitch: Predicted pitch angle in degrees.
    :param ref_pitch: Reference pitch angle in degrees.
    :return: Absolute pitch error in degrees.
    """
    pitch_error = pitch - ref_pitch
    if signed:
        return pitch_error
    return abs(pitch_error)


def calc_correlation(ref_pitch, homography_params):
	"""
	Calculate the correlation between the reference pitch and each homography parameter.

    :param ref_pitch (numpy.ndarray): Array of reference pitch values.
    :param homography_params (numpy.ndarray): 2D array of homography parameters. 
                                        Shape: (num_frames, 9).
    :return correlations (list): List of correlation coefficients for each parameter.
	"""
	correlations = []
	for i in range(homography_params.shape[1]):
		param = homography_params[:, i]
		corr, _ = pearsonr(ref_pitch, param)
		correlations.append(corr)
	return correlations


def calc_dtw(ref_pitch, homography_params, dist=2):
    """
    Calculate the Dynamic Time Warping (DTW) distance between the reference pitch
    and each homography parameter.

    :param ref_pitch (numpy.ndarray): Array of reference pitch values.
    :param homography_params (numpy.ndarray): 2D array of homography parameters.
                                               Shape: (num_frames, 9).
    :param dist (function): Distance metric to use for DTW.
    :return dtw_distances (list): List of DTW distances for each parameter.
    """
    ref_pitch = np.ravel(ref_pitch)
    dtw_distances = []
    print("homography_params shape:", homography_params.shape)
    for i in range(homography_params.shape[1]):
        param = np.ravel(homography_params[:, i])
        # dist of 2 is Euclidean distance
        distance, _ = fastdtw(ref_pitch, param, dist=dist)
        dtw_distances.append(distance)
    return dtw_distances


def test_homography_metrics():
    # Sample data for testing
    ref_pitch = np.linspace(0, 10, 5)  # Reference pitch values
    homography_params = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],   # Frame 1
        [2, 3, 4, 5, 6, 7, 8, 9, 10],  # Frame 2
        [3, 4, 5, 6, 7, 8, 9, 10, 11], # Frame 3
        [4, 5, 6, 7, 8, 9, 10, 11, 12],# Frame 4
        [5, 6, 7, 8, 9, 10, 11, 12, 13]# Frame 5
    ])

    # Test calc_correlation
    correlations = calc_correlation(ref_pitch, homography_params)
    print("Correlations:")
    for i, corr in enumerate(correlations):
        print(f"Param {i}: {corr:.3f}")

    # Test calc_dtw
    dtw_distances = calc_dtw(ref_pitch, homography_params, dist=2)
    print("DTW Distances:")
    for i, dtw in enumerate(dtw_distances):
        print(f"Param {i}: {dtw:.3f}")


if __name__ == '__main__':
    # Test Case 1: Perpendicular vectors
    normal1 = np.array([0.0, 0.0, 1.0])
    normal2 = np.array([1.0, 0.0, 0.0])
    angle_deg = calc_signed_angle_between_norms(normal1, normal2)
    print(f"Angle between {normal1} and {normal2}: {angle_deg} degrees")  # Expected: 90.0 degrees

    # Test Case 2: Vectors with negative rotation
    normal1 = np.array([1.0, 0.0, 0.0])
    normal2 = np.array([0.0, -1.0, 0.0])
    angle_deg = calc_signed_angle_between_norms(normal1, normal2)
    print(f"Angle between {normal1} and {normal2}: {angle_deg} degrees")  # Expected: -90.0 degrees

    # Test Case 3: Parallel vectors
    normal1 = np.array([1.0, 0.0, 0.0])
    normal2 = np.array([1.0, 0.0, 0.0])
    angle_deg = calc_signed_angle_between_norms(normal1, normal2)
    print(f"Angle between {normal1} and {normal2}: {angle_deg} degrees")  # Expected: 0.0 degrees

    # Test Case 4: Antiparallel vectors
    normal1 = np.array([1.0, 0.0, 0.0])
    normal2 = np.array([-1.0, 0.0, 0.0])
    angle_deg = calc_signed_angle_between_norms(normal1, normal2)
    print(f"Angle between {normal1} and {normal2}: {angle_deg} degrees")  # Expected: 180.0 degrees

    # Test Case 5: Vectors parallel to reference normal
    normal1 = np.array([0.0, 0.0, 1.0])
    normal2 = np.array([0.0, 0.0, -1.0])
    angle_deg = calc_signed_angle_between_norms(normal1, normal2)
    print(f"Angle between {normal1} and {normal2}: {angle_deg} degrees")  # Expected: 180.0 degrees

    # ---- DTW and Correlation Tests ---- #
    test_homography_metrics()