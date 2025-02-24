import warnings

import cv2
import numpy as np

# OpenCV and NumPy coordinate order: (x, y) or (width, height)
# Origin: top-left corner (as in computer vision)
# point order: (top-left, top-right, bottom-right, bottom-left) / ROI, src, dst /

### --- Basic homography --- ###

def read_img(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img


def proc_img(img, size=None):
    if size is not None:
        img = cv2.resize(img, size)
    return img


def show_img(img, title="Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calc_homography(src_pts, dst_pts, method=cv2.RANSAC, threshold=5.0):
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    try:
        hg_mat, mask = cv2.findHomography(src_pts, dst_pts, method, threshold)
    except cv2.error as e:
        warnings.warn(f"Error in findHomography: {e}", UserWarning)
        return None, None
    if hg_mat is None:
        warnings.warn("Could not compute homography matrix", UserWarning)
        return None, None  # return sentinel values for outside handling
    return hg_mat, mask


def warp_img(img, hg_mat, width_coords=None, height_coords=None):
    """ Warps an image to a new perspective using a homography matrix.
        
        If width_coords and height_coords are provided, the output
        size is calculated from them (useful for BEV transformation, 
        width can be the lower two points of the ROI).
    """
    if width_coords is None or height_coords is None:
        width = img.shape[1]
        height = img.shape[0]
    else:
        width = abs(width_coords[1] - width_coords[0])
        height = abs(height_coords[1] - height_coords[0])
        
    img_warped = np.zeros((height, width, 3), dtype=np.uint8)
    img_warped = cv2.warpPerspective(
        img, hg_mat, (width, height), borderMode=cv2.BORDER_CONSTANT
    )
    return img_warped


def generate_roi_mask(img, roi_pts):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(roi_pts)], 255)
    return mask

### --- Keypoint detection and matching functions --- ###

def detect_keypoints(img, detector_type="ORB", mask=None):
    if detector_type == "ORB":
        detector = cv2.ORB_create()
    elif detector_type == "SIFT":
        detector = cv2.SIFT_create()
    else:
        raise ValueError("Supported detectors: 'ORB' or 'SIFT'.")
    keypoints, descriptors = detector.detectAndCompute(img, mask=mask)
    return keypoints, descriptors


def match_features(desc1, desc2, detector_type="ORB"):
    if detector_type in ["ORB"]:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif detector_type in ["SIFT"]:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:
        raise ValueError("Supported detectors: 'ORB' or 'SIFT'.")
    matches = bf.match(desc1, desc2)
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

### --- Filtering functions --- ###

def filter_matches(matches, keypoints1, keypoints2, num_points=4):
    """Filter and select keypoints based on matches. Returns 
       points in the format required for cv2.findHomography.
       Use a minimum of 4 points but 15-30 points are recommended.
    """
    # `keypoints1` for src_pts (from img1) and `keypoints2` for dst_pts (to img2)
    src_pts = [keypoints1[m.queryIdx].pt for m in matches[:num_points]]
    dst_pts = [keypoints2[m.trainIdx].pt for m in matches[:num_points]]
    
    # switch the order when unpacking to find homography for the other direction
    return np.float32(src_pts).reshape(-1, 1, 2), np.float32(dst_pts).reshape(-1, 1, 2)

### --- Visualization functions --- ###

def visualize_features(img, kpts, color=(0, 255, 0)):
    img_with_kpts = cv2.drawKeypoints(img, kpts, None, color=color)
    return img_with_kpts

  
def visualize_matches(img1, img2, kpts1, kpts2, matches):
    img_with_matches = cv2.drawMatches(img1, kpts1, img2, kpts2, matches, None, flags=2)
    return img_with_matches


def visualize_ransac_matches(img1, img2, keypoints1, keypoints2, matches, mask):
    """
    Visualizes inlier and outlier matches based on the RANSAC mask.
    
    Parameters:
        img1, img2: Images between which keypoints are matched.
        keypoints1, keypoints2: Keypoints detected in img1 and img2.
        mask: RANSAC mask indicating inliers (1) and outliers (0) 
              (returned by cv2.findHomography).
    
    Returns:
        Displays images with inlier matches in green and outlier matches in red.
    """
    # Separate matches into inliers and outliers based on the RANSAC mask
    inlier_matches = [matches[i] for i in range(len(mask)) if mask[i]]
    outlier_matches = [matches[i] for i in range(len(mask)) if not mask[i]]
    
    # Visualize inliers and outliers
    img_inliers = visualize_matches(img1, img2, keypoints1, keypoints2, inlier_matches)
    img_outliers = visualize_matches(img1, img2, keypoints1, keypoints2, outlier_matches)
    
    # Display inliers and outliers
    show_img(img_inliers, title="Inlier Matches")
    show_img(img_outliers, title="Outlier Matches")


# TODO: revise this
def visualize_ransac_matches_combined(img1, img2, keypoints1, keypoints2, matches, mask):
    """
    Visualizes inlier and outlier matches together based on the RANSAC mask.

    Parameters:
        img1, img2 (numpy.ndarray): Images between which keypoints are matched.
        keypoints1, keypoints2 (list of cv2.KeyPoint): Keypoints from both images.
        matches (list of cv2.DMatch): Matched keypoints.
        mask (numpy.ndarray): RANSAC mask indicating inliers (1) and outliers (0)
                              (returned by cv2.findHomography).

    Returns:
        numpy.ndarray: Image visualizing both inlier and outlier matches.
    """
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    height = max(height1, height2)
    width = width1 + width2
    combined_img = np.zeros((height, width, 3), dtype=np.uint8)
    combined_img[:height1, :width1] = img1
    combined_img[:height2, width1:] = img2

    # draw both inliers and outliers on the combined image
    for i, match in enumerate(matches[:len(mask)]):  # Limit to min(len(matches), len(mask))
        pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype(int) + np.array([width1, 0]))
        color = (0, 255, 0) if mask[i] else (0, 0, 255)  # Green for inliers, red for outliers
        cv2.line(combined_img, pt1, pt2, color, 1)

    return combined_img


### --- Preproccesing functions --- ###

def calculate_scaled_rois(rois, src_img_size, dst_img_size):
	src_width, src_height = src_img_size
	dst_width, dst_height = dst_img_size

	# scale factors for width and height
	scale_x = dst_width / src_width
	scale_y = dst_height / src_height

	# scale each ROI
	scaled_rois = []
	for roi in rois:
		scaled_roi = [
			(int(point[0] * scale_x), int(point[1] * scale_y)) for point in roi
		]
		scaled_rois.append(scaled_roi)
	return scaled_rois


### --- Postproccesing functions --- ###

def get_cam_to_lidar_rot(do_rot_z=True, rot_z_deg=180):
    """
    Returns a rotation matrix to transform the camera coordinate system
    into the LiDAR coordinate system (right-handed y-backward).If you
    chage rot_z to 90, it will transform into right-handed x-forward.
    """
    # rotation +90 degrees around X-axis
    radians_x = np.deg2rad(90)
    cos_x = np.cos(radians_x)
    sin_x = np.sin(radians_x)

    R_x_90 = np.array([
        [1,      0,       0],
        [0,  cos_x,  -sin_x],
        [0,  sin_x,   cos_x]
    ])
    # rotation +180 degrees around Z-axis
    radians_z = np.deg2rad(rot_z_deg)
    cos_z = np.cos(radians_z)
    sin_z = np.sin(radians_z)

    R_z_180 = np.array([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [0,      0,     1]
    ])
    # combined rotation
    if do_rot_z:
        R = R_z_180 @ R_x_90
    else:
        R = R_x_90
    return R


def calculate_pitch_from_normal(n_norm):
    """
    Calculate pitch angle from the normalized ground plane normal vector 
    in a right-handed y-backward reference frame.

    Parameters:
        n_norm (numpy.ndarray): Normalized ground plane normal vector [x, y, z].

    Returns:
        float: Pitch angle in degrees.
    
    Raises:
        ValueError: If the input is not a 3-element numpy array or is a zero vector.
    
    Example:
        >>> calculate_pitch_from_normal(np.array([0, 0, 1]))
        0.0
        >>> calculate_pitch_from_normal(np.array([0, -1, 0]))
        90.0
        >>> calculate_pitch_from_normal(np.array([0, 1, 0]))
        -90.0
    """
    # input validation
    if not isinstance(n_norm, np.ndarray):
        raise TypeError("Input normal vector must be a numpy array.")
    if n_norm.shape != (3,):
        raise ValueError("Input normal vector must have shape (3,).")
    
    # ensure the normal vector is not the zero vector
    norm = np.linalg.norm(n_norm)
    if norm == 0:
        raise ValueError("Normal vector must not be the zero vector.")
    
    # normalize
    n_norm = n_norm / norm
    
    # calculate pitch using arctan2 for numerical stability
    pitch_rad = np.arctan2(-n_norm[1], np.sqrt(n_norm[0]**2 + n_norm[2]**2))
    pitch_deg = np.degrees(pitch_rad)
    
    return pitch_deg


### --- Debugging functions --- ###

def print_homography(hg_mat):
    print("Homography Matrix:")
    for row in hg_mat:
        print("[" + " ".join(f"{val:8.2f}" for val in row) + "]")