import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import pandaset


def read_pandaset(dataset_dir):
    """Loads dataset object from given path. 
       Required to read sequences.
    """
    dataset = pandaset.DataSet(dataset_dir)
    seq_list = sorted(dataset.sequences(with_semseg=False))
    return dataset, seq_list


def load_panda_seq_cam(dataset_obj, seq_list, seq):
    if isinstance(seq_list, str):
        seq_list = [seq_list]
    for curr_seq in seq_list:
        if curr_seq == seq:
            seq_obj = dataset_obj[curr_seq].load_camera()
            cam_obj = seq_obj.camera["front_camera"]
    return cam_obj


def load_panda_seq_gps(dataset_obj, seq_list, seq):
    if isinstance(seq_list, str):
        seq_list = [seq_list]
    for curr_seq in seq_list:
        if curr_seq == seq:
            seq_obj = dataset_obj[curr_seq].load_gps()
            gps_obj = seq_obj.gps[:]
    return gps_obj


def load_panda_seq_lidar(dataset_obj, seq_list, seq):
    if isinstance(seq_list, str):
        seq_list = [seq_list]
    for curr_seq in seq_list:
        if curr_seq == seq:
            seq_obj = dataset_obj[curr_seq].load_lidar()
            lidar_obj = seq_obj
    return lidar_obj

def parse_panda_seq_pc(lidar_obj, sensor = None):
    lidar_data = lidar_obj.lidar
    if sensor is not None:
        lidar_data.set_sensor(sensor)  # select a specific LiDAR sensor    
    pcs_np_xyz = [frame.to_numpy()[:, :3] for frame in lidar_data]
    return pcs_np_xyz

def parse_panda_seq_lidar_poses(lidar_obj, sensor = None):
    lidar_obj.lidar.set_sensor(sensor)
    poses = lidar_obj.lidar.poses[:]
    return poses

def parse_panda_seq_poses(cam_obj, gps_obj):
    abs_rot_mats = []
    pitch_xyz = []
    vels = []

    poses = cam_obj.poses[:]
    timestamps = cam_obj.timestamps[:]

    rel_timestamps = [t - timestamps[0] for t in timestamps]

    for i, pose in enumerate(poses):
        qw = pose["heading"]["w"]
        qx = pose["heading"]["x"]
        qy = pose["heading"]["y"]
        qz = pose["heading"]["z"]
        
        # create rotation object and convert to 3x3 matrix
        rotation = R.from_quat([qx, qy, qz, qw])
        rot_mat = rotation.as_matrix()  # 3x3 matrix
        abs_rot_mats.append(rot_mat)

        # calculate pitch for XYZ
        pitch_xyz_val = np.arctan2(
            -2.0 * (qy * qz - qw * qx), qw**2 - qx**2 - qy**2 + qz**2
        ) # in radians
        pitch_xyz.append( np.degrees(pitch_xyz_val) )
        
        # access velocity from GPS data
        gps_entry = gps_obj[i]
        yvel = gps_entry['yvel'] * 3.6
        xvel = gps_entry['xvel'] * 3.6
        vel = np.sqrt(yvel**2 + xvel**2)
        vels.append(vel)

    return rel_timestamps, abs_rot_mats, pitch_xyz, vels


def get_panda_seq_rel_rot_mats(abs_rot_mats):
    relative_rotations = []
    for i in range(len(abs_rot_mats) - 1):
        # absolute rotation matrices for consecutive frames
        R_i = abs_rot_mats[i]
        R_next = abs_rot_mats[i + 1]
        # compute relative rotation
        R_relative = R_i.T @ R_next
        relative_rotations.append(R_relative)
    return relative_rotations


def get_panda_seq_intrinsics(cam_obj):
    camera_K = cam_obj.intrinsics
    fx = camera_K.fx
    fy = camera_K.fy
    cx = camera_K.cx
    cy = camera_K.cy

    camera_K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    return camera_K


def pil_to_numpy(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_panda_seq_images(cam_obj, pil=False):
    images = cam_obj[:]
    if pil:
        return images
    images_np = [pil_to_numpy(img) for img in images] 
    return images_np
