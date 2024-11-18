import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_groundtruth(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            values = line.split()
            timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, values)
            # Quaternion to 4x4 transformation matrix
            rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [tx, ty, tz]
            poses.append(pose)
    return np.array(poses)

inputs = ["1_plant", "1_teddy", "2_coke", "2_dishes", "2_flowerbouquet", "2_flowerbouquet_brownbackground", "2_metallic_sphere", "2_metallic_sphere2", "3_cabinet", "3_large_cabinet", "3_teddy"]
outputs = ["data/scan " + str(i) + "/pose.npy" for i in range(11)]

for i in range(11):
    poses = parse_groundtruth("C:/Users/NickA/OneDrive/Documents/IIT Fall 2024/CS 512/TUM RGB-D/rgbd_dataset_freiburg" + inputs[i] + "/sync/groundtruth.txt")
    np.save(outputs[i], poses)
