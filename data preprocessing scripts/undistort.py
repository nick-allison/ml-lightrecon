import os
import cv2
import numpy as np
import shutil  # For copying files directly

# Camera parameters for Freiburg 1, 2, and 3
CAMERA_PARAMS = {
    "freiburg1": {
        "K_color": np.array([[591.1, 0, 331.0],
                             [0, 590.1, 234.0],
                             [0, 0, 1]]),
        "D_color": np.array([-0.0410, 0.3286, 0.0087, 0.0051, -0.5643]),
        "K_depth": np.array([[591.1, 0, 331.0],
                             [0, 590.1, 234.0],
                             [0, 0, 1]]),
        "D_depth": np.array([-0.0410, 0.3286, 0.0087, 0.0051, -0.5643])
    },
    "freiburg2": {
        "K_color": np.array([[585.0, 0, 320.0],
                             [0, 585.0, 240.0],
                             [0, 0, 1]]),
        "D_color": np.array([-0.2297, 1.4766, -0.0002, -0.0004, -3.4199]),
        "K_depth": np.array([[585.0, 0, 320.0],
                             [0, 585.0, 240.0],
                             [0, 0, 1]]),
        "D_depth": np.array([-0.2297, 1.4766, -0.0002, -0.0004, -3.4199])
    },
    "freiburg3": {
        # No distortion parameters needed; images are already undistorted
        "K_color": None,
        "D_color": None,
        "K_depth": None,
        "D_depth": None
    }
}

def parse_sync_file(sync_file):
    """
    Parses a sync file to extract timestamps and relative paths.
    
    Args:
        sync_file (str): Path to the sync file (e.g., rgb.txt or depth.txt).
        
    Returns:
        list of tuples: Each tuple contains (timestamp, relative_path).
    """
    with open(sync_file, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith("#")]
    return [(line.split()[0], line.split()[1]) for line in lines]

def process_synced_images(input_dir, output_dir, dataset_type, rgb_sync_file, depth_sync_file):
    """
    Processes only synced rgb and depth images listed in the respective sync files.
    
    Args:
        input_dir (str): Directory containing the input dataset.
        output_dir (str): Directory to save the processed images.
        dataset_type (str): Dataset type ("freiburg1", "freiburg2", or "freiburg3").
        rgb_sync_file (str): Path to the sync file for RGB images.
        depth_sync_file (str): Path to the sync file for depth images.
    """
    if dataset_type not in CAMERA_PARAMS:
        raise ValueError("Invalid dataset_type. Use 'freiburg1', 'freiburg2', or 'freiburg3'.")

    # Determine processing mode based on dataset_type
    is_freiburg3 = dataset_type == "freiburg3"

    # Parse sync files
    rgb_sync = parse_sync_file(rgb_sync_file)
    depth_sync = parse_sync_file(depth_sync_file)

    # Process RGB and depth images
    for sync_data, subdir, camera_key in zip(
        [rgb_sync, depth_sync],
        ["rgb", "depth"],
        ["K_color", "K_depth"]
    ):
        input_subdir_path = os.path.join(input_dir, subdir)
        output_subdir_path = os.path.join(output_dir, "color" if subdir == "rgb" else "depth")

        # Ensure output subdirectory exists
        os.makedirs(output_subdir_path, exist_ok=True)

        for idx, (timestamp, relative_path) in enumerate(sync_data):
            img_path = os.path.join(input_dir, relative_path)
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} does not exist. Skipping.")
                continue

            if is_freiburg3:
                # For Freiburg 3, simply copy the file to the output directory
                output_path = os.path.join(output_subdir_path, f"{idx}.png" if subdir == "depth" else f"{idx}.jpg")
                shutil.copy(img_path, output_path)
                print(f"Copied {subdir} image to {output_path}")
            else:
                # Undistort for Freiburg 1 and 2
                K = CAMERA_PARAMS[dataset_type][camera_key]
                D = CAMERA_PARAMS[dataset_type][camera_key.replace("K", "D")]

                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Skipping invalid image file: {img_path}")
                    continue

                h, w = img.shape[:2]

                # Compute new camera matrix
                new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)

                # Undistort the image
                undistorted_img = cv2.undistort(img, K, D, None, new_K)

                # Determine output file format
                output_filename = f"{idx}.jpg" if subdir == "rgb" else f"{idx}.png"
                output_path = os.path.join(output_subdir_path, output_filename)
                cv2.imwrite(output_path, undistorted_img)

                print(f"Saved undistorted {subdir} image to {output_path}")


import os
import cv2
import numpy as np
import shutil  # For copying files directly

# Camera parameters for Freiburg 1, 2, and 3
CAMERA_PARAMS = {
    "freiburg1": {
        "K_color": np.array([[591.1, 0, 331.0],
                             [0, 590.1, 234.0],
                             [0, 0, 1]]),
        "D_color": np.array([-0.0410, 0.3286, 0.0087, 0.0051, -0.5643]),
        "K_depth": np.array([[591.1, 0, 331.0],
                             [0, 590.1, 234.0],
                             [0, 0, 1]]),
        "D_depth": np.array([-0.0410, 0.3286, 0.0087, 0.0051, -0.5643])
    },
    "freiburg2": {
        "K_color": np.array([[585.0, 0, 320.0],
                             [0, 585.0, 240.0],
                             [0, 0, 1]]),
        "D_color": np.array([-0.2297, 1.4766, -0.0002, -0.0004, -3.4199]),
        "K_depth": np.array([[585.0, 0, 320.0],
                             [0, 585.0, 240.0],
                             [0, 0, 1]]),
        "D_depth": np.array([-0.2297, 1.4766, -0.0002, -0.0004, -3.4199])
    },
    "freiburg3": {
        "K_color": np.array([[535.4, 0.0, 320.1],
                             [0.0, 539.2, 247.6],
                             [0.0, 0.0, 1.0]]),
        "D_color": np.zeros(5),  # All distortion coefficients are zero
        "K_depth": np.array([[580.8, 0.0, 308.8],
                             [0.0, 581.8, 253.0],
                             [0.0, 0.0, 1.0]]),
        "D_depth": np.zeros(5)  # All distortion coefficients are zero
    }
}

def parse_sync_file(sync_file):
    """
    Parses a sync file to extract timestamps and relative paths.
    
    Args:
        sync_file (str): Path to the sync file (e.g., rgb.txt or depth.txt).
        
    Returns:
        list of tuples: Each tuple contains (timestamp, relative_path).
    """
    with open(sync_file, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith("#")]
    return [(line.split()[0], line.split()[1]) for line in lines]

def process_synced_images(input_dir, output_dir, dataset_type, rgb_sync_file, depth_sync_file):
    """
    Processes only synced rgb and depth images listed in the respective sync files.
    
    Args:
        input_dir (str): Directory containing the input dataset.
        output_dir (str): Directory to save the processed images.
        dataset_type (str): Dataset type ("freiburg1", "freiburg2", or "freiburg3").
        rgb_sync_file (str): Path to the sync file for RGB images.
        depth_sync_file (str): Path to the sync file for depth images.
    """
    if dataset_type not in CAMERA_PARAMS:
        raise ValueError("Invalid dataset_type. Use 'freiburg1', 'freiburg2', or 'freiburg3'.")

    # Parse sync files
    rgb_sync = parse_sync_file(rgb_sync_file)
    depth_sync = parse_sync_file(depth_sync_file)

    # Process RGB and depth images
    for sync_data, subdir, camera_key in zip(
        [rgb_sync, depth_sync],
        ["rgb", "depth"],
        ["K_color", "K_depth"]
    ):
        input_subdir_path = os.path.join(input_dir, subdir)
        output_subdir_path = os.path.join(output_dir, "color" if subdir == "rgb" else "depth")

        # Ensure output subdirectory exists
        os.makedirs(output_subdir_path, exist_ok=True)

        for idx, (timestamp, relative_path) in enumerate(sync_data):
            img_path = os.path.join(input_dir, relative_path)
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} does not exist. Skipping.")
                continue

            # Get intrinsic matrix and distortion coefficients
            K = CAMERA_PARAMS[dataset_type][camera_key]
            D = CAMERA_PARAMS[dataset_type][camera_key.replace("K", "D")]

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Skipping invalid image file: {img_path}")
                continue

            h, w = img.shape[:2]

            # Compute new camera matrix
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1)

            # Undistort the image
            undistorted_img = cv2.undistort(img, K, D, None, new_K)

            # Determine output file format
            output_filename = f"{idx}.jpg" if subdir == "rgb" else f"{idx}.png"
            output_path = os.path.join(output_subdir_path, output_filename)
            cv2.imwrite(output_path, undistorted_img)

            print(f"Saved undistorted {subdir} image to {output_path}")


input_base = "C:\\Users\\NickA\\OneDrive\\Documents\\IIT Fall 2024\\CS 512\\TUM RGB-D\\rgbd_dataset_freiburg"
output_base = "C:\\Users\\NickA\\OneDrive\\Documents\\IIT Fall 2024\\CS 512\\ml-finerecon\\data\\scan "

inputs = ["1_plant", "1_teddy", "2_coke", "2_dishes", "2_flowerbouquet", "2_flowerbouquet_brownbackground", "2_metallic_sphere", "2_metallic_sphere2", "3_cabinet", "3_large_cabinet", "3_teddy"]
outputs = [str(i) for i in range(11)]
ds = ["freiburg1", "freiburg1", "freiburg2", "freiburg2", "freiburg2", "freiburg2", "freiburg2", "freiburg2", "freiburg3", "freiburg3", "freiburg3"]


if __name__ == "__main__":
    # Example usage
    input_directory = "path/to/your/input/dataset"  # Update with your input dataset directory
    output_directory = "path/to/save/processed/dataset"  # Update with your output dataset directory
    dataset = "freiburg1"  # Choose between "freiburg1", "freiburg2", or "freiburg3"
    rgb_sync_file_path = "path/to/sync/rgb.txt"  # Update with path to rgb.txt
    depth_sync_file_path = "path/to/sync/depth.txt"  # Update with path to depth.txt


    for i in range(8, 10):
        input_directory = input_base + inputs[i]  # Update with your input dataset directory
        output_directory = output_base + outputs[i]  # Update with your output dataset directory
        dataset = ds[i]  # Choose between "freiburg1", "freiburg2", or "freiburg3"
        rgb_sync_file_path = input_base + inputs[i] + "\\sync\\rgb.txt"  # Update with path to rgb.txt
        depth_sync_file_path = input_base + inputs[i] + "\\sync\\depth.txt"  # Update with path to depth.txt

        process_synced_images(input_directory, output_directory, dataset, rgb_sync_file_path, depth_sync_file_path)

    #undistort_images(input_directory, output_directory, dataset)

