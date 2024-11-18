import os

def align_timestamps(rgb_file, depth_file, groundtruth_file, output_dir, threshold=0.01):
    """
    Aligns the timestamps from RGB, depth, and groundtruth files within a specified threshold.
    Creates new text files with aligned data, ensuring that each original entry is used at most once.

    Args:
        rgb_file (str): Path to the file containing RGB timestamps.
        depth_file (str): Path to the file containing depth timestamps.
        groundtruth_file (str): Path to the groundtruth file.
        output_dir (str): Directory to save the filtered text files.
        threshold (float): Maximum allowed difference between timestamps for alignment.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and parse RGB, depth, and groundtruth files
    def load_file(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if not line.startswith("#") and line.strip()]

    rgb_lines = load_file(rgb_file)
    depth_lines = load_file(depth_file)
    groundtruth_lines = load_file(groundtruth_file)

    # Extract timestamps and data lines
    def extract_timestamps_and_data(lines, has_filename=True):
        timestamps = []
        data_lines = []
        for line in lines:
            values = line.split()
            if has_filename:
                timestamps.append(float(values[0]))
                data_lines.append(line)
            else:
                timestamps.append(float(values[0]))
                data_lines.append(line)
        return timestamps, data_lines

    rgb_timestamps, rgb_data = extract_timestamps_and_data(rgb_lines)
    depth_timestamps, depth_data = extract_timestamps_and_data(depth_lines)
    groundtruth_timestamps, groundtruth_data = extract_timestamps_and_data(
        groundtruth_lines, has_filename=False
    )

    # Initialize indices and used flags
    rgb_index = 0
    depth_index = 0
    rgb_used = set()
    depth_used = set()

    aligned_rgb = []
    aligned_depth = []
    aligned_groundtruth = []

    for gt_idx, gt_time in enumerate(groundtruth_timestamps):
        # Find the closest unused RGB timestamp
        while rgb_index < len(rgb_timestamps) and rgb_timestamps[rgb_index] < gt_time - threshold:
            rgb_index += 1

        rgb_match_idx = None
        if rgb_index < len(rgb_timestamps):
            if abs(rgb_timestamps[rgb_index] - gt_time) <= threshold and rgb_index not in rgb_used:
                rgb_match_idx = rgb_index
                rgb_used.add(rgb_match_idx)
                rgb_index += 1

        # Find the closest unused depth timestamp
        while depth_index < len(depth_timestamps) and depth_timestamps[depth_index] < gt_time - threshold:
            depth_index += 1

        depth_match_idx = None
        if depth_index < len(depth_timestamps):
            if abs(depth_timestamps[depth_index] - gt_time) <= threshold and depth_index not in depth_used:
                depth_match_idx = depth_index
                depth_used.add(depth_match_idx)
                depth_index += 1

        # If both matches are found, align them
        if rgb_match_idx is not None and depth_match_idx is not None:
            aligned_rgb.append(rgb_data[rgb_match_idx])
            aligned_depth.append(depth_data[depth_match_idx])
            aligned_groundtruth.append(groundtruth_data[gt_idx])

    # Save aligned data to new files
    def save_aligned_file(output_path, data):
        with open(output_path, 'w') as f:
            f.write("# Aligned data\n")
            for line in data:
                f.write(line + '\n')

    save_aligned_file(os.path.join(output_dir, os.path.basename(rgb_file)), aligned_rgb)
    save_aligned_file(os.path.join(output_dir, os.path.basename(depth_file)), aligned_depth)
    save_aligned_file(os.path.join(output_dir, os.path.basename(groundtruth_file)), aligned_groundtruth)

    print(f"Aligned files saved in {output_dir}")

inputs = ["1_plant", "1_teddy", "2_coke", "2_dishes", "2_flowerbouquet", "2_flowerbouquet_brownbackground", "2_metallic_sphere", "2_metallic_sphere2", "3_cabinet", "3_large_cabinet", "3_teddy"]

if __name__ == "__main__":
    # Example usage
    for i in range(11):
       rgb_file_path = "C:/Users/NickA/OneDrive/Documents/IIT Fall 2024/CS 512/TUM RGB-D/rgbd_dataset_freiburg" + inputs[i] + "/rgb.txt"         # Path to the RGB timestamps file
       depth_file_path = "C:/Users/NickA/OneDrive/Documents/IIT Fall 2024/CS 512/TUM RGB-D/rgbd_dataset_freiburg" + inputs[i] + "/depth.txt"     # Path to the depth timestamps file
       groundtruth_file_path = "C:/Users/NickA/OneDrive/Documents/IIT Fall 2024/CS 512/TUM RGB-D/rgbd_dataset_freiburg" + inputs[i] + "/groundtruth.txt"  # Path to the groundtruth file
       output_directory = "C:/Users/NickA/OneDrive/Documents/IIT Fall 2024/CS 512/TUM RGB-D/rgbd_dataset_freiburg" + inputs[i] + "/sync"  # Directory to save aligned files
       alignment_threshold = 0.01               # Threshold for timestamp alignment in seconds
       align_timestamps(rgb_file_path, depth_file_path, groundtruth_file_path, output_directory, alignment_threshold)
