import numpy as np

# Replace 'your_file.npy' with the path to your .npy file
file_path = 'data/scan 9/pose.npy'

try:
    # Load the .npy file
    data = np.load(file_path)
    
    #Print the shape
    print("Shape of the .npy file:")
    print(data.shape)

    # Print the contents
    print("Contents of the .npy file:")
    print(data)
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred while reading the .npy file: {e}")
