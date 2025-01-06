# LightRecon: Optimized FineRecon for Real-Time 3D Reconstruction

LightRecon is a computationally efficient version of FineRecon, designed for real-time 3D reconstruction. By incorporating sparse 3D convolutions, a refined voxel occupancy prediction network, simplified depth guidance, and adaptive point back-projection, LightRecon significantly reduces processing times and memory usage with minimal sacrifice in reconstruction quality.

---

## Features

- **Efficient 3D Convolutions**: Reduces computational cost by focusing on non-empty regions of voxel grids.
- **Voxel Occupancy Prediction Refinement**: Lightweight network with reduced channels and no residual blocks.
- **Simplified Depth Guidance**: Replaces standard CNNs with MobileNet for depth feature extraction.
- **Adaptive Point Back-Projection**: Enhances detail-oriented regions while reducing focus on less significant areas.

---
## Getting Started

### Prerequisites

Ensure you have Python installed along with the required dependencies:

~~~bash
pip install -r requirements.txt
~~~

**Note**: You may need to downgrade pip to install some packages successfully.
### Dataset Preparation

LightRecon supports the TUM RGB-D dataset, which can be downloaded from:

- [TUM RGB-D Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)

Additionally, preprocessed data for LightRecon is available:

- [Preprocessed Dataset](https://drive.google.com/file/d/1QQJmcM96_CU3hDliXw1D2AqMPR7tn3wC/view)
- [Ground Truth Data](https://drive.google.com/file/d/16iTexGrZkqrZq--cLHblcUU3PHZsNBzU/view)

This is a very small dataset, so it is recommended that you supplement it with data from other multi-view RGB-D datasets. The project includes scripts to preprocess and organize custom datasets.

---
### Configuration

Clone the repository:

~~~bash
git clone https://github.com/nick-allison/ml-finerecon.git
cd ml-finerecon
~~~

Modify the `config.yml` file:

- Set `dataset_dir` to the path where the dataset is stored.
- Set `pred_depth_dir` to the path of the depth predictions directory.
- Set `tsdf_dir` to the path of the ground truth TSDF directory.

If you extract the dataset and ground truth files into the root directory of the repository, you can use the default `config.yml` file.

---
### Running LightRecon

Open the Jupyter notebook `light_recon.ipynb`:

~~~bash
jupyter notebook light_recon.ipynb
~~~

Execute the cells to explore LightRecon’s functionality, including voxel grid processing, depth-guidance features, and 3D reconstruction outputs.

---
## Results

LightRecon demonstrates significant efficiency improvements compared to FineRecon:

| Component           | Original Params | Modified Params | Reduction (%) |
|---------------------|-----------------|-----------------|---------------|
| CNN2D               | 2,000,000      | 2,000,000       | 0%            |
| CNN3D               | 600,000        | 5,200           | 99.1%         |
| Occupancy Predictor | 7,000          | 545             | 92.2%         |

Total parameter reduction: ~15%.

---
## Limitations

While LightRecon achieves reduced computational cost, it remains too resource-intensive for effective training on personal computers. Future work may focus on further optimizations or cloud-based solutions.

---

## References

For detailed methodology and insights, refer to the accompanying project report. Key references include:

- [FineRecon: Depth-Guided Large-Scale 3D Reconstruction](https://arxiv.org/abs/2304.01480)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

---

## Contributing

Feel free to submit pull requests or open issues to suggest improvements and new features.

---

## License

- [License](https://github.com/nick-allison/ml-lightrecon/blob/main/LICENSE)
