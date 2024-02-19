# DistillNet: Fast Particle Reconstruction using Neural Networks

DistillNet is an implementation of the model described in the paper titled  ["Distilling particle knowledge for fast reconstruction at high-energy physics experiments"](https://doi.org/10.48550/arXiv.2311.12551) (A. Bal, T. Brandes, F. Iemmi, M. KLute, B. Maier, V. Mikuni, T. K. Arrestad). The model is designed for fast and efficient Pileup removal tasks in high-energy physics experiments.

## Getting Started

To use DistillNet, follow these steps:

1. **Download Repository**: Clone or download the DistillNet repository from [GitHub](https://github.com/tbrandes01/distillnet).

2. **Install Dependencies**: Install the required dependencies listed in `requirements.txt` by running:
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/lsowa/tayloranalysis.git
    ```
3. **Configuration**: Modify the configuration file (distillnet_config.py) to set the desired filepaths for the downloaded dataset ([link to dataset](https://doi.org/10.5281/zenodo.10670183)), as well as the results, and other parameters.

4. **Training**: Execute the ```train_distillnet.py``` script to train the DistillNet model using the downloaded dataset.

5. **Results**: The training script automatically tests the model on Missing Transverse Energy performance after training, which can be found in the Results directory. For jet clustering, the script ```cluster_jets.py``` can be executed. 

#### Taylor Analysis Module
For further analysis, you can utilize the Taylor Analysis Module available at from [GitHub](https://github.com/lsowa/tayloranalysis). This module assists in understanding the importance of input features for the neural network's output.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
- [Tristan Brandes](https://github.com/tbrandes01)
