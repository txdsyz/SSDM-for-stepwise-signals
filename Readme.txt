#Acknowledgement
Parts of this code are adapted from yet-another-pytorch-tutorial-v2 (MIT License).

The core architecture of this project and its three key supporting files (“module.py”, “util.py”, “diffusion.py”) are derived from or modified based on the following open-source projects:
Choi, S. yet-another-pytorch-tutorial-v2，GitHub: https://github.com/sjchoi86/yet-another-pytorch-tutorial-v2  


1.Core Function
The SSDM model reconstructs the original signal from a noisy stepwise signal.

2.File Organization

Before running, ensure the following files are located in the same folder:
“Main.py”, “Train.py”, "Specialized.py",   "diffusion.py”, “util.py”, “module.py”.

3.Installation
Python Version: Python 3.12 is recommended (this project has been tested in a Python 3.12.3 environment).
Create a virtual environment (optional but recommended), install dependencies using the provided requirements file:  pip install -r requirements.txt

4.Usage
We provide three  scripts for different use cases:

(1) Inference with a pretrained model ("Main.py")
"main.py"is designed for direct testing using a pretrained model. No training is required.
Steps:
1. Download the pretrained model file  ("model.pth") in the ".pth" file.
2. Modify the following paths at the beginning of "main.py":
    "model_path",   "test_data_root"
3. Run: python main.py


(2) Training the model from scratch ("Train.py").
"Train.py" allows users to train the diffusion model.
Steps:
1. Modify dataset paths:
   "train_dataset",    "test_dataset"
2. Adjust the hyperparameter block in the script as needed(Optional)
3.Run training with:python Train.py


(3)Specialized models for different state numbers ("Specialized.py") 
"Specialized.py" provides specialized pretrained models for stepwise signals with different numbers of states (e.g., 2-state, 3-state, 4-state).

Steps:
1. Select the pretrained model in the ".pth" file corresponding to your signal type:
   "2-state.pth" for 2-state model
   "3-state.pth" for 3-state model
   "4-state.pth" for 4-state model
2. Modify the  "train_dataset"  in "Specialized.py" to point to the selected model
3. Modify "test_dataset" according to your dataset
4. Run:python Specialized.py


The dataset and pretrained models are available at Zenodo:

Dataset:
https://doi.org/10.5281/zenodo.18432190.

Pretrained models (.pth):
 https://doi.org/10.5281/zenodo.18417193.


This project is released under the MIT License.
