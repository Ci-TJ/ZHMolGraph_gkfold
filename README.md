# ZHMolGraph

RNA-protein interactions are critical to various life processes, including fundamental translation and gene regulation. Identifying these interactions is vital for understanding the mechanisms underlying life processes. Then, ZHMolGraph is an advanced pipeline that integrates graph neural network sampling strategy and unsupervised pre-trained large language models to enhance binding predictions for novel RNAs and proteins.


# Setting up ZHMolGraph and Predicting RNA-protein interactions

## Requirements

- We provide the script and model for validating the results of ZHMolGraph. Any machines with a GPU and an Ubuntu system should work.

- We recommend using Anaconda to create a virtual environment for this project.

- you will need a major software package: `pytorch`. The following commands will create a virtual environment and install the necessary packages. Note that we install the GPU version of PyTorch (`torch==1.8.1+cu11`) for training purpose.

```bash
conda create -n ZHMolRPGraphPytorch-1.8 python=3.8
conda activate ZHMolRPGraphPytorch-1.8
pip install tqdm
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard
pip install jupyter
```

- All Python modules and corresponding versions required for ZHMolGraph are listed here: requirements.txt

- Use pip install -r requirements.txt to install the related packages. 


# Code and Data

## Data Files

## Code 

Here we describe the Jupyter Notebooks scripts used in ZHMolGraph.

