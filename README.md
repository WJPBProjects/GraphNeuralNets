A repository for investigating graph data with Graph Neural Networks

Note that this is a limited selection of files and notebooks to demonstrate the work, due to size limitations on the original repo

## Setup PyTorch using conda

Create and activate a Python 3 conda environment:

   ```shell
   conda create -n pymira python=3.8
   conda activate pymira
   ```
   
Install PyTorch using conda:
   
   ```shell
   conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
   ```
remember to check your pytorch, torchvision and cudatoolkit versions are correct before moving on, e.g. for cuda using:
   
   ```shell
   python -c "import torch; print(torch.version.cuda)"
   ```
   
## Setup PyTorch using virtualenv

Create and activate a Python 3 virtual environment:

   ```shell
   virtualenv -p python3 <path_to_envs>/pymira
   source <path_to_envs>/pymira/bin/activate
   ```
   
Install PyTorch using pip:
   
   ```shell
   pip install torch torchvision
   ```
   
## Install PyTorch Geometric:

Note that the versions specified in the URLs must match with the versions installed above (e.g. 102 = 10.2 for cuda)
   
   ```shell   
   pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
   pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
   pip install -q torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html
   pip install -q torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu102.html   
   pip install -q torch-geometric
   ```
   
## Install additional Python packages:
   
   ```shell
   pip install matplotlib jupyter pandas seaborn scikit-learn tensorboard cmake openmesh pytorch-lightning
   ```
   
### License & Acknowledgements
This project is part of a summer MSc project at Imperial College London. Please contact the author for further details.
