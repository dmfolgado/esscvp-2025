### Fraunhofer's Workshop exercises

The repo is divided in the diferent themes:
 - machine learning (ML, traditional methods)
 - deep learning (DL)

### Installation
1. Install either miniconda (recommended) or conda: https://docs.anaconda.com/free/miniconda/miniconda-install/
2. Create the conda environment for the workshop: `conda env create -f environment.yml`
3. Activate the conda environment: `conda activate mldl-ws`
4. Install the ML virtual environment: `make build-env-ml`. If you are using Windows, run these 3 commands instead:
* `python -m venv .venv-ml`
* `.venv-ml\Scripts\activate`;
* `python -m pip install -r machine_learning/unsupervised/pca_requirements.txt -r machine_learning/supervised/clf_requirements.txt -r machine_learning/supervised/rgr_requirements.txt`
5. Install the DL virtual environment: `make build-env-dl`. If you are using Windows, you need to run these 3 commands instead:
* `python -m venv .venv-dl`
* `.venv-dl\Scripts\activate`
* `python -m pip install -r deep_learning/pytorch/requirements.txt`

* Note: You also need to install CUDA to train DL models on GPU: https://developer.nvidia.com/cuda-downloads.

6. Download the `data` and unzip it to the repository root folder




### Contributing

We are using tools to clean the code and make sure everything is good. On every commit the tools will clean your code.

Suggestion: use ```git add <your files>``` then run ```pre-commit``` then run again ```git add <your files>``` pre-commit will automatically make some changes.
