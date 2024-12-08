# Initialize Conda (if needed)
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate the environment
conda create --name isc python=3.11 -y
conda activate isc

# Install required packages
conda install -c brainiak -c defaults -c conda-forge brainiak
conda install -c conda-forge nilearn numpy scipy pandas matplotlib scikit-learn nibabel seaborn h5py

# (Optional) Install Jupyter Lab
conda install -c conda-forge jupyterlab
python -m ipykernel install --user --name=isc --display-name "Python (isc)"

# Verify the installation
python -c "import brainiak; import nilearn; print('BrainIAK:', brainiak.__version__); print('Nilearn:', nilearn.__version__)"
