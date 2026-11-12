git lfs install
git clone https://github.com
cd mlomics-sklearn

conda env create -f environment.yml
conda activate mlomics_sklearn

python -m ipykernel install --user --name=mlomics_sklearn --display-name "Python (MLOmics)"

jupyter lab