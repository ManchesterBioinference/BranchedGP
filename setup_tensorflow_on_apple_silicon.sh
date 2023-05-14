# First run `make install`, then run this script
pip install --upgrade pip
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal
conda install -c conda-forge -y pandas jupyter
pip install tensorflow_datasets

# Instructions from https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706
