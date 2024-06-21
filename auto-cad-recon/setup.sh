# For pre-installing broke package
pip install antlr4-python3-runtime==4.9.3
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python getch

# For Ubuntu20.04 only
pip install numba==0.55 matplotlib==3.6

# For rendering
pip install -U open3d
