pip install -U torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install -U pillow==9.5.0

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install opencv-python
