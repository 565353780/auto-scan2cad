cd ..
git clone https://github.com/565353780/mesh-manage.git
git clone https://github.com/565353780/udf-generate.git
git clone https://github.com/565353780/scannet-dataset-manage.git
git clone https://github.com/565353780/scan2cad-dataset-manage.git
git clone https://github.com/565353780/shapenet-dataset-manage.git
git clone https://github.com/565353780/scannet-sim-manage.git
git clone https://github.com/565353780/conv-onet.git
git clone https://github.com/565353780/points-shape-detect.git
git clone https://github.com/565353780/global-pose-refine.git
git clone https://github.com/565353780/global-to-patch-retrieval.git
git clone https://github.com/565353780/implicit-3d-understanding.git
git clone https://github.com/565353780/image-to-cad.git
git clone https://github.com/565353780/noc-transform.git
git clone https://github.com/565353780/pytorch-3d-r2n2.git

cd mesh-manage
./setup.sh

cd ../udf-generate
./setup.sh

cd ../scannet-dataset-manage
./setup.sh

cd ../scan2cad-dataset-manage
./setup.sh

cd ../shapenet-dataset-manage
./setup.sh

cd ../scannet-sim-manage
./setup.sh

cd ../conv-onet
./setup.sh

cd ../points-shape-detect
./setup.sh

cd ../global-pose-refine
./setup.sh

cd ../global-to-patch-retrieval
./setup.sh

cd ../implicit-3d-understanding
./setup.sh

cd ../image-to-cad
./setup.sh

cd ../noc-transform
./setup.sh

cd ../pytorch-3d-r2n2
./setup.sh

# For pre-installing broke package
pip install antlr4-python3-runtime==4.9.3
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python getch

# For Ubuntu20.04 only
pip install numba==0.55 matplotlib==3.6

# For rendering
pip install -U open3d
