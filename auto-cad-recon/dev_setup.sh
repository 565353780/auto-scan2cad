cd ..
git clone git@github.com:565353780/mesh-manage.git
git clone git@github.com:565353780/udf-generate.git
git clone git@github.com:565353780/scannet-dataset-manage.git
git clone git@github.com:565353780/scan2cad-dataset-manage.git
git clone git@github.com:565353780/shapenet-dataset-manage.git
git clone git@github.com:565353780/scannet-sim-manage.git
git clone git@github.com:565353780/conv-onet.git
git clone git@github.com:565353780/points-shape-detect.git
git clone git@github.com:565353780/global-pose-refine.git
git clone git@github.com:565353780/global-to-patch-retrieval.git
git clone git@github.com:565353780/implicit-3d-understanding.git
git clone git@github.com:565353780/image-to-cad.git
git clone git@github.com:565353780/noc-transform.git
git clone git@github.com:565353780/pytorch-3d-r2n2.git

cd mesh-manage
./dev_setup.sh

cd ../udf-generate
./dev_setup.sh

cd ../scannet-dataset-manage
./dev_setup.sh

cd ../scan2cad-dataset-manage
./dev_setup.sh

cd ../shapenet-dataset-manage
./dev_setup.sh

cd ../scannet-sim-manage
./dev_setup.sh

cd ../conv-onet
./dev_setup.sh

cd ../points-shape-detect
./dev_setup.sh

cd ../global-pose-refine
./dev_setup.sh

cd ../global-to-patch-retrieval
./dev_setup.sh

cd ../implicit-3d-understanding
./dev_setup.sh

cd ../image-to-cad
./dev_setup.sh

cd ../noc-transform
./dev_setup.sh

cd ../pytorch-3d-r2n2
./dev_setup.sh

# For pre-installing broke package
pip install antlr4-python3-runtime==4.9.3
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python getch

# For Ubuntu20.04 only
pip install numba==0.55 matplotlib==3.6

# For rendering
pip install -U open3d