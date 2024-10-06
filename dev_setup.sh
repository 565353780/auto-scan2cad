git clone git@github.com:565353780/conv-onet.git
git clone git@github.com:565353780/implicit-3d-understanding.git
git clone git@github.com:565353780/image-to-cad.git
git clone git@github.com:565353780/pytorch-3d-r2n2.git

cd ./mesh-manage
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

cd ../habitat-sim-manage
./dev_setup.sh

cd ../detectron2-detect
./dev_setup.sh

cd ../scene-layout-detect
./dev_setup.sh

# post fix
pip install -U torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U open3d
