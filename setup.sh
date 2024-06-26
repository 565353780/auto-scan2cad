git clone https://github.com/565353780/conv-onet.git
git clone https://github.com/565353780/implicit-3d-understanding.git
git clone https://github.com/565353780/image-to-cad.git
git clone https://github.com/565353780/pytorch-3d-r2n2.git

cd ./mesh-manage
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

cd ../habitat-sim-manage
./setup.sh

cd ../detectron2-detect
./setup.sh

cd ../scene-layout-detect
./setup.sh
