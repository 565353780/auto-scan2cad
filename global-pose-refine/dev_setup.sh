cd ..
git clone https://github.com/565353780/auto-cad-recon.git
git clone https://github.com/565353780/mesh-manage.git
git clone https://github.com/565353780/udf-generate.git
git clone https://github.com/565353780/scannet-dataset-manage.git
git clone https://github.com/565353780/scan2cad-dataset-manage.git
git clone https://github.com/565353780/shapenet-dataset-manage.git
git clone https://github.com/565353780/scannet-sim-manage.git

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

pip install tqdm open3d numpy

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
