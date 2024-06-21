cd ..
git clone git@github.com:565353780/mesh-manage.git
git clone git@github.com:565353780/habitat-sim-manage.git
git clone git@github.com:565353780/detectron2-detect.git
git clone git@github.com:565353780/scene-layout-detect.git

cd mesh-manage
./dev_setup.sh

cd ../habitat-sim-manage
./dev_setup.sh

cd ../detectron2-detect
./dev_setup.sh

cd ../scene-layout-detect
./dev_setup.sh
