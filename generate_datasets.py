import sys

sys.path.append("./scannet-dataset-manage/")
sys.path.append("./scan2cad-dataset-manage/")
sys.path.append("./udf-generate/")

from scannet_dataset_manage.Demo.object_spliter import demo as demo_split_object
from scannet_dataset_manage.Demo.object_bbox_generator import demo as demo_generate_object_bbox
from scannet_dataset_manage.Demo.glb_generator import demo as demo_generate_glb

from scan2cad_dataset_manage.Demo.object_model_map_generator import demo as demo_generate_object_model_map

from udf_generate.Demo.udf_generate_manager import demo as demo_generate_udf_folder

if __name__ == "__main__":
    #TODO: you need to edit the dataset path in these demo functions

    demo_split_object() # --> "ScanNet/objects/"
    demo_generate_object_bbox() # --> "ScanNet/bboxes/"
    demo_generate_glb() # --> "ScanNet/glb/"

    demo_generate_object_model_map() # --> "Scan2CAD/object_model_maps/"

    demo_generate_udf_folder() # --> "ShapeNet/udfs/"
