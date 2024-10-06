from scannet_dataset_manage.Demo.object_spliter import demo as demo_split_object
from scannet_dataset_manage.Demo.object_bbox_generator import demo as demo_generate_object_bbox
from scannet_dataset_manage.Demo.glb_generator import demo as demo_generate_glb

if __name__ == "__main__":
    #TODO: you need to edit the dataset path in these demo functions
    demo_split_object() # --> "ScanNet/objects/"
    demo_generate_object_bbox() # --> "ScanNet/bboxes/"
    demo_generate_glb() # --> "ScanNet/glb/"
