from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad
from habitat_sim_manage.Data.pose import Pose
from habitat_sim_manage.Module.data_collector import DataCollector

def demo():
    glb_file_path = \
        '/home/chli/chLi/Dataset/ScanNet/scans/scene0474_02/scene0474_02_vh_clean.glb'
    control_mode = "pose"
    save_dataset_folder_path = './output/scene0474_02_vh_clean/'
    wait_key = 1

    data_collector = DataCollector(glb_file_path, control_mode, save_dataset_folder_path)

    # data_collector.pose_controller.pose = Pose(Point(4, -0.5, -4.2), Rad(0.2, 0.0))
    data_collector.pose_controller.pose = Pose(Point(0, 0, 0), Rad(0.0, 0.0))
    data_collector.sim_loader.setAgentState(
        data_collector.pose_controller.getAgentState())

    data_collector.startKeyBoardControlRender(wait_key)
    return True
