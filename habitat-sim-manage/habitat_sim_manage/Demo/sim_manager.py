from tqdm import tqdm

from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad
from habitat_sim_manage.Data.pose import Pose
from habitat_sim_manage.Module.sim_manager import SimManager

def demo_test_speed():
    glb_file_path = \
        "/home/chli/scan2cad/scannet/scans/scene0474_02/scene0474_02_vh_clean.glb"
    control_mode = "pose"

    sim_manager = SimManager()
    sim_manager.loadSettings(glb_file_path)
    sim_manager.setControlMode(control_mode)

    sim_manager.pose_controller.pose = Pose(Point(1.7, 1.5, -2.5),
                                            Rad(0.2, 0.0))
    sim_manager.sim_loader.setAgentState(
        sim_manager.pose_controller.getAgentState())

    input_key_list = sim_manager.pose_controller.input_key_list
    for i in tqdm(range(1000)):
        input_key = list(input_key_list)[i % (len(input_key_list) - 2)]
        sim_manager.keyBoardPoseControl(input_key)
    return True


def demo():
    glb_file_path = \
        "/home/chli/chLi/Dataset/ScanNet/scans/scene0474_02/scene0474_02_vh_clean.glb"
    control_mode = "circle"
    wait_key = 1

    sim_manager = SimManager()
    sim_manager.loadSettings(glb_file_path)
    sim_manager.setControlMode(control_mode)

    sim_manager.circle_controller.pose = Pose(Point(1.8, -0.25, -2.2),
                                              Rad(0.2, 0.0))
    sim_manager.sim_loader.setAgentState(
        sim_manager.pose_controller.getAgentState())

    sim_manager.startKeyBoardControlRender(wait_key)
    return True
