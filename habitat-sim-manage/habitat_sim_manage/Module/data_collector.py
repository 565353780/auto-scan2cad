import os
import cv2
import shutil
import numpy as np
from copy import deepcopy
from getch import getch
from scipy.spatial.transform import Rotation as R


from habitat_sim_manage.Config.config import SIM_SETTING
from habitat_sim_manage.Module.sim_manager import SimManager

class DataCollector(SimManager):
    def __init__(self, glb_file_path=None, control_mode=None, save_dataset_folder_path=None):
        super().__init__()

        self.save_dataset_folder_path = None
        self.image_folder_path = None
        self.sparse_folder_path = None
        self.image_pose_file = None
        self.image_idx = 1

        if glb_file_path is not None:
            assert self.loadSettings(glb_file_path)

        if control_mode is not None:
            assert self.setControlMode(control_mode)

        if save_dataset_folder_path is not None:
            assert self.createDataset(save_dataset_folder_path)
        return

    def reset(self):
        super().reset()
        self.save_dataset_folder_path = None
        self.image_folder_path = None
        self.sparse_folder_path = None
        self.image_pose_file = None
        self.image_idx = 1
        return True

    def saveSceneInfo(self):
        hfov = None
        for sensor in self.sim_loader.cfg.agents[0].sensor_specifications:
            if sensor.uuid == 'color_sensor':
                hfov = float(self.sim_loader.cfg.agents[0].sensor_specifications[0].hfov) * np.pi / 180.
                break

        if hfov is None:
            print('[ERROR][DataCollector::saveSceneInfo]')
            print('\t hfov get failed! please check your camera name and update me!')
            return False

        focal = SIM_SETTING['width'] / 2.0 / np.tan(hfov / 2.0)

        camera_txt = '1 PINHOLE ' + \
            str(SIM_SETTING['width']) + ' ' + \
            str(SIM_SETTING['height']) + ' ' + \
            str(focal) + ' ' + \
            str(focal) + ' ' + \
            str(SIM_SETTING['width'] / 2.0) + ' ' + \
            str(SIM_SETTING['height'] / 2.0)

        with open(self.sparse_folder_path + 'cameras.txt', 'w') as f:
            f.write(camera_txt + '\n')

        points_txt = '1 0 0 0 0 0 0 0 1'
        with open(self.sparse_folder_path + 'points3D.txt', 'w') as f:
            f.write(points_txt + '\n')
        return True

    def createDataset(self, save_dataset_folder_path):
        self.save_dataset_folder_path = save_dataset_folder_path

        if self.save_dataset_folder_path[-1] != '/':
            self.save_dataset_folder_path += '/'

        if os.path.exists(self.save_dataset_folder_path):
            shutil.rmtree(self.save_dataset_folder_path)

        self.image_folder_path = self.save_dataset_folder_path + 'images/'
        self.sparse_folder_path = self.save_dataset_folder_path + 'sparse/0/'
        self.image_pose_file = self.sparse_folder_path + 'images.txt'

        os.makedirs(self.image_folder_path, exist_ok=True)
        os.makedirs(self.sparse_folder_path, exist_ok=True)

        if not self.saveSceneInfo():
            print('[ERROR][DataCollector::createDataset]')
            print('\t saveSceneInfo failed!')
            return False

        return True

    def getCameraPose(self, pos, quat, axis='+x-z+y'):
        sign_map = {
            '+': 1.0,
            '-': -1.0,
        }
        axis_map = {
            'x': 0,
            'y': 1,
            'z': 2,
        }

        x_sign = sign_map[axis[0]]
        y_sign = sign_map[axis[2]]
        z_sign = sign_map[axis[4]]

        x_idx = axis_map[axis[1]]
        y_idx = axis_map[axis[3]]
        z_idx = axis_map[axis[5]]

        quat_list = [-quat.x, -quat.y, -quat.z, quat.w]

        matrix = R.from_quat(quat_list).as_matrix()
        dpos = - matrix.dot(pos)

        new_pos = [x_sign * dpos[x_idx], y_sign * dpos[y_idx], z_sign * dpos[z_idx]]
        new_quat = [quat_list[3], -1.0 * x_sign * quat_list[x_idx], -1.0 * y_sign * quat_list[y_idx], -1.0 * z_sign * quat_list[z_idx]]

        return new_pos, new_quat

    def getCameraPoseV2(self, pos, quat, axis='+x-z+y'):
        sign_map = {
            '+': 1,
            '-': -1,
        }
        axis_map = {
            'x': 0,
            'y': 1,
            'z': 2,
        }

        x_sign = sign_map[axis[0]]
        y_sign = sign_map[axis[2]]
        z_sign = sign_map[axis[4]]

        x_idx = axis_map[axis[1]]
        y_idx = axis_map[axis[3]]
        z_idx = axis_map[axis[5]]

        quat_list = [quat.x, quat.y, quat.z, quat.w]
        new_quat = [
            -1 * x_sign * quat_list[x_idx],
            -1 * y_sign * quat_list[y_idx],
            -1 * z_sign * quat_list[z_idx],
            quat_list[3]
        ]

        matrix = R.from_quat(new_quat).as_matrix()

        new_pos = [x_sign * pos[x_idx], y_sign * pos[y_idx], z_sign * pos[z_idx]]

        new_pos = - matrix.dot(new_pos)

        return new_pos, new_quat

    def saveImage(self, image):
        image = (image * 255.0).astype(np.uint8)

        cv2.imwrite(self.image_folder_path + str(self.image_idx) + '.png', image)

        agent_state = self.sim_loader.getAgentState()

        pos = deepcopy(agent_state.position)
        quat = deepcopy(agent_state.rotation)

        new_pos, new_quat = self.getCameraPose(pos, quat)

        pose_txt = str(self.image_idx)
        pose_txt += ' ' + str(new_quat[0])
        pose_txt += ' ' + str(new_quat[1])
        pose_txt += ' ' + str(new_quat[2])
        pose_txt += ' ' + str(new_quat[3])
        pose_txt += ' ' + str(new_pos[0])
        pose_txt += ' ' + str(new_pos[1])
        pose_txt += ' ' + str(new_pos[2])
        pose_txt += ' 1 '
        pose_txt += str(self.image_idx) + '.png\n'

        with open(self.image_pose_file, 'a') as f:
            f.write(pose_txt + '\n')

        self.image_idx += 1
        return True

    def startKeyBoardControlRender(self, wait_key):
        #  self.resetAgentPose()
        self.cv_renderer.init()

        while True:
            image = self.cv_renderer.renderFrame(self.sim_loader.observations, True)
            if image is None:
                break

            self.saveImage(image)

            self.cv_renderer.waitKey(wait_key)

            agent_state = self.sim_loader.getAgentState()
            print("agent_state: position", agent_state.position, "rotation",
                  agent_state.rotation)

            input_key = getch()
            if not self.keyBoardControl(input_key):
                break
        self.cv_renderer.close()
        return True
