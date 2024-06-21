from getch import getch

from habitat_sim_manage.Module.sim_loader import SimLoader
from habitat_sim_manage.Module.controller.action_controller import ActionController
from habitat_sim_manage.Module.controller.pose_controller import PoseController
from habitat_sim_manage.Module.controller.circle_controller import CircleController
from habitat_sim_manage.Module.renderer.cv_renderer import CVRenderer


class SimManager(object):

    def __init__(self):
        self.sim_loader = SimLoader()
        self.action_controller = ActionController()
        self.pose_controller = PoseController()
        self.circle_controller = CircleController()
        self.cv_renderer = CVRenderer()
        self.control_mode_dict = {
            "action": self.keyBoardActionControl,
            "pose": self.keyBoardPoseControl,
            "circle": self.keyBoardCircleControl,
        }
        self.control_mode_list = self.control_mode_dict.keys()

        self.control_mode = "pose"
        return

    def reset(self):
        self.sim_loader.reset()
        self.action_controller.reset()
        self.pose_controller.reset()
        self.cv_renderer.reset()
        self.control_mode_dict = {
            "action": self.keyBoardActionControl,
            "pose": self.keyBoardPoseControl,
        }
        self.control_mode_list = self.control_mode_dict.keys()

        self.control_mode = "pose"
        return

    def loadSettings(self, glb_file_path):
        self.sim_loader.loadSettings(glb_file_path)
        return True

    def setControlMode(self, control_mode):
        if control_mode not in self.control_mode_list:
            print("[WARN][SimManager::setControlMode]")
            print("\t control_mode not valid! set to [pose] mode")
            return True
        self.control_mode = control_mode
        return True

    def resetAgentPose(self):
        init_agent_state = self.pose_controller.getInitAgentState()
        self.sim_loader.setAgentState(init_agent_state)
        return True

    def keyBoardActionControl(self, input_key):
        if input_key == "q":
            return False

        action = self.action_controller.getAction(input_key)
        if action is None:
            print("[WARN][SimManager::keyBoardActionControl]")
            print("\t input key not valid!")
            return True

        self.sim_loader.stepAction(action)
        return True

    def keyBoardPoseControl(self, input_key):
        if input_key == "q":
            return False

        agent_state = self.pose_controller.getAgentStateByKey(input_key)

        self.sim_loader.setAgentState(agent_state)
        return True

    def keyBoardCircleControl(self, input_key):
        if input_key == "q":
            return False

        agent_state = self.circle_controller.getAgentStateByKey(input_key)

        self.sim_loader.setAgentState(agent_state)
        return True

    def keyBoardControl(self, input_key):
        return self.control_mode_dict[self.control_mode](input_key)

    def startKeyBoardControlRender(self, wait_key):
        #  self.resetAgentPose()
        self.cv_renderer.init()

        while True:
            if not self.cv_renderer.renderFrame(self.sim_loader.observations):
                break
            self.cv_renderer.waitKey(wait_key)

            agent_state = self.sim_loader.getAgentState()
            print("agent_state: position", agent_state.position, "rotation",
                  agent_state.rotation)

            input_key = getch()
            if not self.keyBoardControl(input_key):
                break
        self.cv_renderer.close()
        return True
