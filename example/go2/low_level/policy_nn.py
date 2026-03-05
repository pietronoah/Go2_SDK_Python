
import time
import sys
import os
import math
import numpy as np
import torch
import pygame

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2w
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from utils import scale_axis, quat_rotate_inverse
from config_loader.config_loader import load_config, load_actor_network


# ---------------- Constants ----------------
JOYSTICK_THRESHOLD = 0.05
CONTROL_INTERVAL = 0.02  # 50 Hz
DEFAULT_KP = 25.0
DEFAULT_KD = 0.5

# ---------------- Joystick Handler ----------------
class JoystickHandler:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Detected joystick: {self.joystick.get_name()}")
        else:
            print("No joystick connected.")

    def get_commands(self, fallback=np.zeros(3)) -> np.ndarray:
        if not self.joystick:
            return fallback

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return fallback

        axes = [self.joystick.get_axis(i) for i in [0, 1, 2, 5]]
        summed_axes = (1 + axes[2]) - (1 + axes[3])
        axes = np.array([axes[0], axes[1], summed_axes])

        scaled_axes = [scale_axis(i, axes[i]) for i in range(len(axes))]
        scaled_axes[0], scaled_axes[1] = scaled_axes[1], scaled_axes[0]

        # Apply threshold
        return np.array([x if abs(x) >= JOYSTICK_THRESHOLD else 0 for x in scaled_axes])

    def safety_button_pressed(self) -> bool:
        if not self.joystick:
            return False
        return bool(self.joystick.get_button(3))
    
    def sit_down_button_pressed(self) -> bool:
        if not self.joystick:
            return False
        pygame.event.pump()
        return bool(self.joystick.get_button(0))
    
    def stand_up_button_pressed(self) -> bool:
        if not self.joystick:
            return False
        pygame.event.pump()
        return bool(self.joystick.get_button(1))


# ---------------- Policy Class ----------------
class Policy:
    def __init__(self, config_path='config.yaml', joystick=None):
        config = load_config(config_path)
        self.default_joint_angles = np.array(config['robot']['default_joint_angles'])

        # Policy selection from config.yaml
        policy_cfg = config.get('policy', {})
        self.gait = policy_cfg.get('gait', 'trot')  # 'trot' or 'pronk'

        # model path resolution: prefer explicit mapping in config, fall back to defaults
        model_paths = policy_cfg.get('model_paths', {})
        default_paths = {'trot': 'nn/trot_gunoo.pt', 'pronk': 'nn/pronk_gunoo_2.pt'}
        model_path = model_paths.get(self.gait, policy_cfg.get('model_path', default_paths.get(self.gait)))

        self.actor_network = load_actor_network(config, model_path=model_path).to('cpu')
        self.joystick = joystick
        self.last_action = np.zeros(12)

        # Timing parameters
        self.timestep = 0.0
        # period can be configured per gait in config, otherwise use sensible defaults
        periods = policy_cfg.get('periods', {})
        default_periods = {'trot': 0.5, 'pronk': 0.4}
        self.period = periods.get(self.gait, default_periods.get(self.gait, 0.5))
        self.phase_FL = np.zeros(1)
        self.phase_FR = np.zeros(1)
        self.phase_HL = np.zeros(1)
        self.phase_HR = np.zeros(1)


    def compute_observation(self, state: LowState_):
        commands = self.joystick.get_commands() if self.joystick else np.zeros(3)
        body_quat = np.array([
            state.imu_state.quaternion[1],
            state.imu_state.quaternion[2],
            state.imu_state.quaternion[3],
            state.imu_state.quaternion[0]
        ])
        body_vel = np.array(state.imu_state.gyroscope[:3])
        joint_angles = np.array([m.q for m in state.motor_state[:12]])
        joint_velocities = np.array([m.dq for m in state.motor_state[:12]])

        gravity_body = quat_rotate_inverse(
            torch.tensor(body_quat, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
        ).squeeze().numpy()

        # phase assignment depending on selected gait
        if self.gait == 'pronk':
            # All legs in phase
            self.phase_FL[0] = (self.timestep) % self.period / self.period
            self.phase_FR[0] = self.phase_FL[0]
            self.phase_HL[0] = self.phase_FL[0]
            self.phase_HR[0] = self.phase_FL[0]
        else:
            # default to trot: diagonal legs out of phase by 0.5
            self.phase_FL[0] = (self.timestep) % self.period / self.period
            self.phase_FR[0] = (self.phase_FL[0] + 0.5) % 1
            self.phase_HL[0] = (self.phase_FL[0] + 0.5) % 1
            self.phase_HR[0] = self.phase_FL[0]

        self.timestep += CONTROL_INTERVAL

        obs = np.concatenate((
            body_vel,
            gravity_body,
            commands,
            joint_angles - self.default_joint_angles,
            joint_velocities,
            self.last_action,
            np.sin(2*np.pi*self.phase_FL),
            np.sin(2*np.pi*self.phase_FR),
            np.sin(2*np.pi*self.phase_HL),
            np.sin(2*np.pi*self.phase_HR)
        ))

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        return obs

    def infer_action(self, state: LowState_):
        obs = self.compute_observation(state)
        with torch.no_grad():
            actions = self.actor_network(obs).numpy()[0]
        self.last_action = actions.copy()
        return actions


# ---------------- Helper Functions ----------------
def check_safety_stops(state: LowState_, joystick: JoystickHandler) -> bool:
    body_quat = state.imu_state.quaternion
    inclination = 2 * np.arcsin(np.sqrt(body_quat[1]**2 + body_quat[2]**2))
    return inclination > np.pi / 8 or joystick.safety_button_pressed()


def joint_linear_interpolation(init_pos, target_pos, rate):
    rate = np.clip(rate, 0.0, 1.0)
    return init_pos * (1 - rate) + target_pos * rate




# ---------------- Robot Controller ----------------
class RobotController:
    def __init__(self, config_path='config.yaml'):
        self.Kp = DEFAULT_KP
        self.Kd = DEFAULT_KD
        self.low_state: LowState_ = None
        self.crc = CRC()
        self.lowCmdWriteThreadPtr = None

        # Joystick
        self.joystick = JoystickHandler()

        # Policy
        self.policy_module = Policy(config_path, self.joystick)

        # Standing-up sequence
        self.start_pos = np.zeros(12)
        self.target_pos_1 = np.array([0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                                      -0.2, 1.36, -2.65, 0.2, 1.36, -2.65])
        self.target_pos_2 = np.array([0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                                      0.0, 0.67, -1.3, 0.0, 0.67, -1.3])
        self.target_pos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]
        self.percent_1 = 0.0
        self.percent_2 = 0.0
        self.percent_3 = 0.0
        self.percent_4 = 0.0
        self.duration_1 = 50
        self.duration_2 = 50
        self.duration_3 = 100
        self.duration_4 = 150
        self.first_run = True

    # ---------------- Initialization ----------------
    def init(self):
        # Initialize LowCmd
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_cmd.head = [0xFE, 0xEF]
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for m in self.low_cmd.motor_cmd:
            m.mode = 0x01
            m.q = go2w.PosStopF
            m.dq = go2w.VelStopF
            m.kp = self.Kp
            m.kd = self.Kd
            m.tau = 0

        # Publisher & subscriber
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._lowstate_callback, 10)

        # Sport client & motion switcher
        self.sc = SportClient(); self.sc.Init()
        self.msc = MotionSwitcherClient(); self.msc.Init()

        self.stand_up = True
        self.sit_down = False

    # ---------------- Start control loop ----------------
    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            name="write_policy_cmd",
            interval=CONTROL_INTERVAL,
            target=self._lowcmd_write
        )
        self.lowCmdWriteThreadPtr.Start()

    # ---------------- State callback ----------------
    def _lowstate_callback(self, msg: LowState_):
        self.low_state = msg

    # ---------------- Command loop ----------------
    def _lowcmd_write(self):
        if self.low_state is None:
            return

        if self.stand_up == True:
            self.start_pos = np.array([m.q for m in self.low_state.motor_state[:12]])
            self.stand_up = False
            self.percent_1 = 0.0
            self.percent_2 = 0.0
            self.percent_3 = 0.0
            self.percent_4 = 0.0

        if check_safety_stops(self.low_state, self.joystick):
            print("Safety condition triggered. Stopping robot.")
            os._exit(1)

        if self.joystick.sit_down_button_pressed() and self.percent_3 == 1.0:
            self.sit_down = True

        if self.joystick.stand_up_button_pressed() and self.percent_4 == 1.0:
            self.sit_down = False
            self.stand_up = True

        # Standing up sequence
        self._stand_up_sequence()

        # Policy control after standing up
        if self.percent_1 == 1 and self.percent_2 == 1 and self.percent_3 == 1 and self.sit_down == False and self.stand_up == False:
            actions = self.policy_module.infer_action(self.low_state)
            des_pos = self.policy_module.default_joint_angles + 0.25 * actions
            for i in range(12):
                m = self.low_cmd.motor_cmd[i]
                m.q = des_pos[i]
                m.dq = 0
                m.kp = self.Kp
                m.kd = self.Kd
                m.tau = 0

        self._sit_down_sequence()

        # Send command
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    # ---------------- Stand up sequence ----------------
    def _stand_up_sequence(self):
        print("Standing up sequence in progress...")
        if self.percent_1 < 1:
            self.percent_1 += 1.0 / self.duration_1
            self.percent_1 = min(self.percent_1, 1)
            q_cmd = joint_linear_interpolation(self.start_pos, self.target_pos_1, self.percent_1)
        elif self.percent_2 < 1:
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)
            q_cmd = joint_linear_interpolation(self.target_pos_1, self.target_pos_2, self.percent_2)
        elif self.percent_3 < 1:
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            q_cmd = self.target_pos_2
        else:
            return

        for i in range(12):
            m = self.low_cmd.motor_cmd[i]
            m.q = q_cmd[i]
            m.dq = 0
            m.kp = 60.0
            m.kd = 5.0
            m.tau = 0

    # ---------------- Stand up sequence ----------------
    def _sit_down_sequence(self):
        if self.sit_down == True:
            self.percent_4 += 1.0 / self.duration_4
            self.percent_4 = min(self.percent_4, 1)
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_4) * self.target_pos_2[i] + self.percent_4 * self.target_pos_3[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 60.0
                self.low_cmd.motor_cmd[i].kd = 5.0
                self.low_cmd.motor_cmd[i].tau = 0


# ---------------- Main ----------------
if __name__ == '__main__':
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    controller = RobotController()
    controller.init()
    controller.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        sys.exit(0)
