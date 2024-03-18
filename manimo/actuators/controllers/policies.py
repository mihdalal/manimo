from typing import Dict

import torch
import torchcontrol as toco
from polymetis.utils.data_dir import get_full_path_to_urdf
from torchcontrol.transform import Transformation as T
from torchcontrol.transform import Rotation as R

class JointPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, desired_joint_pos, kq, kqd, **kwargs):
        """
        Args:black
            desired_joint_pos (int):    Number of steps policy should execute
            hz (double):                Frequency of controller
            kq, kqd (torch.Tensor):     PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.q_desired = torch.nn.Parameter(desired_joint_pos)

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, self.q_desired, torch.zeros_like(qd_current)
        )

        return {"joint_torques": output}


class CartesianPDPolicy(toco.PolicyModule):
    """
    Performs PD control around a desired cartesian position
    """

    def __init__(
        self, joint_pos_desired, use_feedforwad, kq, kqd, kx, kxd, **kwargs
    ):
        super().__init__(**kwargs)
        # Get urdf robot model from polymetis
        panda_urdf_path = get_full_path_to_urdf("franka_panda/panda_arm.urdf")
        panda_ee_link_name = "panda_link8"
        self.robot_model = toco.models.RobotModelPinocchio(
            panda_urdf_path, panda_ee_link_name
        )
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=True
        )

        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(
            kq, kqd, kx, kxd
        )

        self.use_feedforwad = use_feedforwad

        self.joint_pos_desired = torch.nn.Parameter(joint_pos_desired)
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
        )

        torque_out = torque_feedback

        if self.use_feedforwad:
            torque_feedforward = self.invdyn(
                joint_pos_current,
                joint_vel_current,
                torch.zeros_like(joint_pos_current),
            )  # coriolis

            torque_out += torque_feedforward

        return {"joint_torques": torque_out}

class OperationalSpaceLowFreq(toco.PolicyModule):
    def __init__(
        self,
        q_current: torch.Tensor,
        call_freq: float,
        interm_freq: float,
        Kp,
        Kd,
        ignore_gravity: bool = True,
    ):
        super().__init__()

        self.call_freq = call_freq
        self.interm_freq = interm_freq
        self.robot_freq = 1000
        
        panda_urdf_path = get_full_path_to_urdf("franka_panda/panda_arm.urdf")
        panda_ee_link_name = "panda_link8"
        self.robot_model = toco.models.RobotModelPinocchio(
            panda_urdf_path, panda_ee_link_name
        )
        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(q_current)

        self.interp_arr = torch.repeat_interleave(
            torch.linspace(0, 1, int(self.robot_freq / self.interm_freq) + 1)[1:],
            int(self.interm_freq / self.call_freq),
        )[:, None]

        # self.interp_arr = torch.linspace(0, 1, int(self.robot_freq / self.call_freq) + 1)[1:, None]
        self.i = 0
        self.N = self.interp_arr.shape[0]

        self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
        self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)
        self.new_target = torch.nn.Parameter(
            torch.tensor(True, dtype=torch.bool), requires_grad=False
        )

        self.ee_pos_trajectory = ee_pos_current + self.interp_arr * torch.zeros_like(
            ee_pos_current
        )
        self.ee_quat_trajectory = ee_quat_current + self.interp_arr * torch.zeros_like(
            ee_quat_current
        )

        # self.x_start = ee_pos_current
        # self.r_start = R.from_quat(ee_quat_current)
        # self.rv_delta = torch.zeros(3)
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.pose_pd = toco.modules.feedback.CartesianSpacePDFast(Kp, Kd)

        # self.p_traj, self.pd_traj, _ = _min_jerk_spaces(self.N, self.N * 1.0 / self.robot_freq)
        # Initialize step count
        self.i = 0

    def _update_target(self, pos, quat):
        # TODO: Compare if target is new. If not, update self.N and self.i
        # TODO: Plan trajectory
        # print(self.ee_pos_desired - pos)

        # self.x_start = pos

        # # Plan rotation
        # self.r_start = R.from_quat(quat)
        # r_goal = R.from_quat(self.ee_quat_desired)
        # r_delta = r_goal * self.r_start.inv()
        # self.rv_delta = r_delta.as_rotvec()

        self.ee_pos_trajectory = pos + self.interp_arr * (self.ee_pos_desired - pos)
        quat_start = R.from_quat(quat)
        quat_delta = (R.from_quat(self.ee_quat_desired) * quat_start.inv()).as_rotvec()
        for i in range(self.N):
            r = R.from_rotvec(quat_delta * self.interp_arr[i]) * quat_start
            self.ee_quat_trajectory[i, :] = r.as_quat()
        # self.ee_quat_trajectory = quat + self.interp_arr * (self.ee_quat_desired - quat)

        self.new_target.copy_(torch.tensor(False, dtype=torch.bool))
        self.i = 0

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        ee_pos_current, ee_quat_current = self.robot_model.forward_kinematics(q_current)
        if self.new_target:
            self._update_target(ee_pos_current, ee_quat_current)

        jacobian = self.robot_model.compute_jacobian(q_current)
        ee_twist_current = jacobian @ qd_current

        ee_pos_desired = self.ee_pos_trajectory[self.i, :]
        ee_quat_desired = self.ee_quat_trajectory[self.i, :]
        # ee_twist_desired = self.ee_twist_trajectory[self.i, :]

        # D = self.ee_pos_desired - self.x_start
        # ee_pos_desired = self.x_start + D * self.p_traj[self.i]
        # r = R.from_rotvec(self.rv_delta * self.p_traj[self.i]) * self.r_start
        # ee_quat_desired = r.as_quat()
        # ee_twist_desired = torch.cat([D, self.rv_delta], dim=-1) * self.pd_traj[self.i]
        # print(ee_twist_desired)
        # Control logic
        wrench_feedback = self.pose_pd(
            ee_pos_current,
            ee_quat_current,
            ee_twist_current,
            ee_pos_desired,
            ee_quat_desired,
            # ee_twist_desired
            torch.zeros_like(ee_twist_current),
        )
        M_ee = self.robot_model.compute_inertia_ee(q_current)
        torque_feedback = jacobian.T @ M_ee @ wrench_feedback
        # torque_feedforward = self.invdyn(
        #     q_current, qd_current, torch.zeros_like(q_current)
        # )  # coriolis

        torque_out = torque_feedback  # + torque_feedforward

        # Increment & termination
        if self.i < self.N - 1:
            self.i += 1

        return {"joint_torques": torque_out}