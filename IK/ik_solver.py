import mujoco
import numpy as np

class InverseKinematicsSolver:
    """逆运动学求解器封装类
    
    属性:
        damping (float): 阻尼最小二乘系数
        max_iter (int): 最大迭代次数
    """
    
    def __init__(self, model, damping=0.001, max_iter=100):
        self.model = model
        self.damping = damping
        self.max_iter = max_iter
        self.ee_site_name = "link_tcp"

    def solve(self, data, target_pos, target_vel=None, dt=0.001):
        """带速度解的改进逆运动学
        
        参数:
            data (mujoco.MjData): 仿真数据结构
            target_pos (np.ndarray): 目标位置(3维)
            target_vel (np.ndarray): 目标速度(3维)
            dt (float): 时间步长
        
        返回:
            tuple: (关节角度(7维), 关节速度(7维))
        """
        q = data.qpos[:7].copy()
        site_id = self.model.site(self.ee_site_name).id
        
        for _ in range(self.max_iter):
            mujoco.mj_forward(self.model, data)
            current_pos = data.site(site_id).xpos
            err = target_pos - current_pos
            
            # 计算雅可比矩阵
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
            J = jacp[:, :7]
            
            # 阻尼伪逆解
            Jinv = np.linalg.pinv(J.T @ J + self.damping**2 * np.eye(7)) @ J.T
            delta_q = Jinv @ err
            q += delta_q * dt
            
            # 关节限位保护
            q = np.clip(q, self.model.jnt_range[:7,0], self.model.jnt_range[:7,1])
            data.qpos[:7] = q
            
            if np.linalg.norm(err) < 1e-4:
                break
        
        # 速度计算
        if target_vel is not None and J is not None:
            qd = Jinv @ target_vel
        else:
            qd = np.zeros(7)
        
        return q.copy(), qd.copy()

    @staticmethod
    def validate_joint_limits(q, model):
        """关节限位验证方法"""
        return np.clip(q, model.jnt_range[:7,0], model.jnt_range[:7,1])