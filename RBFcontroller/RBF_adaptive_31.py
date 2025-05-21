import sys
sys.path.append(r'c:\xarm7')
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from IK.ik_solver import InverseKinematicsSolver

class TrajectoryGenerator:
    def __init__(self, dt=0.002):
        self.radius = 0.2
        self.freq = 0.5
        self.center = np.array([0.3, -0.2, 0.9])
        self.z_amp = 0.05
        self.t = 0.0
        self.dt = dt
        
    def get_trajectory(self):
        self.t += self.dt
        theta = 2*np.pi*self.freq*self.t
        
        pos = self.center + [
            self.radius*np.cos(theta),
            self.radius*np.sin(theta),
            self.z_amp*np.sin(4*np.pi*self.freq*self.t)
        ]
        
        vel = [
            -2*np.pi*self.freq*self.radius*np.sin(theta),
            2*np.pi*self.freq*self.radius*np.cos(theta),
            4*np.pi*self.freq*self.z_amp*np.cos(4*np.pi*self.freq*self.t)
        ]
        
        acc = [
            -(2*np.pi*self.freq)**2*self.radius*np.cos(theta),
            -(2*np.pi*self.freq)**2*self.radius*np.sin(theta),
            -(4*np.pi*self.freq)**2*self.z_amp*np.sin(4*np.pi*self.freq*self.t)
        ]
        
        return np.array(pos), np.array(vel), np.array(acc)

class SMCController:
    def __init__(self, n_joints=7, dt=0.002):
        self.n = n_joints
        self.c = 3.0*np.ones(n_joints)  # 增大滑模面系数加快收敛速度
        self.eta = 15.0*np.ones(n_joints)  # 增大切换增益抑制抖振
        self.gamma = 0.02
        self.phi = 0.05
        self.dt = dt
        
        # RBF参数
        self.n_centers = 25  # 增加RBF中心数量提升函数逼近能力
        self.centers = np.stack([
            np.column_stack([
                np.linspace(-np.pi, np.pi, self.n_centers),
                np.linspace(-5, 5, self.n_centers)
            ]) for _ in range(n_joints)
        ])
        self.widths = 0.5*np.ones(self.n_centers)
        self.W = np.zeros((n_joints, self.n_centers))

    def rbf(self, x, j):
        return np.exp(-np.linalg.norm(x - self.centers[j], axis=1)**2 / (2 * self.widths**2))
    
    def control_law(self, q, dq, q_d, dq_d, ddq_d):
        if not isinstance(dq_d, np.ndarray):
            dq_d = np.array(dq_d)
        if not isinstance(ddq_d, np.ndarray):
            ddq_d = np.array(ddq_d)
        assert dq_d.shape == (7,) and ddq_d.shape == (7,)  # 添加维度校验断言
        e = q - q_d
        edot = dq - dq_d
        s = self.c*e + edot
        
        u = np.zeros(self.n)
        f_hat_values = np.zeros(self.n)
        for j in range(self.n):
            h = self.rbf(np.array([q[j], dq[j]]), j)
            f_hat = self.W[j] @ h
            f_hat_values[j] = f_hat
            
            # 带边界层的符号函数
            sat = np.sign(s[j])
            
            u[j] = -self.c[j]*edot[j] - f_hat + ddq_d[j] - self.eta[j]*sat
            self.W[j] += self.gamma * s[j] * h * self.dt
            
        return np.clip(u, -50, 50), f_hat_values, s

# 初始化系统
model = mujoco.MjModel.from_xml_path('scene.xml')
if model is None:
    raise ValueError('模型文件加载失败，请检查scene.xml路径是否正确')
print(f"模型加载成功，物体数量: {model.nbody}，关节数量: {model.njnt}")  # 添加调试信息
data = mujoco.MjData(model)
if data is None:
    raise RuntimeError('数据初始化失败')
print("数据初始化成功")  # 添加调试信息
controller = SMCController(dt=model.opt.timestep)
traj_gen = TrajectoryGenerator(model.opt.timestep)
ik_solver = InverseKinematicsSolver(model)

# 初始位置设置
data.qpos[:7] = ik_solver.solve(data, traj_gen.center)[0]
data.qpos[7:] = 0
mujoco.mj_forward(model, data)

# 数据记录
log = {
    'time': [], 'target': [], 'actual': [],
    'error': [], 'control': [], 'joints': [],
    's': [], 'f_hat': [], 'tau_saturation': []
}

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 调试：打印模型和数据状态
    print(f"模型加载状态：{model is not None}")
    print(f"数据初始化状态：{data is not None}")
    # 精确计时器配置
    SIM_DURATION = 10.0  # 精确5秒
    num_steps = int(SIM_DURATION / model.opt.timestep)
    time_compensation = 0.0
    step = 0

    # 严格5秒终止条件（允许1个时间步误差）
    while step < num_steps and viewer.is_running():
        # 轨迹生成
        target_pos, target_vel, target_acc = traj_gen.get_trajectory()
        
        # 逆运动学
        q_d = ik_solver.solve(data, target_pos)[0]
        
        # 使用轨迹生成器提供的真实导数
        jac_pos = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, None, model.site(ik_solver.ee_site_name).id)
        J = jac_pos[:, :7]
        dq_d = np.linalg.pinv(J) @ target_vel.reshape(3,1)  # 将 target_vel 转换为列向量
        dq_d = dq_d.flatten()
        assert isinstance(dq_d, np.ndarray)
        ddq_d = np.linalg.pinv(J) @ (target_acc - J @ dq_d)
        ddq_d = ddq_d.flatten()
        assert isinstance(ddq_d, np.ndarray)
        
        # 控制计算
        u, f_hat_values, s_values = controller.control_law(data.qpos[:7], data.qvel[:7], q_d, dq_d.copy(), ddq_d.copy())
        data.ctrl[:7] = u
        
        # 记录数据
        log['time'].append(data.time)
        log['target'].append(target_pos.copy())
        log['actual'].append(data.site(model.site(ik_solver.ee_site_name).id).xpos.copy())
        log['error'].append(np.linalg.norm(log['target'][-1]-log['actual'][-1]))
        log['control'].append(u.copy())
        log['joints'].append(data.qpos[:7].copy())
        log['s'].append(s_values)
        log['f_hat'].append(f_hat_values.copy())
        log['tau_saturation'].append(np.sum(np.abs(u) >= 49.9))
        
        # 仿真步进
        mujoco.mj_step(model, data)
        step += 1
        viewer.sync()

# 可视化分析
plt.figure(figsize=(16,12))

# 轨迹跟踪
ax1 = plt.subplot(3,2,1, projection='3d')
target = np.array(log['target'])
actual = np.array(log['actual'])
ax1.plot(target[:,0], target[:,1], target[:,2], 'r--', label='期望轨迹')
ax1.plot(actual[:,0], actual[:,1], actual[:,2], 'b-', label='实际轨迹')
ax1.set_title('末端执行器轨迹跟踪')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.legend()

# 跟踪误差
ax2 = plt.subplot(3,2,2)
ax2.plot(log['time'], np.array(log['error'])*1000, 'g-')
ax2.set_title('位置跟踪误差')
ax2.set_xlabel('时间(s)')
ax2.set_ylabel('误差 (m)')

# 控制输入
ax3 = plt.subplot(3,2,3)
ctrl = np.array(log['control'])
for j in range(7):
    ax3.plot(log['time'], ctrl[:,j], label=f'关节{j+1}')
ax3.set_title('控制力矩')
ax3.set_xlabel('时间(s)')
ax3.set_ylabel('控制力矩 (N·m)')
ax3.legend()

# 关节角度
ax4 = plt.subplot(3,2,4)
q = np.array(log['joints'])
t = np.array(log['time'])
for j in range(7):
    ax4.plot(t, q[:,j], label=f'关节{j+1}')
ax4.set_title('关节角度变化')
ax4.set_xlabel('时间(s)')
ax4.set_ylabel('关节角度 (rad)')
ax4.legend()

# 滑模面
ax5 = plt.subplot(3,2,5)
s = np.array(log['s'])
ax5.plot(t, np.linalg.norm(s, axis=1), 'k-')
ax5.set_title('滑模面s')
ax5.set_xlabel('时间(s)')
ax5.set_ylabel('滑模面范数 (无量纲)')

# RBF估计误差与真实误差
ax6 = plt.subplot(3,2,6)
f_hat = np.array(log['f_hat'])
ax6.plot(t, np.linalg.norm(f_hat, axis=1), 'r-', label='估计值')
ax6.set_title('RBF估计值')
ax6.set_xlabel('时间(s)')
ax6.set_ylabel('估计值')

plt.tight_layout()
plt.show()