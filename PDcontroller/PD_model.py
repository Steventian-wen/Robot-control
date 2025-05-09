import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib import rcParams

# 中文字体设置
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 系统参数配置
SIM_DURATION = 10.0        # 仿真总时长(s)
CTRL_FREQ = 500           # 控制频率(Hz)
LAMBDA = 15.0             # 滑模面系数
Kd = np.diag([500]*7)     # 滑模控制增益矩阵

def inverse_kinematics(model, data, target_pos, target_vel=None, dt=0.001, max_iter=100):
    """带速度解的改进逆运动学"""
    q = data.qpos[:7].copy()
    ee_id = model.body("right_inner_knuckle").id
    
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        current_pos = data.body(ee_id).xpos
        err = target_pos - current_pos
        
        # 计算雅可比矩阵
        J = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, J, None, ee_id)
        J = J[:, :7]
        
        # 阻尼伪逆解
        damping = 0.001
        Jinv = np.linalg.pinv(J.T @ J + damping**2 * np.eye(7)) @ J.T
        delta_q = Jinv @ err
        q += delta_q * dt
        
        # 关节限位保护
        q = np.clip(q, model.jnt_range[:7,0], model.jnt_range[:7,1])
        data.qpos[:7] = q
        
        if np.linalg.norm(err) < 1e-4:
            break
    
    # 速度计算
    if target_vel is not None and J is not None:
        qd = Jinv @ target_vel
    else:
        qd = np.zeros(7)
    
    return q.copy(), qd.copy()

def sliding_mode_controller(model, data, qd, qd_dot, qd_ddot):
    """改进的滑模控制器"""
    nv = model.nv
    
    # 当前状态
    q = data.qpos[:7].copy()
    q_dot = data.qvel[:7].copy()
    
    # 误差计算
    q_tilde = q - qd
    q_tilde_dot = q_dot - qd_dot
    
    # 参考轨迹
    qr_dot = qd_dot - LAMBDA * q_tilde
    qr_ddot = qd_ddot - LAMBDA * q_tilde_dot
    
    # 滑模面
    s = q_tilde_dot + LAMBDA * q_tilde
    
    # 获取动力学参数
    H = np.zeros((nv, nv), dtype=np.float64)
    mujoco.mj_fullM(model, H, data.qM)
    H_joints = H[:7, :7]
    
    # 计算科里奥利力
    C = np.zeros(nv)
    mujoco.mj_rnePostConstraint(model, data)
    mujoco.mj_rne(model, data, 1, C)
    
    # 计算重力项
    data.qacc[:] = 0
    data.qfrc_bias[:] = 0
    mujoco.mj_rne(model, data, 0, data.qfrc_bias)
    G = data.qfrc_bias[:7]
    
    # 控制律
    tau = H_joints @ qr_ddot + C[:7] + G - Kd @ s
    return np.clip(tau, -300, 300)

def circular_trajectory(t, radius=0.2, freq=0.5):
    """生成带速度加速度的轨迹"""
    omega = 2 * np.pi * freq
    theta = omega * t
    
    pos = np.array([
        radius * np.cos(theta) +0.3,
        radius * np.sin(theta)-0.2,
        0.6
    ])
    
    vel = np.array([
        -radius * omega * np.sin(theta),
        radius * omega * np.cos(theta),
        0
    ])
    
    acc = np.array([
        -radius * omega**2 * np.cos(theta),
        -radius * omega**2 * np.sin(theta),
        0
    ])
    
    return pos, vel, acc

# 初始化模型
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)
model.opt.timestep = 1/CTRL_FREQ
mujoco.mj_resetDataKeyframe(model, data, 0)

# 数据记录
num_steps = round(SIM_DURATION * CTRL_FREQ) 
time_axis = np.linspace(0, SIM_DURATION, num_steps)

log = {
    'target_pos': np.zeros((num_steps, 3)),
    'actual_pos': np.zeros((num_steps, 3)),
    'tau': np.zeros((num_steps, 7)),
    'q_error': np.zeros((num_steps, 7))
}

# 主控制循环
step = 0
prev_qd_dot = np.zeros(7)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 直接使用步数作为循环条件
    while step < num_steps and viewer.is_running():
        # 生成目标轨迹
        t = step / CTRL_FREQ
        target_pos, target_vel, target_acc = circular_trajectory(t)
        log['target_pos'][step] = target_pos
        
        # 逆运动学求解
        qd, qd_dot = inverse_kinematics(model, data, target_pos, target_vel)
        
        # 计算关节加速度
        if step == 0:
            qd_ddot = np.zeros(7)
        else:
            qd_ddot = (qd_dot - prev_qd_dot) * CTRL_FREQ
        prev_qd_dot = qd_dot.copy()
        
        # 计算控制力矩
        tau = sliding_mode_controller(model, data, qd, qd_dot, qd_ddot)
        
        # 应用控制
        data.ctrl[:7] = tau
        
        # 记录数据
        log['actual_pos'][step] = data.body("right_inner_knuckle").xpos
        log['tau'][step] = tau
        log['q_error'][step] = data.qpos[:7] - qd
        
        # 仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1

    # 自动关闭前同步最后一次状态
    viewer.sync()

# 可视化结果
plt.figure(figsize=(14, 10))

# 三维轨迹跟踪
ax1 = plt.subplot(2, 2, 1, projection='3d')
ax1.plot(log['target_pos'][:,0], log['target_pos'][:,1], log['target_pos'][:,2], 
        'r--', label='目标轨迹')
ax1.plot(log['actual_pos'][:,0], log['actual_pos'][:,1], log['actual_pos'][:,2],
        'b-', alpha=0.5, label='实际轨迹')
ax1.set_title('三维轨迹跟踪')
ax1.legend()

# 位置误差
ax2 = plt.subplot(2, 2, 2)
pos_error = np.linalg.norm(log['actual_pos'] - log['target_pos'], axis=1)
ax2.plot(time_axis, pos_error * 1000)
ax2.set_title('末端位置跟踪误差')
ax2.set_ylabel('误差 (mm)')

# 控制力矩
ax3 = plt.subplot(2, 2, 3)
for i in range(7):
    ax3.plot(time_axis, log['tau'][:,i], label=f'关节{i+1}')
ax3.set_title('关节控制力矩')
ax3.set_ylabel('力矩 (N·m)')
ax3.legend()

# 关节角度误差
ax4 = plt.subplot(2, 2, 4)
for i in range(7):
    ax4.plot(time_axis, np.degrees(log['q_error'][:,i]), label=f'关节{i+1}')
ax4.set_title('关节角度跟踪误差')
ax4.set_ylabel('误差 (°)')

# 添加图例（调整到最佳显示位置）
ax4.legend(
    loc='upper right',   # 定位在右上角
    ncol=2,              # 分2列显示
    fontsize=8,          # 缩小字体
    framealpha=0.5       # 半透明背景
)

plt.tight_layout()
plt.show()