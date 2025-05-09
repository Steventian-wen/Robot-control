import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体设置
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def inverse_kinematics(model, data, target_pos, max_iter=100, tol=1e-4):
    """逆运动学求解器"""
    ee_id = model.body("right_inner_knuckle").id
    q = data.qpos.copy()
    
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        current_pos = data.body(ee_id).xpos
        err = target_pos - current_pos
        
        J = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, J, None, ee_id)
        J = J[:, :7]
        
        # 阻尼最小二乘解
        damping = 0.001
        delta_q = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
        q[:7] += delta_q
        
        # 关节限位保护
        q[:7] = np.clip(q[:7], model.jnt_range[:7,0], model.jnt_range[:7,1])
        data.qpos[:7] = q[:7]
        
        if np.linalg.norm(err) < tol:
            break
    return q[:7]

def circular_trajectory(t, radius=0.2, freq=0.5):
    """圆周轨迹生成器"""
    theta = 2 * np.pi * freq * t
    return np.array([
        radius * np.cos(theta) +0.3,
        radius * np.sin(theta)-0.2,
        0.6
    ])

# 仿真参数配置
SIM_DURATION = 10.0      # 总仿真时间（秒）
CTRL_FREQ = 500         # 控制频率（Hz）
TIMESTEP = 0.002        # 物理仿真步长

# 初始化模型
model = mujoco.MjModel.from_xml_path('scene.xml')
model.opt.timestep = TIMESTEP
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

# 控制参数
Kp = np.array([1200, 1100, 900, 800, 600, 400, 300])
Kd = 0.1 * Kp

# 数据记录配置
num_steps = int(SIM_DURATION / TIMESTEP)
time_axis = np.arange(0, SIM_DURATION, TIMESTEP)[:num_steps]

# 预分配存储空间
target_positions = np.zeros((num_steps, 3))
actual_positions = np.zeros((num_steps, 3))
joint_angles = np.zeros((num_steps, 7))
control_signals = np.zeros((num_steps, 7))

# 开始仿真
step = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while step < num_steps and viewer.is_running():
        # 计算当前时间
        t = step * TIMESTEP
        
        # 生成目标轨迹
        target_pos = circular_trajectory(t)
        target_positions[step] = target_pos
        
        # 逆运动学求解
        q_target = inverse_kinematics(model, data, target_pos)
        
        # PD控制 + 重力补偿
        mujoco.mj_inverse(model, data)
        gravity_comp = data.qfrc_inverse[:7]
        tau = Kp * (q_target - data.qpos[:7]) - Kd * data.qvel[:7] + gravity_comp
        
        # 应用控制并记录数据
        data.ctrl[:7] = tau
        actual_positions[step] = data.body("right_inner_knuckle").xpos
        joint_angles[step] = data.qpos[:7]
        control_signals[step] = tau
        
        # 物理仿真步进
        mujoco.mj_step(model, data)
        viewer.sync()
        
        step += 1

# 结果可视化
plt.figure(figsize=(14, 10))

# 三维轨迹跟踪结果
ax1 = plt.subplot(2, 2, 1, projection='3d')
ax1.plot(target_positions[:,0], target_positions[:,1], target_positions[:,2], 
        label='目标轨迹', linestyle='--')
ax1.plot(actual_positions[:,0], actual_positions[:,1], actual_positions[:,2], 
        label='实际轨迹', alpha=0.7)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('末端执行器轨迹跟踪')
ax1.legend()

# 关节角度变化
ax2 = plt.subplot(2, 2, 2)
for j in range(7):
    ax2.plot(time_axis, np.degrees(joint_angles[:,j]), 
            label=f'关节 {j+1}')
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('关节角度 (°)')
ax2.set_title('关节运动状态')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 控制信号
ax3 = plt.subplot(2, 2, 3)
for j in range(7):
    ax3.plot(time_axis, control_signals[:,j], label=f'关节 {j+1}')
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('控制力矩 (N·m)')
ax3.set_title('控制信号输出')

# 跟踪误差分析
ax4 = plt.subplot(2, 2, 4)
position_error = np.linalg.norm(actual_positions - target_positions, axis=1)
ax4.plot(time_axis, position_error*1000, color='r')
ax4.set_xlabel('时间 (s)')
ax4.set_ylabel('跟踪误差 (mm)')

plt.tight_layout()
plt.show()