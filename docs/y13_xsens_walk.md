# 新实验者 Xsens → 人体模型 → 力矩追踪

当前输入只使用 `data/y13_MVN_3_8_01.xlsx`，并保留同名 `.mvn` 原生工程。
新记录与旧实验数据来自不同实验者；旧 H5、`xsense_raw.xlsx`、`fp_raw.csv` 不参与本流程。
`.mvn` 未被逆向解析，不声称从中提取了原始 IMU 或标定参数。

新增的零骨盆辅助 RL 路线复用本页的模型和拟合，见 [mjlab 人体追踪基线](mjlab_human_tracking.md)。
本页下面的物理追踪说明仍对应原逆动力学诊断控制器。

## 已知数据和显式假设

- 用户提供新实验者体重 **62.5 kg**、身高 **1.72 m**。
- XLSX：52462 帧、240 Hz、Live 处理，首末帧约 218.59 秒。
- 使用 `Segment Orientation - Quat` 中的骨盆、双侧大腿、小腿和足部七个骨段四元数。
  它们是 Xsens 已完成融合和骨段标定的输出，不是直接把未标定的传感器四元数当作骨段姿态。
- 其余上身没有独立传感器约束；躯干三个关节暂固定在模型参考姿态，不能称为实测上身动作。
- Frame 0 暂作为模型零位标定参考。记录开头是否为标准站姿尚未确认，绝对关节角存在零位不确定性。
- 默认原地跑台模式；皮带速度暂按文件名假设为 3.8 km/h，即 1.05556 m/s，尚未得到实验确认。
- 足部几何、摩擦、接触软度、关节力矩上限均为仿真参数，不是受试者测量值。

以上参数与状态集中在 `config/y13_trial.json`，不要从旧实验复制身高、体重或外力。

## 模型适配

`tools/prepare_xsens_model.py` 从新 XLSX 的身体段原点估计大腿、小腿长度和髋宽，
调整模板运动链。当前估计双侧大腿约 0.4954 m、小腿约 0.4214 m、髋宽约 0.2001 m。
这是 Xsens 骨架原点间距，不是 IMU 安装点间距，也不是独立尺量的人体尺寸。

质量按用户体重缩放至 62.5 kg；各段质心和惯性按段长/质量因子缩放，
上身整体缩放到头部显示几何顶点高度 1.72 m。它是参数化模板近似，身体段质量分配和惯性张量
并非实验测得。模型原始模板保持不变，试验模型与假设写入 `outputs/y13/human.xml` 和 `model_scaling.json`。

## 运动拟合

`tools/fit_xsens_human.py` 按名称读取七个骨段，检查帧号和四元数有效性。
用旋转矩阵/SO(3) 残差求解每侧 6 个关节坐标，保留膝、踝的模型转轴和范围，
不直接复制 Xsens 的 Euler 关节角。骨盆姿态由四元数直接映射到浮动基座。

每个骨段的暂定参考变换为：

```text
R_target_i(t) = R_xsens_i(t) @ R_xsens_i(ref).T @ R_model_i(neutral)
```

这是一种参考姿态对齐，不是已经完成的解剖轴标定。其零位是否合适需要检查开头站姿或替换参考帧。
骨盆被指定跟随，所以骨盆方向残差为零是构造结果，不表示传感器绝对误差为零。

模型没有完全复现 Xsens 的膝耦合和足部自由度，因此会有非零拟合残差。
`orientation_residual_deg` 为每个骨段的 SO(3) 旋转角误差；`segment_position_residual_m`
是对齐参考姿态后的骨盆相对关节原点残差，用作诊断，不是经过外部动捕验证的误差。

原记录在约 117.03 秒、176.95 秒有共同姿态突跳和大幅位置突跳。
工具把“共同世界参考系改变”作为校正假设，标记每次跳变附近约 ±0.25 秒为无效；
它没有证明异常原因，也没有将这些帧修复成可信测量。任何速度/加速度计算和物理追踪都不跨无效段。

为避免模板刚性足底穿地，另记录骨盆高度调整，使每帧最低足底落在约 -1 mm。
这是行走接触假设，不是测得的骨盆轨迹。原始角度拟合保留在 `qpos`，
动力学追踪使用独立平滑的 `tracking_qpos`。原地模式不使用累积的 Xsens 水平根位移。

## 力矩与物理追踪

`tools/simulate_xsens_walk.py` 初始化一次人体状态，之后通过 `mj_step` 积分；
人体 qpos 不会在控制循环中被参考轨迹覆盖。环境的跑带滑动坐标单独维护，代表外部驱动的跑台。
当前仅建模足底与跑带接触；排除左右近似足底盒之间的碰撞，没有配置全身自碰撞。

默认 `--assistance pelvis` 是用于检查动作与动力学接口的**骨盆辅助追踪**：

1. 根据关节/骨盆姿态误差和速度误差生成期望加速度。
2. 由 MuJoCo 逆动力学（包括软接触）计算所需广义力。
3. 15 个关节力矩写入 motor；未驱动的 6 个基座分量作为显式骨盆外力/力矩施加并完整记录。
4. 读取真实仿真接触力，换算为足底中心的世界系六维 wrench。

辅助力不是人体内部关节力矩，不能隐藏。当前辅助力较大、力矩峰值也较大，
表明参考动作、接触模型和动力学可行性仍不匹配；该结果不能称为自主稳定行走或生理力矩验证。
力矩上限 1500 Nm 是防止数值失控的诊断上限，不是人体力量参数。

`--assistance none` 为实验性的浮动基座控制：局部调整期望加速度，使逆动力学的六个
未驱动基座力接近零，只输出关节力矩，不施加骨盆外力。当前同一行走片段的测试会失稳，
失败报告与轨迹单独保存。它还不是可靠的步行控制器。

`tools/compute_joint_torques.py` 根据记录的状态、加速度和仿真足部 wrench 重建净力矩：

```text
required = M @ qacc + bias - passive - sum(J_foot.T @ wrench_world) - other_constraint_force
```

`other_constraint_force` 单独记录关节限位等其余约束。重建的驱动坐标项应与 motor 输出一致，
基座项应与已记录的骨盆辅助一致。这是在验证程序内的动力学与力矩约定，
不是证明真实人体力矩准确。力台数据没有被用来填补或验证这个新实验。

本次实际运行的数值结果与失败边界见 [验证结果](y13_results.md)。

## 一键运行

在项目根目录、已有 `humanoid_sim` 环境中：

```bash
conda activate humanoid_sim
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/imu-toque-mpl python tools/run_xsens_pipeline.py
```

也可只生成模型和拟合：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/imu-toque-mpl python tools/run_xsens_pipeline.py --fit-only
```

已有模型和拟合时，只重跑控制与分析：

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/imu-toque-mpl python tools/run_xsens_pipeline.py --skip-fit
```

仅重新准备模型：

```bash
python tools/prepare_xsens_model.py --mass-kg 62.5 --height-m 1.72
python tools/fit_xsens_human.py --model outputs/y13/human.xml --output outputs/y13/fit.npz --no-align-heading
```

检查无辅助模式，保留独立失败报告（失败返回非零退出码）：

```bash
python tools/simulate_xsens_walk.py --assistance none --start 25 --duration 10 --output outputs/y13/walk_free.npz
```

## 结果与查看

所有生成结果在 Git 忽略的 `outputs/y13/`：

| 文件 | 内容 |
|---|---|
| `human.xml` / `model_scaling.json` | 试验人体模型和缩放假设 |
| `fit.npz` / `fit.json` | 60 Hz 的 `(13116, 22)` qpos、15 个关节角、姿态/位置残差、有效性与模型哈希 |
| `joint_angles.csv` / `.png` | 关节角数组/曲线；CSV 角度单位 rad |
| `fit_residuals.csv` / `.png` | 七个骨段方向残差（deg）与关节原点残差（m） |
| `walk_assisted.npz` / `.json` | 默认 25–35 秒片段的物理状态、控制力矩、足部接触、骨盆辅助、数值残差 |
| `simulation_torques.csv` | 15 个 motor 力矩（Nm）、骨盆辅助及双足仿真 wrench |
| `joint_torques.npz` / `.json` | 按动力学重建的力矩和与 motor 输出的差异 |
| `simulation_diagnostics.png` | 关节力矩、骨盆辅助、接触力与追踪误差 |
| `walk_free.npz` / `.json` | 无骨盆辅助的独立测试，检查 `completed` 与 `failure` |
| `walk_assisted.gif` | 已记录物理状态的动画，需用下面命令生成 |

运行带实时窗口的力矩驱动仿真（每步重新计算控制并执行物理积分；当前仍有骨盆辅助）：

```bash
python tools/simulate_xsens_walk.py --viewer
```

查看已记录的力矩驱动仿真（播放器只重放记录，不重新执行控制）：

```bash
python tools/view_human_motion.py --motion outputs/y13/walk_assisted.npz --duration 10
```

查看全程运动拟合，或无头导出 GIF：

```bash
python tools/view_human_motion.py --motion outputs/y13/fit.npz --start 25 --duration 10
MUJOCO_GL=egl python tools/view_human_motion.py --output outputs/y13/walk_assisted.gif
```

## 下一阶段验收

首先核对参考站姿、跑台速度和 Xsens 工程身体尺寸；完善足底接触几何及分段姿态拟合。
随后加入有约束的接触/平衡规划，在可观测范围内调整参考加速度与质心运动，逐步减少骨盆辅助，
要求在零辅助下跨越多次左右支撑转换，并检查摩擦、CoP、关节力矩与残差。
再考虑用同一新实验者的同步测力数据验证人体力矩与接触力；旧实验外力不能替代。
