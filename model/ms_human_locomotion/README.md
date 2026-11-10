# MS-Human Locomotion：关节净力矩用简化人体

本目录提供原生 MuJoCo MJCF 人体模型，面向三维运动拟合、关节净力矩及足部接触力分析。
模型的质量、关节中心、转轴和惯性来自 MS-Human Locomotion；肌肉激活与肌肉力分配不在本版本范围内。
当前完成模型派生和力学验收；Xsens 人体运动拟合、受试者个体化及实验测力逆动力学尚待接入。

## 文件与来源

- `human.xml`：直接加载的模型，无外部网格依赖。
- `derivation.json`：合并分组、每段完整惯性、关节映射、简化假设及原模型参考姿态。
- `inspection.json`：当前环境的编译维度、惯性检查和站立姿态结果。
- `preview.png`：静态侧面、正面预览；不是动力学稳定站立的证据。
- `source/`：固定提交的未修改上游子集，用于离线重建。
- `source/manifest.json`：来源、提交和每个上游文件的 SHA-256。
- `source/LICENSE`：上游 Apache-2.0 许可证，适用于包含的上游文件及派生模型。

上游：LNSGroup/MS-Human-700，提交 `2d686957aefd5739cf4d2859a6acfd94c8f84400`，
入口 `MS-Human-700-Locomotion.xml`，取回日期 2026-11-10。
派生代码：`tools/build_ms_human.py`。本目录的模型、可视形状和合并结构是对上游的修改。
`source/` 是派生所需子集，不是可以单独运行的完整肌肉仿真副本。

上游部分肋骨通过网格推断惯性，没有显式 `inertial`。因此保留了 24 个肋骨 OBJ 和资产声明，
重建时由 MuJoCo 推断这些惯性。其他身体均使用上游显式惯性，骨骼显示网格不参与本派生。
重建需要 Python、NumPy 和 MuJoCo；此次在 MuJoCo 3.9.0 验证。渲染额外使用 Pillow 和可用的 GL 后端。

## 模型结构与输出

世界坐标：x 向前、y 向左、z 向上；单位 m、rad、kg、N、Nm；重力 `[0, 0, -9.81]`。

| 部分 | 活动刚体数 | 速度自由度 |
|---|---:|---:|
| 骨盆 | 1 | 6（自由基座） |
| 左右大腿 | 2 | 6（每髋三个） |
| 左右小腿 | 2 | 2（每膝一个） |
| 左右距骨 | 2 | 2（每侧踝 pitch 一个） |
| 左右足 | 2 | 2（每侧 subtalar / roll 一个） |
| 上身整体 | 1 | 3（躯干相对骨盆） |
| 合计 | 10 | 21 |

`nq=22`，因为浮动基座姿态使用四元数；`nv=21`，`nu=15`，无肌肉激活状态、肌腱和等式约束。
保留距骨与足的两个刚体，是为了保留上游踝、距下关节不同的转轴和位置。

左右下肢关节使用以下名称，其中 `{side}` 为 `right` 或 `left`：

| 本项目名称 | 上游坐标 | 正方向与解释 |
|---|---|---|
| `{side}_hip_pitch` | `hip_flexion_r/l` | 正方向为上游髋屈曲方向 |
| `{side}_hip_roll` | `hip_adduction_r/l` | 正方向为上游髋内收方向，左右局部轴不同 |
| `{side}_hip_yaw` | `hip_rotation_r/l` | 保留上游旋转方向，左右局部轴不同 |
| `{side}_knee_pitch` | `knee_angle_r/l` | 正方向为膝屈曲 |
| `{side}_ankle_pitch` | `ankle_angle_r/l` | 保留上游斜向解剖转轴 |
| `{side}_ankle_roll` | `subtalar_angle_r/l` | 保留距下关节转轴，不是人为正交的 roll 轴 |

髋关节旋转顺序为 pitch、roll、yaw；躯干新增 `torso_pitch/roll/yaw` 三个坐标，
以 lumbar5 原点为枢轴、轴方向取自骨盆局部坐标并表达在躯干参考帧中。
躯干角度范围和人体力量上限尚未标定，所以躯干不设范围，所有力矩输入不设上限。
下肢保留原模型坐标范围；它们是模型范围，不等同于受试者临床活动度。

关节力矩是相应坐标的广义力，正功率定义为 `tau * qvel > 0`；不是世界坐标中的
三个独立力矩分量。尤其 hip、ankle 的名称是方向别名，不意味着其转轴两两正交。
精确的轴、锚点、范围、原名称见 `derivation.json`，运行时地址见 `inspection.json`。

**不要假定 actuator 顺序就是 qpos 顺序。** 当前构建器先加入下肢驱动再加入躯干驱动，
而 MJCF 的树遍历确定 qpos 顺序。应通过名称、`jnt_qposadr`、`jnt_dofadr` 和
`actuator_trnid` 查找地址。

## 简化与惯性保留

- 骨盆、骶骨和 `Abdomen` 合并为骨盆。
- lumbar5 及以上的脊柱、头、肋骨和双臂合并为上身；新增三个躯干自由度。
  双臂固定在上游姿态，不能描述独立摆臂。
- 髌骨固定到大腿，脚趾固定到足；固定参考角为零膝屈曲时的原约束值和零脚趾角。
- 膝关节原有平移和附加旋转冻结在原多项式于零屈曲时的值，仅保留主屈曲轴。
  因此这是固定轴膝，不能复现原模型随屈曲移动的膝轴和髌骨路径。
- 身体合并使用质心加权、旋转惯性张量及平行轴定理。在参考姿态保留总质量、全身质心
  和全身惯性；在辅助关节冻结的条件下保持下肢运动链的动力学。
- 清除原模型用于仿真调节的 armature、阻尼、刚度及关节摩擦；没有隐藏的被动关节力。
- 每个非基座关节提供 `gear=1` 的直接力矩 motor，仅用作净力矩接口；不是生理力量模型。

参考姿态中，保留角全为零，膝辅助坐标取约束的常数项，并将近似足底平移到世界地面 z=0。
这是本项目的 `standing` 关键帧，不是上游的 `init` 关键帧。

总质量为约 **60.216523 kg**，完整保留上游。现有实验文件记载的 85 kg 受试者尚未个体化，
目前不可把结果当成该受试者的力矩。后续需要根据实验身体尺寸、质量及惯性估计调整模型，
不能通过沿用原机器人比例缩小整条人体轨迹来代替。

## 接触与动力学的边界

`left_foot/right_foot` 是足底中心 site，参考站姿坐标轴与世界重合，运动后随足旋转。
`left_heel/right_heel/left_toe/right_toe` 是几何参考点。
足底盒半长 0.13 m、半宽 0.045 m、半厚 0.01 m；摩擦系数 0.8。
它们是当前显式近似参数，没有从力台或受试者标定。仅足底与地面碰撞，无身体自碰撞。
本模型可以用于后续接触仿真，但当前尚未验证稳定运动控制或接触力预测。

后续给定运动与外力的关节净力矩计算采用：

```text
required = M(q) @ qacc + qfrc_bias - qfrc_passive - sum(J_foot.T @ wrench_world)
```

这里 `wrench_world = [Fx, Fy, Fz, Mx, My, Mz]`，力矩关于对应足底 site，所有分量表达在世界系。
`J_foot` 的行顺序应为平移三行、旋转三行。该模型中 `qfrc_passive=0`。
取各关节 `jnt_dofadr` 对应项得到广义净力矩，浮动基座六项应作为残差报告，不能当作驱动力矩。
此表达假定外接触已通过外力给定，不再重复叠加同一接触的求解器约束力。

现有力台 CSV 缺少自由力矩分量，且同步、坐标变换和足归属尚待核对；该数据不能被直接当作
已经完整测得的六维足部 wrench。双支撑外力未知时，仍需要附加假设或测量。

## 重建、检查和查看

在项目根目录，使用已有 `humanoid_sim` 环境：

```bash
conda activate humanoid_sim
python tools/build_ms_human.py
python tools/inspect_human_model.py --output model/ms_human_locomotion/inspection.json
MUJOCO_GL=egl python tools/inspect_human_model.py --preview model/ms_human_locomotion/preview.png
python tools/inspect_human_model.py --viewer
python -m unittest discover -s tests -p 'test_human*.py' -v
```

`--viewer` 只查看静态姿态，不执行动力学步进。GUI 需要桌面显示；无头渲染可用 EGL，
如本机不支持 EGL 可选择已安装的其他 MuJoCo GL 后端。直接查看 `preview.png` 不需要依赖。

验收覆盖：上游校验和与离线重建、质量/全身质心/惯性保留、辅助关节冻结后的多姿态正运动学、
下肢质量矩阵与重力项、各段惯性正定与三角不等式、多姿态质量矩阵正定、
重力与势能梯度、直接力矩驱动及无肌肉/被动力。它们检验派生实现的一致性，不代表人体实验准确度。

## 后续接入

下一步添加面向受试者尺寸、目标身体段和本模型坐标的 Xsens 拟合，输出 `(N, 22)` qpos；
之后在正确处理四元数导数、滤波和实验外力对齐的基础上计算净力矩。
原始记录和预览方法见 `data/README.md`，开发顺序以根目录 `README.md` 为准。
