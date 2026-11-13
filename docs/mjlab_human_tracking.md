# mjlab 人体动作追踪基线

此基线使用新实验者的 62.5 kg、1.72 m MS-Human 派生模型和 Xsens 拟合，
通过 PPO 学习关节目标修正，目标是在**无骨盆辅助**条件下追踪步行动作。
现有 `simulate_xsens_walk.py` 的有辅助诊断流程继续保留，RL 使用独立工具和输出目录。

## 环境

本机已建立项目内 `.venv-mjlab`：复用 `unitree_rl_mjlab` 的 PyTorch/mjlab 依赖，
在隔离环境中安装 MuJoCo 3.5.0。原环境只补齐缺少的 `scipy==1.16.3`，MuJoCo 版本未改。
这样修正了原环境中 mjlab 要求 `MuJoCo>=3.5.0`、实际却为 MuJoCo 3.4.0 的依赖冲突。
可选依赖锁定在 `requirements-mjlab.txt`；接口针对 `mjlab==1.2.0`，
不是按上游 main API 猜测实现。MuJoCo 版本与原来的 `humanoid_sim` 不同，请分开使用。

```bash
source .venv-mjlab/bin/activate
python tools/mjlab_human.py --help
```

`.venv-mjlab` 使用 system-site-packages 复用本机已有环境，并非完全独立的依赖副本。
如需在其他机器新建独立环境：

```bash
conda create -n human_mjlab python=3.11
conda activate human_mjlab
python -m pip install -r requirements-mjlab.txt
```

训练和评估需要 NVIDIA GPU。`check` 可以在没有 GPU 的环境中验证模型和任务接口。
所有日志、动作与模型均保存在本地，不需要 WandB 账号或上传采集数据。

## 准备参考动作

先按 [Xsens 流程](y13_xsens_walk.md) 生成 `outputs/y13/human.xml` 和 `fit.npz`，再运行：

```bash
python tools/mjlab_human.py prepare --start 25 --duration 10
python tools/mjlab_human.py check
```

默认从新实验第 25–35 秒生成 501 帧、50 Hz 的 `outputs/y13/mjlab/motion.npz`。
转换工具在 CPU 上做同一个人体模型的 FK，保留模型真实的 body/joint 顺序；
不调用硬编码 G1 模型的 `csv_to_npz.py`。根四元数和骨段四元数均为 **wxyz**。

| 数组 | 默认 shape | 含义 |
|---|---|---|
| `qpos` / `qvel` | `(501, 22)` / `(501, 21)` | 自由骨盆和 15 个关节的参考状态 |
| `joint_pos` / `joint_vel` | `(501, 15)` | 12 个下肢关节，随后为 3 个躯干关节 |
| `body_pos_w` / `body_quat_w` | `(501, 10, 3)` / `(501, 10, 4)` | 所有实体刚体，排除 world |
| `body_lin_vel_w` / `body_ang_vel_w` | `(501, 10, 3)` | 刚体原点的世界系速度，非质心线速度 |

转换会校验拟合/模型 SHA-256、关节顺序和有效帧，拒绝跨越重置/突跳区间。
`human.xml` 是独立训练实体：保留质量/惯性，移除原模型地面和执行器，由 mjlab 添加静止平面和 PD。
只启用双足对地接触；未配置全身自碰撞。未观测躯干增加明确的训练关节范围。

**坐标假设：** 原始拟合是原地跑台动作。此基线采用静止地面，给根位置加上
`x += speed * elapsed_time`，同时重新计算所有速度。这相当于匀速参考系变换，
前提是跑带向 -X 运动且速度恒定。默认 `--speed 1.05555556` 来自文件名中的 3.8 km/h，
尚未得到实验确认。这里没有把 Xsens 漂移的水平根位移当作测量真值。

## 控制和奖励

- 自由骨盆 6 维无执行器；策略只控制 15 个 hinge，完全不施加骨盆外力或外力矩。
- 策略和参考为 50 Hz；物理为 200 Hz，implicitfast，4 个物理子步。
- Actor 为 90 维：参考关节位置/速度、骨盆参考误差、基座速度、当前关节状态、上一动作。
  Critic 为 153 维，另含骨段相对位置和方向。这是有仿真完整状态的离线追踪任务，
  不是只凭七路实时 IMU 就能部署的策略。
- 动作是参考关节位置的修正，截断到 ±3，再乘腿部 0.35 rad、躯干 0.15 rad 的尺度。
  最终位置目标夹到训练关节范围内，由 MuJoCo 内置 PD 执行。
- 奖励包含七骨段方向/位置/速度、下肢关节追踪；惩罚动作变化、力矩、功率、关节越界、
  躯干偏离中立和足部滑动。滑动惩罚的接触判定目前使用足部高度近似。
- 躯干参考为未观测的中立先验，没有声称测得躯干运动。

| 关节组 | Kp（Nm/rad） | Kd（Nm·s/rad） | 力矩限幅（Nm） |
|---|---:|---:|---:|
| 髋 | 300 | 20 | 200 |
| 膝 | 350 | 20 | 200 |
| 踝 | 120 | 8 | 120 |
| 躯干 | 200 | 15 | 150 |

这些是基线数值控制参数，不是实验者的刚度、阻尼或力量测量值。
位置执行器使用目标位置和当前速度形成 PD 力矩；没有隐含的骨盆伺服。

训练采用随机参考帧初始化、适应性采样和失败后重置。片段结束通过 timeout 结束回合。
局部命令类显式禁止上游在片段末尾直接重新采样并写人体状态，避免隐藏的中途瞬移。

## 运行

先检查 GPU 步进和边界行为：

```bash
python tools/mjlab_human.py smoke --num-envs 32 --steps 100
```

启动 PPO，默认 1024 个并行环境、3000 次更新：

```bash
python tools/mjlab_human.py train --num-envs 1024 --iterations 3000
```

可以用 `--run-dir outputs/y13/mjlab/runs/my_baseline` 指定一个**尚不存在**的运行目录，
用 `--seed` 指定随机种子。每次运行保存独立参考快照、模型哈希、版本、超参数、TensorBoard 和 checkpoint。
输出目录默认在 Git 忽略的 `outputs/` 内。

```bash
tensorboard --logdir outputs/y13/mjlab/runs
```

训练按 Ctrl+C 会保存当前 checkpoint 并在 `run.json` 中标记中断。
恢复训练时指定原参考快照和 checkpoint；新结果保存到新目录：

```bash
python tools/mjlab_human.py train \
  --reference-dir outputs/y13/mjlab/runs/MY_RUN/reference \
  --checkpoint outputs/y13/mjlab/runs/MY_RUN/model_199.pt \
  --iterations 1000
```

## 连续评估和可视化

评估会**禁用所有自动重置**，从片段开头开始，失败立即停止并保存失败报告。
片段长度不足会报错，不会把循环播放或随机初始化后的短回合累计成连续成功。

```bash
# 零策略动作，纯参考 PD 对照；失败时返回退出码 2，也会保存轨迹。
python tools/mjlab_human.py evaluate --output outputs/y13/mjlab/reference_pd.npz

# 指定真实 checkpoint；自动读取其冻结的 reference 目录。
python tools/mjlab_human.py evaluate \
  --checkpoint outputs/y13/mjlab/runs/MY_RUN/model_199.pt \
  --output outputs/y13/mjlab/policy_evaluation.npz \
  --viewer
```

窗口运行的是实时策略控制的物理仿真；时间到或失败会关闭。`completed: true` 只代表该次
无重置地跑完请求时长，不单独代表追踪精度合格或人体力矩准确。

生成的 NPZ、JSON 和同名 XML 可以由现有播放器重放，也可在无头模式导出 GIF：

```bash
conda activate humanoid_sim
python tools/view_human_motion.py --motion outputs/y13/mjlab/policy_evaluation.npz
MUJOCO_GL=egl python tools/view_human_motion.py \
  --motion outputs/y13/mjlab/policy_evaluation.npz \
  --output outputs/y13/mjlab/policy_evaluation.gif
MPLBACKEND=Agg python tools/plot_mjlab_tracking.py \
  --trace outputs/y13/mjlab/policy_evaluation.npz
```

NPZ 输出真实 GPU `actuator_force`（Nm）、关节/骨段误差、动作、状态和双足六维 wrench。
**接触 wrench 当前由 CPU MuJoCo 在记录的 GPU 状态/控制下重新求解，用作诊断**；
它不是直接导出的 GPU 接触求解器结果，也不是实验测力。另记录 CPU/GPU 执行器力矩差异。
骨盆外加力在评估中逐步检查为零。
绘图工具另输出关节角/参考/力矩/足部 wrench 的 CSV、误差与受力诊断图、双侧关节角对照图。

## 验证边界

参考动作仍有暂定站姿、脚底几何和骨盆高度调整等假设。RL 为满足模型动力学而修正动作，
所以应同时比较追踪残差和稳定性；训练奖励上升不能代替无重置评估。
力矩与接触力取决于这些假设、控制器和奖励权重，当前没有新实验的测力真值。

离线转换和可选 mjlab CPU 测试分别位于 `tests/test_mjlab_reference.py`、`tests/test_mjlab_task.py`。
GPU 接口验收由 `smoke` 执行，检查观测/奖励有限、无外加力，以及片段结束不瞬移。
本次实际训练与评估数字另见 [基线结果](mjlab_human_results.md)。

上游参考：[mjlab](https://github.com/mujocolab/mjlab)、
[动作模仿数据格式](https://github.com/mujocolab/mjlab/blob/v1.2.0/docs/source/training/motion_imitation.rst)。
