# mjlab 人体追踪基线：本次验证结果

2026-11-11，本机 RTX 5080。输入为新实验者 62.5 kg、1.72 m 的 Xsens 拟合，
默认训练片段为原记录第 25–35 秒。没有使用旧 H5、旧 Xsens 或旧力台。

## 已实现和执行

- 生成 501 帧、50 Hz、带人体 FK 和模型哈希的参考动作。
- 人体 22 维 qpos、21 维 qvel、15 个关节执行器，骨盆无执行器。
- 跑通 PPO、本地 TensorBoard、参考快照、checkpoint 保存、续训和实时评估。
- 32 个 GPU 环境 × 100 步冒烟测试通过：观测/奖励有限、外加力为零、片段末尾不瞬移。
- 21 项不同单元测试通过，分两个环境执行：原环境 18 项，mjlab 环境 8 项（其中 5 项重叠）。
- `pip check` 在项目 `.venv-mjlab` 环境通过。
- 实际导出并检查完整回放模型、PNG、CSV 和 10 秒 GIF。

首先在已有环境完成 200 次 PPO 更新（1024 个并行环境）。依赖审计发现已有环境中的
MuJoCo 3.4 不符合 mjlab 1.2 的 MuJoCo >=3.5 要求，因此新增项目隔离环境，保留原环境版本。
在匹配的 MuJoCo 3.5.0 / MuJoCo Warp 3.5.0 环境中复测，并恢复 checkpoint 继续 5 次更新。
两次运行的真实版本分别记录在 `runs/baseline_ppo_002/run.json`、`baseline_ppo_003/run.json`，
不能把前一轮的 MuJoCo 版本记作 3.5。

最终 checkpoint 为：

```text
outputs/y13/mjlab/runs/baseline_ppo_003/model_203.pt
```

编号沿用 RSL-RL 恢复时的 iteration index，并不直接等于总更新次数。

## 无重置连续评估

| 条件 | 连续运行结果 |
|---|---|
| 纯参考 PD，零策略动作、零骨盆辅助 | 约 0.98 秒失稳 |
| 中间策略 `model_100.pt`，零骨盆辅助 | 约 2.74 秒失稳 |
| 200 次更新后 `model_199.pt`，零骨盆辅助 | 完成 10 秒 |
| 最终续训策略 `model_203.pt`，零骨盆辅助 | 完成 10 秒 |

最终评估在 MuJoCo 3.5.0 / mjlab 1.2.0 下实际运行，保存为 `outputs/y13/mjlab/policy_203.*`：

| 指标 | 结果 |
|---|---:|
| 外加骨盆力/力矩 | 全程为零 |
| 自动重置次数 | 0 |
| 连续时长 | 10.00 s |
| 15 个关节绝对误差 95 分位 | 12.69° |
| 七骨段方向误差 95 分位 | 17.85° |
| 根位置误差中位 / 95 分位 / 最大 | 0.464 / 0.903 / 0.975 m |
| 根向前位移（首末记录状态） | 11.505 m |
| 关节力矩绝对值 95 分位 / 峰值 | 64.79 / 139.25 Nm |
| CPU/GPU 执行器力矩最大差 | 2.93e-5 Nm 以下 |

根位置误差以假定的匀速前进参考为目标；明显存在速度/位移偏差，说明“完成 10 秒”并不等于
高精度动作重建。下一步应加强根速度/位移与足底接触追踪，并评估更长片段和扰动。
当前结果只是**同一个训练片段内的一次确定性评估**，未证明对新动作、其他速度或其他实验者的泛化。

关节力矩取自实际 GPU `actuator_force`；接触 wrench 是 CPU 在记录状态下重新求解的诊断值。
两者均未经过本次实验同步测力验证，不能当作已经识别出的真实人体力矩/外力。

## 查看本次策略

在项目根目录：

```bash
source .venv-mjlab/bin/activate
python tools/mjlab_human.py evaluate \
  --checkpoint outputs/y13/mjlab/runs/baseline_ppo_003/model_203.pt \
  --output outputs/y13/mjlab/policy_203.npz \
  --viewer
```

窗口运行 10 秒实时策略控制，完成后关闭。快速查看已经生成的动画：
[10 秒动作](../outputs/y13/mjlab/policy_203.gif)、
[受力与误差](../outputs/y13/mjlab/policy_203.png)、
[关节角对照](../outputs/y13/mjlab/policy_203_angles.png)。

继续训练（结果写入新的默认运行目录）：

```bash
source .venv-mjlab/bin/activate
python tools/mjlab_human.py train \
  --reference-dir outputs/y13/mjlab/runs/baseline_ppo_003/reference \
  --checkpoint outputs/y13/mjlab/runs/baseline_ppo_003/model_203.pt \
  --num-envs 1024 --iterations 1000
```

完整原理、安装与从头运行命令见 [基线说明](mjlab_human_tracking.md)。
