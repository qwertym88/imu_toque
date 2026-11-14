# Y13 时序接触状态基线

此工具仅使用 Y13 Xsens XLSX、相同 Y13 的拟合和匹配人体模型。旧样本 `fp_raw.csv` 与
旧样本 `xsense_raw.xlsx` 相互配对，**绝不作为 Y13 接触标签或外力输入**。配置中的
`force_plate.input` 固定为空；路径或拟合来源不匹配时工具拒绝运行。

## 当前实现

```bash
conda activate humanoid_sim
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/imu-toque-mpl \
  python tools/detect_xsens_contacts.py --plot-start 25 --plot-duration 10
```

`tools/contact_detection.py` 从 Y13 导出读取左右足和小腿的：

- 足部线加速度模长；
- 足部角速度和角加速度模长；
- 足部相对小腿的角速度；
- 足部相对骨盆的平移速度。

各特征进行 15 ms 平滑和 `log1p` 压缩，随后拟合两个对角 Gaussian 状态，使用 HMM 转移概率
加入时间连续性。低动态状态被命名为 `contact-like`，这是无监督状态的语义假设，不是已知标签。
默认状态持续时间先验为 0.45 秒；这是可调建模参数，不是当前受试者的测量周期。

HMM 在每个有效连续时间段独立推断，不跨 Xsens 导航重置和无效区间传播状态。
高频 240 Hz 状态随后采样到拟合/WBC 使用的 60 Hz 时间戳。

## 输出

结果写入 Git 忽略的 `outputs/y13/contact/`：

| 文件 | 内容 |
|---|---|
| `contact_inference.npz` | 240 Hz 特征/HMM 参数/posterior、离散候选状态、有效区间和 60 Hz 状态 |
| `contact_inference.json` | 样本来源、方法参数、候选占空比、事件时间与限制 |
| `contact_states.csv` | 可供后续 WBC 预览/导入的候选左右支撑状态、未校准 score、有效和不确定标记 |
| `contact_preview.png` | 足部加速度、角速度、HMM posterior、左右组合状态图 |

CSV 的二值列名包含 `candidate`；不能把它直接称为实验真值。`support_mask_candidate` 只在
posterior 不低于 0.8 时置真，低置信帧和重置区间由 `valid` / `uncertain` 标记。

离线 shadow QP 可用以下命令检查逆动力学方程和支撑约束：

```bash
python tools/human_wbc_shadow.py --start 25 --duration 10
```

它使用拟合姿态、速度和加速度，在 MuJoCo 人体模型上求解广义加速度、双足六维 wrench 和 15 个关节力矩，
并施加动力学平衡、候选支撑脚零加速度、摆动脚零 wrench、摩擦锥、CoP、扭转摩擦和诊断性力矩边界。
此工具不推进仿真、不施加力矩；输出只用于检查方程可行性与量级，不能视为测得的人体负载。
报告将候选状态与拟合姿态下的 MuJoCo 地面/脚底几何重叠逐帧比较。该重叠只是姿态几何代理，不能代替
力台或人工标注的真实接触标签；全帧统计包含没有高置信候选而跳过 QP 的帧。

## 在 MuJoCo 查看 shadow 结果

```bash
conda activate humanoid_sim
python tools/view_wbc_shadow.py
```

默认加载 `outputs/y13/{human.xml,fit.npz,wbc_shadow.npz}` 和
`outputs/y13/contact/contact_inference.npz`，播放已保存的 QP 时间窗口（当前为 25–35 秒）。
如 shadow 输出尚未生成，先执行 `python tools/human_wbc_shadow.py --start 25 --duration 10`。
这只回放拟合姿态，叠加离线 QP 预测结果；没有积分动力学，也没有将力矩施加给模型。

- 绿色脚底：当前帧的高置信支撑候选；灰色脚底：未被选为支撑。
- 绿色箭头：QP 预测的世界坐标接触力，默认 1000 N 对应 1 m。六维 wrench 中的力矩未画箭头。
- 橙色圆点：候选状态与模型地面/脚底几何重叠不一致；不是实验判错标签。
- 窗口文字：时间、QP 状态、左右脚候选/未校准 score/几何重叠、力模长、最大绝对值关节力矩。
  跳过或失败的求解不画力箭头，并显示力矩不可用。

默认循环播放；空格暂停/继续，左右方向键逐帧查看，`R` 从头播放，`Esc` 关闭。
`--once` 播放一次后停在末帧，窗口保持打开；`--speed 0.5` 半速播放。
`--start 27 --duration 2` 仅查看已保存结果中的 27–29 秒，不会重新计算 QP。
可用 `--force-scale 0.0005` 缩短箭头。

无桌面环境时检查所有帧的加载、姿态和标记构造：

```bash
python tools/view_wbc_shadow.py --check
```

该入口核对模型、Xsens 来源、时间戳和支撑候选；输入不匹配时拒绝回放。
旧的 `simulate_xsens_walk.py --viewer` 查看的是另一个有骨盆辅助的力矩追踪原型，不能用于
判断本次 shadow QP 是否能够独立行走。

## Shadow 力矩前向驱动试验

要直接检查已算出的 QP 关节力矩能否在 MuJoCo 接触环境中驱动人体，可运行：

```bash
python tools/simulate_shadow_torques.py --start 25.267 --duration 1 --viewer
```

这个试验从拟合姿态/速度初始化一次，之后只施加 shadow QP 的关节力矩，让 MuJoCo 推进人体和跑台接触；
没有姿态反馈、骨盆辅助或逐帧状态重置。QP 未求解的帧没有力矩，因此脚本对有效样点作线性插值，并在报告中
列出插值比例；这是试验近似。出现数值发散时会暂停并留在末态，方便观察失稳姿态。

结果写入 `outputs/y13/walk_shadow_torque.npz/.json`，可生成短 GIF：

```bash
MUJOCO_GL=egl python tools/view_human_motion.py \\
  --motion outputs/y13/walk_shadow_torque.npz --duration 1 --output outputs/y13/walk_shadow_torque.gif
```

本次从 25.267 秒的匹配支撑帧开始，开环模拟约 0.264 秒后右踝 pitch 速度达到 115 rad/s，触发安全停止；
关节误差 P95 已达约 38.5 度。窗口开头约 31.7% 的拟合帧需要插值力矩。此结果表明当前 QP 力矩序列在
当前模型/接触参数下不能直接形成稳定开环步态；由于没有反馈、接触参数未校准且部分命令经过插值，
它并不单独证明某个力矩估计必然错误，也不等价于完整 WBC 控制器测试。

## 当前证据和限制

本地只有一个 Y13 平地跑台样本，用户已说明坡道和楼梯记录正在整理。2026-11-13 的全记录运行
检测到左右脚低活动状态占比约 45.9%，每侧约 165–179 个状态转换，主要同侧接触开始间隔中位数
约 1.12 秒；有效输出帧约 99.4%。在 25–35 秒预览中，候选区间交替出现。

用同一 XLSX 的足部位置/速度做了单独一致性体检：候选接触样帧的足段原点高度中位数约 0.115 m，
候选摆动帧约 0.162–0.172 m；相对骨盆速度中位数约 0.87–0.88 m/s 和 1.18–1.21 m/s。
这与跑台上支撑脚随皮带相对骨盆运动、摆动脚高度较高的预期相容，但两种状态有重叠，
这些派生运动学量也不是外部接触真值。周期形态与运动学一致性**不能证明候选区间等于真实足底接触**；
接触状态没有标签，HMM posterior 未校准。

骨段原点不是足跟/前足。当前模型不知道真实 heel/toe 接触几何，也没有估计地面法向和楼梯台阶。
后续坡道和楼梯数据到位后，需要加入相应地形与同步视频/接触标注，按整段试验留出测试集，校准事件时刻、
状态转移和持续时间，并区分“物理接触”与“适合施加固定足端约束的稳定支撑”。

历史 WBC 也假设统一地面高度、水平足底和常高度 LIP 参考。即使接触时间序列判定正确，坡道/楼梯仍需
修改接触面法向、足底约束、不同脚高度的 ZMP/支撑多边形以及 COM 预瞄。当前接触估计器尚未接入完整 WBC，
也没有坡道/楼梯地形或独立接触标签。shadow QP 可行性和模型几何重叠度都不能验证候选状态的真实准确率，
更不能验证动态行走稳定性。

运行接触基线：

```bash
python tools/detect_xsens_contacts.py --plot-start 25 --plot-duration 10
```

检查旧样本强制阈值生成的离散标签与统计的历史位置见 `git show 1936b40:data/retarget_xsens.py`；
Y13 候选检测不会读取该旧数据或其力台。
