# 实验数据

**Y13 接触估计和控制只使用 `y13_MVN_3_8_01.mvn` / `.xlsx`。** 新实验者的体重为 62.5 kg、身高为 1.72 m（用户提供）。`xsense_raw.xlsx` 与 `fp_raw.csv` 是另一组、彼此配对的旧样本；旧 H5 属于其他采集，不与 Y13 配对。配置见 `config/y13_trial.json`，流程说明见 `docs/y13_xsens_walk.md`。

本目录保留原始导出文件；预览脚本位于 `tools/`，新生成结果写入 Git 忽略的 `outputs/`。
旧样本 Xsens 与力台记录尚未完成实验时钟、坐标系及足部归属标定；如分析该旧样本，二者只能在确认时间同步和受试者身份后配对。**Y13 流程不得读取 `fp_raw.csv`**。

## Xsens：`xsense_raw.xlsx`

文件内元数据记录：MVN 2024.2.1，HD 处理，60 Hz，记录时间为 2026-06-11 08:16:39 UTC。
2675 帧，23 个身体段，包括骨盆、躯干、头、双臂及双腿。
各工作表通常首列为帧号，首行为字段名；以下 shape 指去除帧号后按身体段重排的数组。

| 工作表/内容 | shape |
|---|---|
| Segment Position | `(2675, 23, 3)` |
| Segment Velocity / Acceleration | 各 `(2675, 23, 3)` |
| Segment Angular Velocity / Acceleration | 各 `(2675, 23, 3)` |
| Segment Orientation - Quat | `(2675, 23, 4)` |
| Center of Mass：位置、速度、加速度 | 各 `(2675, 3)` |
| Sensor Free Acceleration | `(2675, 23, 3)` |
| Sensor Orientation - Quat | `(2675, 23, 4)` |

还包括 Euler、关节角、磁场等工作表。身体段的运动和姿态已由 Xsens 处理；本文件不提供
直接重跑 VQF 所需的完整原始陀螺仪和含重力加速度流。Free Acceleration 不能作为原始加速度替代。
文件名中的 `raw` 表示仓库保留的原始导出文件，不表示所有工作表都是未经处理的传感器读数。

预览整个身体骨架与导出质心，或者保存指定帧（默认输出 `outputs/xsens_preview.png`）：

```bash
python tools/preview_xsens.py
MPLBACKEND=Agg python tools/preview_xsens.py --frame 0 --no-show
```

## 力台：`fp_raw.csv`

共 8446 行、16 列：一列 `SysTime`，以及三块力台各五列：

```text
FP{1,2,3}_Fx, Fy, Fz, Px, Py
```

`Fx/Fy/Fz` 为力，`Px/Py` 为 COP 位置。原实验记录说明力单位为 N，位置按 m 解释；
接入动力学前仍需核对采集导出单位。文件时间戳覆盖约 42.19 秒。

原项目数据说明中记录：Test Subject=Walk，Test Item=1，Age=18，Body Weight=85.0，
Load Weight=0.0，Cutoff Threshold=0.0，Sample Rate=512。
这些是继承的实验备注，尚未独立核实；512 Hz 为标称值，不能代替对实际时间戳的检查。

CSV 没有 `Mx/My/Mz` 或自由力矩。通过力与 COP 可以构造关于选定参考点的 `r × F`，
但缺失自由力矩，不能称为完整测得的六维足部 wrench。
尚需确认各力台的轴向、正负号、世界位置、与 Xsens 的时间对应及左右脚归属。

预览（默认输出 `outputs/force_plate_preview.png`）：

```bash
python tools/preview_force_plates.py
MPLBACKEND=Agg python tools/preview_force_plates.py --no-animate --no-show
```

预览沿用近似 0.6 × 0.4 m 的力台矩形和示意排列，使用 `|Fz|` 展示受力强度。
该排列不是实验世界标定，绝对值处理也不是动力学外力坐标转换。
当前不提供将两份记录自动对齐的入口；对齐应在明确实验时钟关系后实现。

## y13 Xsens：`y13_MVN_3_8_01.mvn` / `.xlsx`

Excel 元数据记录 MVN 2024.2.1、240 Hz、Live 处理，52462 帧，约 218.59 秒，
Recorded Date UTC 为 2026-11-06 06:06:15（北京时间 14:06:15）。
虽有 23 个骨段输出，传感器四元数全程有效的只有骨盆及双侧大腿、小腿、足部七处；
其余上身和足趾传感器通道全零，不能据此声称独立测得全身动作。

用户更正：此记录与旧数据来自不同实验者。当前目标是将这七个骨段姿态拟合到简化 MuJoCo 人体，再进行动力学追踪；不进行与旧 H5 或旧力台的配对比较。
约 117.03 秒、176.95 秒存在姿态和位置同时跳变，骨盆水平轨迹也不是已验证的跑台根位移。
当前流程、用途和限制见 [新 Xsens 开发说明](../docs/y13_xsens_walk.md)，
拟合统计由当前流程生成到 `outputs/y13/fit.json`。
`.mvn` 是原生二进制工程文件，本次没有解码，应保留用于 Xsens 软件中的工程检查和重新导出。
这两份原始采集文件未随代码提交；在运行流程前需要自行放置到上述路径。

## Y13 时序接触估计

`tools/detect_xsens_contacts.py` 使用 Y13 导出的双足骨段线加速度、角速度、角加速度、
小腿角速度和足部相对骨盆速度，拟合双状态对角 Gaussian HMM。该模型不读取旧样本，
不读取 `fp_raw.csv`，且会校验拟合文件的 XLSX 源路径与人体模型哈希。

当前只有 Y13 平地跑台记录，还没有同步接触标签。模型输出的低活动状态只是**候选接触样状态**；
posterior 未经接触真值校准，接触时刻和支撑状态尚不能作为已验证标签。坡道/楼梯泛化也尚未验证。
候选 CSV/NPZ/图写入 `outputs/y13/contact/`，说明见 `docs/y13_contact_inference.md`。
