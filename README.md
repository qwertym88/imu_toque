# 三维人体关节净力矩与接触力评估

本项目基于简化的 MuJoCo 人体刚体模型，面向 Xsens 运动数据和实验测力数据的三维动力学分析。
目标输出是左右髋 pitch/roll/yaw、膝 pitch、踝 pitch/roll 的关节净力矩，以及足部接触力和力矩。

当前已完成 **人体模型派生与力学验收**；尚未完成人体个体化、Xsens 运动拟合及实验逆动力学。
模型不包含肌肉激活或肌肉力分配，当前也没有可直接输出受试者力矩的完整分析入口。

## 当前模型

采用 MS-Human Locomotion 的固定版本，派生为 10 个活动刚体、21 个速度自由度：
浮动基座 6、左右下肢 12、躯干 3；`nq=22`，`nv=21`，`nu=15`。
模型具有完整三维惯性参数，提供直接关节力矩接口，并保留上游来源、许可证和重建校验和。

膝关节简化为固定轴，脚趾和髌骨固定，上身内部姿态固定。踝关节保留上游斜向解剖转轴，
pitch/roll 是关节坐标别名。当前质量约 60.22 kg，尚未按实验记录中的 85 kg 受试者个体化。
详细定义和简化边界见 [模型说明](model/ms_human_locomotion/README.md)。

![人体模型静态预览](model/ms_human_locomotion/preview.png)

## 目录

```text
model/ms_human_locomotion/
  human.xml                 独立加载的简化人体模型
  derivation.json           身体合并、关节轴和惯性来源
  inspection.json           维度、质量与静态检查结果
  preview.png               正面和侧面预览
  source/                   固定提交的上游来源、许可证和必需资源
  README.md                 模型定义、构建和验证说明
data/
  xsense_raw.xlsx            Xsens 导出的身体段与传感器结果
  fp_raw.csv                 三块力台的力与 COP 记录
  README.md                 数据字段、shape 与待标定事项
tools/
  build_ms_human.py          离线重建人体模型
  inspect_human_model.py     模型检查、静态查看与渲染
  preview_xsens.py           原始 Xsens 结果预览
  preview_force_plates.py    原始力台数据预览
tests/
  test_human_model.py        人体模型力学与重建验收
outputs/                    本地生成结果（Git 忽略）
```

## 运行

已在 Python 3.12、MuJoCo 3.9.0 验证；`requirements.txt` 记录此次验证的直接依赖版本。
已有 `humanoid_sim` 环境可直接使用；新环境安装后执行以下命令：

```bash
python -m pip install -r requirements.txt
python tools/inspect_human_model.py
python -m unittest discover -s tests -v
```

桌面查看静态模型：

```bash
python tools/inspect_human_model.py --viewer
```

从仓库内固定来源离线重建，不需要下载完整肌肉模型：

```bash
python tools/build_ms_human.py
```

生成本地预览；无头模型渲染需要可用的 EGL 后端：

```bash
MUJOCO_GL=egl python tools/inspect_human_model.py --preview outputs/human_preview.png
MPLBACKEND=Agg python tools/preview_xsens.py --frame 0 --no-show
MPLBACKEND=Agg python tools/preview_force_plates.py --no-animate --no-show
```

检查 hooks 使用 `.pre-commit-config.yaml`。上游 `source/` 保留原始字节，排除自动格式改写，
其完整性由模型测试中的 SHA-256 校验保证。

## 后续开发顺序

1. 依据受试者数据确定人体尺寸、关节中心和质量/惯性缩放。
2. 将 Xsens 身体段运动拟合到人体模型，检查关节角、足部轨迹和拟合残差。
3. 对齐运动与力台时钟、坐标、单位和足归属，计算实验外力已知时的关节净力矩与基座残差。
4. 在上述基线验证后，开展外接触力未知时的估计与实验比较。

力台当前没有完整六维力矩记录；原始 Xsens 文件是处理后的导出结果，并非完整原始加速度计/
陀螺仪流。相关数据约束见 [数据说明](data/README.md)。

旧 SZAR 机器人、机器人运动重定向和 WBC 控制实验已从本分支工作树移除。
需要追溯时查看清理前提交 `1936b40`；当前开发只围绕上述人体动力学分析流程。
