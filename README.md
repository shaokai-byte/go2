# Unitree Go2 机器狗学习指南（运动控制 + 相机识别 + YOLO11 模型训练）

> **本仓库：<https://github.com/shaokai-byte/go2>**
>
> 状态：已核实（学习路线依据仓库 `camera` / `sport` / `yolo11` 三个核心目录设计，章节直达链接已逐一核对）
> 维护日期：2026-10-12

## 课程定位

以宇树 Unitree Go2 四足机器狗为载体，基于官方 Python 服务接口 `unitree_sdk2_python`，从 SDK 例程出发，一路走到「自定义目标检测模型 + 落地到机器狗」的一条完整学习路径。本仓库不重复官方教程，只负责在官方 `high_level`（高层运动控制）与 `front_camera`（前视相机）例程之上，补齐三层能力：

1. **动作运动控制**：行走与动作的不同控制方式（`sport`）；
2. **视觉识别控制**：基于相机的目标检测、手势识别、巡线、跟随（`camera`）；
3. **yolo模型训练**：YOLO11 从采集、标注、建数据集、训练到生成模型的全流程（`yolo11`）。

硬件背景：Unitree Go2 四足机器狗，`unitree_sdk2_python` 官方 Python SDK，前视相机，高层运动控制接口（`go2_sport_client.py` 中的 `move` / 动作函数）。

## 面向对象

已经跑通官方 `unitree_sdk2_python` 基础例程、想要继续深入的成员。默认会日常使用电脑与 Python 命令行，遇到不懂的库随用随查（本仓库各子目录均有对应 `README zh.md`）。

## 前置知识

有软性前置：必须**先完成官方 SDK 环境部署**，并成功跑通 `unitree_sdk2_python/example/go2/` 下 `high_level` 与 `front_camera` 两个文件夹的例程。**第一次接触机器狗的学习者，请先完成本仓库根目录 `go2基础开发指南.pdf` 的内容，再进入正文。**

## 学习目标

完成后应能独立做到：

1. 用 `sport` 控制机器狗完成行走与动作（含自定义动作序列）；
2. 用 `camera` 让机器狗基于相机完成目标检测、人体/物体跟随、手势识别、颜色巡线；
3. 用 `yolo11` 走通「采集 → 标注 → 建数据集 → 训练 → 生成 `.pt` 模型」全流程，并把自训模型挂到机器狗相机上做识别。

## 知识模块（学习顺序）

整体分三块：**阶段 0 打地基（官方 SDK），阶段 1 造运动能力（身体），阶段 2–3 给狗装眼睛与大脑（视觉 + 自训模型）**。上一阶段过关再进入下一阶段。表中目录名均链接到本仓库对应目录。

| 阶段 | 仓库目录 | 内容 | 过关标准 | 参考时长 |
| 0 环境部署 | [go2基础开发指南.pdf](./go2基础开发指南.pdf) + [官方 SDK 文档](https://support.unitree.com/home/zh/developer/Python) | 部署 `unitree_sdk2_python`，跑通 `high_level` 与 `front_camera` 例程 | 能远程控制机器狗运动并读取前视相机画面 | 1–2 天 |
| 1 动作控制 | [sport](./sport)（[action](./sport/action) → [walk](./sport/walk)） | 动作函数调用与汇总、行走控制的 4 种实现方式 | 能写出自己的动作序列并控制行走距离 | 2–3 天 |
| 2 相机识别 | [camera](./camera)（[coco_ssd_detect](./camera/coco_ssd_detect) → [gesture_detect_control](./camera/gesture_detect_control) → [line_following](./camera/line_following) → [yolo_detect_follow](./camera/yolo_detect_follow)） | SSD/COCO 目标检测与人体跟随、MediaPipe 手势识别控制、OpenCV 颜色巡线、YOLO 检测与跟随 | 至少跑通一条「识别 → 调动作/跟随」的闭环 | 4–7 天 |
| 3 模型训练 | [yolo11](./yolo11)（[train_yolo11](./yolo11/train_yolo11)） | YOLO11 数据采集、标注、数据集构建、训练、预测全流程 | 用自己的数据集训出一个 `.pt` 并完成一次预测 | 1–2 周 |

### 知识结构树

```text
Go2 知识体系（3 大模块）
├── ① 环境与运动控制（身体）
│   ├── 0 官方 SDK 部署（high_level / front_camera 例程）
│   └── 1 sport：action 动作调用 → walk 行走控制
├── ② 视觉识别（眼睛）
│   ├── coco_ssd_detect：SSD + COCO 目标检测 / 人体跟随
│   ├── gesture_detect_control：MediaPipe 手势识别 → 动作控制
│   ├── line_following：OpenCV 颜色巡线（红/白线）
│   ├── yolo_best.pt：现成训练好的 .pt 模型文件
│   └── yolo_detect_follow：YOLO 目标检测与物体跟随
└── ③ 模型训练（大脑）
    └── yolo11 / train_yolo11：采集 → 标注 → 建数据集 → 训练 → 生成模型 → 预测
```

## 实践内容

- **方法**：每章 = 读子目录 `README zh.md` + 在机器狗上跑通例程 + 自己改一个参数复现效果。改完能复现才算学会。
- **主线只有一条：sport → camera → yolo11**。运动控制是分水岭——`sport` 没吃透，后面所有「识别 → 跟随/动作」的闭环只能「照着跑」，不能「改着用」。
- **顺序不能反**：先会用官方接口控制狗（阶段 1），再让相机去「看」（阶段 2），最后才自己训模型「学会看新东西」（阶段 3）。
- **硬件安全**：机器狗动作测试前，清空工作半径内的人员与易碎物；巡线/跟随测试先让机器狗趴下（low 姿态）或低速运行。
- **路径坑**：`camera` 中 YOLO 检测程序调用模型的路径是**绝对路径**，运行时务必改成自己电脑上模型的实际绝对路径；`.pt` 模型文件不要解压。

## 项目衔接

- 学完阶段 1 可参与：动作编排、步伐控制类项目。
- 学完阶段 2 可参与：人体/物体跟随、手势识别、颜色巡线等实训项目（对应仓库内 `实训项目10-手势识别`、`实训项目11-人体跟随运动` PDF）。
- 学完阶段 3 可参与：自定义目标检测（工业零件、足球场景球/门柱、人物姿态等），把自训模型部署到机器狗及各类相机。

## 学习资料与来源

- [Unitree 官方 Python 服务接口文档](https://support.unitree.com/home/zh/developer/Python)（宇树科技，2026-10 访问）：`unitree_sdk2_python` 环境部署与 `high_level` / `front_camera` 官方例程。
- 本仓库 `go2基础开发指南.pdf`：零基础入门必读。
- 各子目录 `README zh.md`：对应模块的环境、程序说明与运行方法。
- [yolo11/train_yolo11/README.txt](./yolo11/train_yolo11/README.txt)：YOLO11 训练/预测参数的详细说明（模型选择、训练参数、预测参数、图表解读）。
