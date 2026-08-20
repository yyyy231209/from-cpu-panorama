# rga_gpu_cpu_stitch

> 4 路摄像头全景拼接 · 第一代纯 CPU 架构（Rockchip RK3588）
>
> 4-camera panoramic stitching · 1st-generation pure-CPU architecture (Rockchip RK3588)

本项目是 4 路摄像头全景拼接系统的**第一代架构**：拼接本身完全由 CPU 完成（单应矩阵 remap + 逐像素归属融合），不依赖 GPU 硬件加速，因此代码最直观、最容易读懂，是全系列的性能基线（baseline）。拼接后的全景画布经 Rockchip RGA 硬件缩放，送入 RK3588 三核 NPU 做裂缝（crack）检测，最终以 MJPEG 流对外输出。

This is the **first-generation** 4-camera panoramic stitching system. Stitching is done entirely on the CPU (homography remap + per-pixel ownership blending) with no GPU acceleration, which makes the code the most intuitive and readable — it serves as the performance baseline of the whole series. The stitched canvas is resized by the Rockchip RGA hardware, fed to the triple-core NPU of the RK3588 for crack detection, and streamed out as MJPEG.

---

## 三代演进 / Three-Generation Evolution

| 代次 Gen | 项目 Project | 拼接实现 Stitching backend |
| --- | --- | --- |
| 第 1 代 1st | **rga_gpu_cpu_stitch（本仓库 / this repo）** | 纯 CPU（`cv::remap`） |
| 第 2 代 2nd | `warp_pipe_project` | GPU warp（OpenCL 管线） |
| 第 3 代 3rd | `zero-copy-vpu-gpu-rga-stitch` | GPU + RGA（零拷贝 zero-copy） |

本仓库作为第一代基线，保留了最直观、最易读的 CPU 拼接实现，用于理解全景拼接的数据流，以及后续 GPU/RGA 加速的动机。

This repo is the baseline of the series: it keeps the most intuitive and readable CPU stitching implementation, which documents the data flow and motivates the later GPU/RGA acceleration.

---

## 架构概览 / Architecture

```
 +-----------+    +-----------+    +-----------+    +-----------+
 | video21   |    | video23   |    | video25   |    | video27   |
 | V4L2 MJPG |    | V4L2 MJPG |    | V4L2 MJPG |    | V4L2 MJPG |
 +-----+-----+    +-----+-----+    +-----+-----+    +-----+-----+
       |                |                |                |
  worker_cpu ×4   worker_cpu ×4    worker_cpu ×4    worker_cpu ×4
   (capture +       (capture +        (capture +       (capture +
    MJPG decode)     MJPG decode)      MJPG decode)     MJPG decode)
       |                |                |                |
       +----------------+----------------+----------------+
                        |
                  CPUQueue (4 × SingleQueue)
                        |
                  stitch thread
              cpu_warp.cpp / cpu_H_fuc.cpp
            (remap + ownership blending)
                        |
                  square padding
                        |
                  RGA imresize (HW)
                        |
                  NPUQueue (dma-heap, 6 bufs + display)
                        |
          worker_npu ×3 (RKNN YOLOv8 crack detect)
                        |
                  main display loop
               RGA imcopy + RGB2BGR + FPS text
                        |
                MjpegStreamer (HTTP :8080)
```

**数据流 / Data flow**

1. **采集 / Capture** — 4 个 `worker_cpu` 线程通过 V4L2 `mmap` 采集 320×240 MJPEG 帧，`cv::imdecode` 解码后推入各自的 `SingleQueue`。
2. **拼接 / Stitch** — 拼接线程从 4 个队列各取一帧，调用 `stitch_four_cpu()` 完成 CPU 全景拼接（`cv::remap` + owner map 逐像素归属）。
3. **缩放 / Resize** — 全景画布填充为正方形，经 RGA `imresize` 缩放到模型输入尺寸，写入 NPU 队列的共享 buffer。
4. **推理 / Inference** — 3 个 `worker_npu` 线程用 RKNN YOLOv8 做裂缝检测，并用 RGA `imfill` 画出检测框。
5. **输出 / Output** — 主循环取出就绪 buffer，RGA `imcopy` + 颜色转换后叠加 FPS，推给 MJPEG 流（HTTP 8080）。

---

## 目录与文件说明 / File Layout

| 文件 File | 说明 Description |
| --- | --- |
| `main.cpp` | 入口：加载单应矩阵、启动各线程、主显示循环与退出清理。Entry point. |
| `cpu_H_fuc.{h,cpp}` | 加载 H12/H32/H42 单应矩阵、计算画布尺寸、生成 remap 映射与逐像素归属图（owner map）。Load homographies, compute canvas & remap maps & owner map. |
| `cpu_warp.{h,cpp}` | 拼接线程：取帧 → `stitch_four_cpu` → 正方形填充 → RGA 缩放入队。Stitch thread. |
| `cpu_queue.{h,cpp}` | 每路摄像头的环形缓冲队列（4 槽位）。Per-camera ring buffer (4 slots). |
| `worker_cpu.{h,cpp}` | V4L2 采集线程（mmap + MJPEG 解码）。V4L2 capture thread. |
| `worker_npu.{h,cpp}` | RKNN YOLOv8 推理线程 + RGA 画框。NPU inference thread. |
| `npu_queue.{h,cpp}` | 基于 `dma-heap` 的共享 buffer 队列，状态机 FREE/FILLED/PROCESSING/READY。dma-heap shared buffer queue with state machine. |
| `mjpeg_streamer.{h,cpp}` | 基于 BSD socket 的 HTTP MJPEG 流服务器。HTTP MJPEG streaming server. |
| `postprocess_crack.{h,cpp}` | 裂缝检测后处理：3 个 stride（8/16/32）、DFL 解码框、NMS、单类别 `crack`。Crack-detection postprocess. |
| `test_model.cpp` | 离线单图推理测试工具。Standalone offline model test tool. |
| `crack_labels.txt` | 类别标签（`crack`）。Class labels. |
| `H12.bin / H32.bin / H42.bin` | 3×3 `double` 单应矩阵（二进制，9×8 字节），相机 2/3/4 相对相机 1 的变换。3×3 double homographies (binary). |
| `CMakeLists.txt` | CMake 构建脚本。Build script. |
| `rga_gpu_cpu_stitch.service` | systemd 开机自启单元示例。Example systemd unit. |
| `include/` | 内置的 yolo 辅助 SDK（rknn_api、yolov8 封装、postprocess、图像工具）。Vendored yolo helper SDK. |

---

## 构建 / Build

目标平台为 Rockchip **RK3588**（`aarch64` Linux）。

Target platform: Rockchip **RK3588** (`aarch64` Linux).

### 依赖 / Dependencies

- OpenCV（≥ 3.x，含 `opencv2/opencv.hpp`）—— `sudo apt install libopencv-dev`
- librga（Rockchip RGA 硬件 2D 加速）—— 由 RK3588 SDK 提供，或从 Rockchip 官方 `librga` 仓库编译安装；部分 RK3588 发行版自带 `librga-dev` 包
- rknnrt（Rockchip RKNN Runtime，NPU）—— 随 RKNN-Toolkit2 / RKNPU2 SDK 提供（`librknnrt.so`），安装后置于系统库路径
- libturbojpeg —— `sudo apt install libturbojpeg-dev`
- OpenMP（编译选项 `-fopenmp`）—— 随 GCC 的 `libgomp` 提供；缺失时 `sudo apt install libgomp1`
- 可选：`3rdparty/` 目录（本地第三方库，缺省时使用系统 `/usr/lib` 等路径）

> 说明：`include/` 目录已随仓库内置（RKNN API、YOLOv8 封装、后处理、图像工具）。`librga` / `rknnrt` / `libturbojpeg` 为系统库，请自行安装（RK3588 SDK 或发行版软件包）。
>
> Note: the `include/` directory is bundled with this repo. `librga` / `rknnrt` / `libturbojpeg` are system libraries — install them yourself (from the RK3588 SDK or distro packages).

### 步骤 / Steps

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

产物为可执行文件 `rga_gpu_cpu_stitch`。The output binary is `rga_gpu_cpu_stitch`.

---

## 运行 / Run

```bash
# 用法 / usage:
#   ./rga_gpu_cpu_stitch [H矩阵目录] [rknn模型路径]
#   ./rga_gpu_cpu_stitch [H-matrix dir] [rknn model path]
./rga_gpu_cpu_stitch . best.rknn
```

- 第一个参数为包含 `H12.bin` `H32.bin` `H42.bin` 的目录（默认 `.`）。
- 第二个参数为 RKNN 模型路径（默认 `best.rknn`，可在参数中覆盖）。
- 摄像头设备节点默认为 `/dev/video21` `/dev/video23` `/dev/video25` `/dev/video27`（V4L2 MJPEG，320×240）。如需调整，请修改 `main.cpp` 中的 `cam_devs`。
- MJPEG 流地址：`http://<设备IP>:8080/`。
- 按 `Ctrl-C` 退出。

```text
First argument: directory containing H12.bin H32.bin H42.bin (default ".").
Second argument: RKNN model path (default "best.rknn").
Camera device nodes default to /dev/video21 /dev/video23 /dev/video25 /dev/video27 (V4L2 MJPEG, 320x240).
MJPEG stream: http://<device-ip>:8080/ .
Press Ctrl-C to exit.
```

### 离线模型测试 / Offline model test

`test_model.cpp` 是独立的单图测试工具（不含摄像头），验证 RKNN 模型 + 后处理是否正常：

`test_model.cpp` is a standalone single-image test tool (no camera) to verify the RKNN model + postprocess:

```bash
g++ -std=c++17 -O2 test_model.cpp postprocess_crack.cpp \
    include/yolo/yolov8.cc include/include_yolo/postprocess.cc \
    include/include_yolo/dma_alloc.cpp \
    include/utils/file_utils.c include/utils/image_utils.c \
    include/utils/image_drawing.c \
    -Iinclude -Iinclude/yolo -Iinclude/include_yolo -Iinclude/utils \
    $(pkg-config --cflags --libs opencv4) -lrknnrt -lrga -lturbojpeg -lm -o test_model
./test_model <image.jpg> <best.rknn>
```

### 开机自启 / systemd autostart

参考 `rga_gpu_cpu_stitch.service`，按需修改 `ExecStart` / `WorkingDirectory` 路径后安装：

See `rga_gpu_cpu_stitch.service`; adjust `ExecStart` / `WorkingDirectory` paths and install:

```bash
sudo cp rga_gpu_cpu_stitch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rga_gpu_cpu_stitch
```

---

## 模型与标定数据 / Model & Calibration Data

### 裂缝检测模型 / Crack detection model

`best.rknn` 是裂缝（crack）检测的 YOLOv8 模型（输入 1024×1024）。它可由 RKNN-Toolkit2 将 `rknn_model_zoo` 中的 YOLOv8 模型转换得到；转换时请保证输出头与 `postprocess_crack.cpp` 的后处理一致（3 个 stride 8/16/32、DFL 解码、单类别 `crack`）。运行或离线测试时通过命令行第二个参数指定模型路径。

`best.rknn` is the YOLOv8 crack-detection model (1024×1024 input). It can be produced by converting the YOLOv8 model from `rknn_model_zoo` with RKNN-Toolkit2; keep the output heads consistent with `postprocess_crack.cpp` (strides 8/16/32, DFL decode, single class `crack`).

### 单应矩阵 / Homographies

`H12.bin` `H32.bin` `H42.bin` 为相机 2/3/4 到相机 1 的 3×3 单应矩阵（`double`，行主序，共 72 字节）。它们由 `cv2.findHomography` 对 4 路相机画面中的匹配特征点标定得到；仓库内附带的矩阵为示例值，实际部署请替换为你自己的标定结果。

`H12.bin`, `H32.bin`, `H42.bin` are the 3×3 homographies (row-major `double`, 72 bytes each) from cameras 2/3/4 to camera 1. They are computed with `cv2.findHomography` from matched feature points across the four camera views; the bundled matrices are examples — replace them with your own calibration results in production.

---

## 踩坑 / 故障排查 / Troubleshooting

### dma-heap 节点权限

- NPU 队列使用 `/dev/dma_heap/system-uncached` 分配共享 buffer。若启动时报 `open /dev/dma_heap/system-uncached fail!`，通常是权限不足或节点名不一致：可用 `sudo` 运行验证，或添加 udev 规则放宽权限，或按内核实际节点名修改 `npu_queue.h` 中的 `DMA_HEAP_PATH`（`system` / `system-uncached` / `cma-uncached` 因内核配置而异）。

### 摄像头设备节点

- 默认设备为 `/dev/video21` `/dev/video23` `/dev/video25` `/dev/video27`。用 `v4l2-ctl --list-devices` 确认节点存在；权限不足时把当前用户加入 `video` 组。如节点编号不同，修改 `main.cpp` 中的 `cam_devs`。

### RGA / rknnrt 常见错误码

- 构建期报 `librga not found` / `librknnrt not found`：未安装对应库，按「依赖」一节补齐，或把库目录加入 `CMAKE_LIBRARY_PATH`。
- 运行期 `rknn_init fail! ret=-6`：模型文件损坏或与 `rknn_api.h` 版本不匹配；`ret=-13`：模型目标平台与当前 SoC 不匹配（需用 RKNN-Toolkit2 按 RK3588 重新转换）。
- RGA 画框/缩放异常时，先确认 RGA 设备节点可访问，并核对 `rga/im2d.h` 与 `librga` 版本一致。

## 许可证 / License

[MIT](./LICENSE) · Copyright (c) 2026

## 参与贡献 / Contributing

见 [CONTRIBUTING.md](./CONTRIBUTING.md)。
