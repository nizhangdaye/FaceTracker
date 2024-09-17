# KalmanFilter-FaceTracker

## 简介

基于人脸检测结果的KalmanFilter人脸跟踪器

## 整体项目目录

```
│  .gitignore
│  CMakeLists.txt
│  README.md
├─assets【若干提供作人脸检测器的SCRFD模型】
│      scrfd_10g-opt2.bin
│      scrfd_10g-opt2.param
│      scrfd_1g-opt2.bin
│      scrfd_1g-opt2.param
│      scrfd_2.5g-opt2.bin
│      scrfd_2.5g-opt2.param
│      scrfd_34g-opt2.bin
│      scrfd_34g-opt2.param
│      scrfd_500m-opt2.bin
│      scrfd_500m-opt2.param
├─build【自己构建的build目录】
├─include
│      cluster.h【聚类相关操作代码文件】
│      configuration.h
│      face_tracker.h【单个人脸追踪器相关代码】
│      hungarian.h【匈牙利算法实现代码】
│      multiobject_tracker.h【MOT实现代码】
│      scrfd_detector.h
│      utils.h
├─src
│      face_tracker.cpp
│      hungarian.cpp
│      main.cpp
│      multiobject_tracker.cpp
│      scrfd_detector.cpp
│
└─third_party
    ├─ncnn
    └─opencv4
```

## 准备环境

* OpenCV
* NCNN【这里是因为实验用了NCNN框架使用模型来做人脸检测，实际应用还是用onnxruntime的】

## 构建

```
# 在CMakeLists.txt所在的目录下构建build目录
mkdir build
# 进入构建的build目录并使用cmake构建工程配置
cd build
cmake ..
# 编译
cmake --build .
# 如果需要指定版本编译，就如下操作
# cmake --build . --config Debug
# cmkae --build . --config Release
# 然后将dll文件放进build目录下Release或Debug，*.exe同级目录即可

```

## 运行

```
# 在build目录下运行
.\Release\demo.exe C:\\Users\\Administrator\\Desktop\\yolo-face-with-landmark-master\\ncnn_project\\data\\singlevideo
# 运行结束后控制台输出如下
C:\Users\Administrator\Desktop\yolo-face-with-landmark-master\ncnn_project\build>.\Release\demo.exe singlevideo
[INFO] Video saved to: singlevideo\result\video17_5.mp4
# 可在输出目录下见到输出的视频结果文件

```

## 许可说明

MIT License