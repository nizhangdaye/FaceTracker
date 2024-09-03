
# LightGlue-OnnxRunner
## 简介
基于人脸检测结果的KalmanFilter人脸跟踪器

# 准备环境
* OpenCV
* NCNN【这里是因为实验用了NCNN框架使用模型来做人脸检测，实际应用还是用onnxruntime的】

### 构建并运行
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

### 

### 许可说明
待定