/*
*/

#ifndef SCRFD_ONNX_RUNNER_H
#define SCRFD_ONNX_RUNNER_H

#undef max
#undef min

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "configuration.h"

struct FaceObject
{
    cv::Rect_<float> rect;
    float prob;
};

static inline float intersection_area(const FaceObject& a, const FaceObject& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

class SCRFDOnnxRunner
{
    private:
        float conf_threshold = 0.30f;
        float nms_threshold = 0.45f;
        int input_width = 640;
        int input_height = 640;
        unsigned int fmc = 3; // feature map count
        bool use_kps = false;
        unsigned int num_anchors = 2;
        std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]
        
        const bool keep_ratio = true;
        static constexpr const unsigned int nms_pre = 1000;
        static constexpr const unsigned int max_nms = 30000;
    
    private:
        const unsigned int num_threads;
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;
        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<char*> InputNodeNames;
        std::vector<std::vector<int64_t>> InputNodeShapes;

        std::vector<char*> OutputNodeNames;
        std::vector<std::vector<int64_t>> OutputNodeShapes;

        std::vector<Ort::Value> output_tensors;

    private:
        /* *********************** */
        cv::Mat NormalizeImage(cv::Mat& Image);
        cv::Mat ResizeImage(const cv::Mat srcImage, int* newh, int* neww, int* top, int* left);
        cv::Mat RGB2Grayscale(cv::Mat& Image);

        /* *********************** */
        cv::Mat PreProcess(Configuration cfg , const cv::Mat& srcImage);
        int Inference(Configuration cfg , const cv::Mat& src);
        int PostProcess(Configuration cfg);
        
    public:
        explicit SCRFDOnnxRunner(Configuration cfg , unsigned int num_threads = 1);
        ~SCRFDOnnxRunner();

        float GetConfThreshold();
        void SetConfThreshold(float thresh);

        float GetNMSThreshold();
        void SetNMSThreshold(float threshold);

        int InitOrtEnv(Configuration cfg);  
    
        std::vector<FaceObject> InferenceImage(Configuration cfg , const cv::Mat& srcImage);
        static void Draw(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects);
};

#endif // SCRFD_ONNX_RUNNER_H
