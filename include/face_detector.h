#ifndef __FACE_DETECTOR_H__
#define __FACE_DETECTOR_H__

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

#include <ncnn/net.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

struct Point{
    float _x;
    float _y;
};

struct Bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

class FaceDetector
{
    public:
        FaceDetector(std::string param_path , std::string model_path);
        void detect(cv::Mat im_bgr,int long_size, std::vector<Bbox> &prebox);
    
    private:
        ncnn::Net facenet;
        ncnn::Mat img;
        int num_thread = 1;
        bool letter_box = true;
        float score_threh = 0.15;
        const float mean_vals[3] = { 0,0,0 };\
        const float norm_vals[3] = { 1.0 / 255.0, 1.0 / 255.0, 1.0 /255.0};
        std::vector<int> minsize0 = {12  , 20 , 32  };
        std::vector<int> minsize1 = {48  , 72 , 128 };
        std::vector<int> minsize2 = {196 , 320, 480 };
    
    private:
        void nms(std::vector<Bbox> &input_boxes, float nms_thresh);
        cv::Mat preprocess(cv::Mat src,const int long_size);
        void postprocess(ncnn::Mat pre,std::vector<int> anchor,std::vector<Bbox> & prebox,
                float confidence_threshold, int net_w, int net_h ,int ori_w , int ori_h);
        static inline float sigmoid(float x);

};

#endif //__FACEDETECTOR_H__