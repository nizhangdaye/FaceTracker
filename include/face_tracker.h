/*
	Last Change : 2024.09.03
*/

#ifndef __FACE_TRACKER_H__
#define __FACE_TRACKER_H__

#undef max
#undef min

#include <opencv2/opencv.hpp>

#define StateType cv::Rect_<float>

class FaceTracker {
private:
    cv::KalmanFilter kf;
    cv::Mat_<float> measurement; // 用以更新的观测矩阵
    std::vector<cv::Rect_<float>> state_history; // 存储历史状态信息，以便后续分析或可视化

public:
    unsigned int id;
    int age; // 跟踪器创建后的帧数
    
    cv::Rect currentPosition; // 保存当前的跟踪位置
    cv::Rect_<float> lastState; // 存储最后一次更新的矩形状态
    int time_since_update; // 自上次状态更新以来经过的时间
    int num_hits; // 跟踪器命中次数，成功匹配到目标的次数
    int continual_hits; // 连续成功匹配到目标的次数
    static int kf_count; 

public:
    FaceTracker();
    FaceTracker(cv::Rect_<float> stateMat);

    ~FaceTracker();

    void initialize(const cv::Rect_<float> stateMat);
    cv::Rect_<float> predict();
    void update(const cv::Rect_<float> new_bbox);
    cv::Rect_<float> xysr2rect(float center_x , float center_y , float s , float r);
    cv::Rect_<float> get_state();

    // double calculateIoU(const cv::Rect& boxA, const cv::Rect& boxB);
};




#endif // __FACE_TRACKER_H__