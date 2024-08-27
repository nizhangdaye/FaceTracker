/*
*/

#ifndef __FACE_TRACKER_H__
#define __FACE_TRACKER_H__

#undef max
#undef min

#include <vector>
#include <map>
#include <set>
#include <limits>

#include <opencv2/opencv.hpp>

#define StateType cv::Rect_<float>

class FaceTracker {
private:
    cv::KalmanFilter kf;
    cv::Mat_<float> measurement;
    cv::Rect currentPosition; // 保存当前的跟踪位置
    cv::Mat templateImage;
    int missingcounter = 0;

public:
    FaceTracker();
    FaceTracker(StateType stateMat);
    void initialize(const cv::Rect& initial_bbox);
    cv::Rect predict();
    void update(const cv::Rect& new_bbox);
    cv::Rect getCurrentPosition() const;
    double calculateIoU(const cv::Rect& boxA, const cv::Rect& boxB);
    bool match(const cv::Rect& detectedBox , double iouThreshold = 0.3);
    int getMissingCounter() const ;
    void increaseMissingCounter(); 
};


class MultiObjectTracker {
private:
    double iouThreshold = 0.6;
    int nextID = 0;
    int maxMissingFrames = 200; // 最大容忍丢失帧数
    std::map<int, FaceTracker> trackers; // ID -> KalmanTracker
    std::set<int> activeIDs; // IDs of active trackers

    bool matchTemplace = true;
    std::vector<cv::Rect> previous_faces; // 存储上一帧的人脸存储结果

    double calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2);
    std::vector<std::vector<double>> computeCostMatrix(const std::vector<cv::Rect>& detected_bboxes);
    std::vector<int> hungarianAlgorithm(const std::vector<std::vector<double>>& costMatrix);
    void removeInactiveTrackers();

public:
    MultiObjectTracker(double iouThreshold = 0.3);
    void update(const std::vector<cv::Rect>& detected_bboxes);
    std::unordered_map<int, cv::Rect> getTrackedObjects() const;
    void initialize(const std::vector<cv::Rect>& detected_bboxes);
    std::set<int> MultiObjectTracker::getActiveIDs() const ;
};


#endif // __FACE_TRACKER_H__