/*
*/

#ifndef __MULTIOBJECT_TRACKER_H__
#define __MULTIOBJECT_TRACKER_H__

#undef max
#undef min

#include <vector>
#include <map>
#include <set>
#include <limits>
#include <iostream>

#include "face_tracker.h"
#include "hungarian.h"

typedef struct TrackingBox
{
    int frame = 0;
    int id = 0;
    cv::Rect_<float> box;
} TrackingBox;

class MultiObjectTracker {
private:
    int total_frames = 0;
    double total_time = 0.0;
    int frame_count = 0; // 当前帧计数
    int max_missing_frames = 15; // 最大容忍跟踪框未更新的帧数，用以删除老旧的跟踪框
    int min_hits = 3; // 一个框被认为有效的最小帧数
    double iouThreshold = 0.3; // 匹配阈值
    std::vector<FaceTracker> trackers;

    std::vector<cv::Rect_<float>> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix; // 存储预测的边界框和检测边界框之间的IOU
    std::vector<int> assignment; // 每个预测边界框和检测边界框的匹配关系,表示每个预测框和检测框的最佳匹配
    std::set<int> unmatchedDetections; // 未匹配的检测边界框集合
    std::set<int> unmatchedTrajectories; // 未匹配的跟踪轨迹集合

    std::set<int> allItems; // 所有检测的集合
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs; //每个 cv::Point 表示一个匹配对，其中 x 是跟踪器的索引，y 是检测框的索引。
    std::vector<TrackingBox> frameTrackingResult;  // 当前帧跟踪结果
    unsigned int trkNum = 0; // 当前帧跟踪器数量
    unsigned int detNum = 0; // 当前帧的检测数量

    double cycle_time = 0.0;

    double calculateIoU(const cv::Rect_<float> rect1 , const cv::Rect_<float> rect2);
    void removeInactiveTrackers();

public:
    MultiObjectTracker(int max_age = 15 , double iouThreshold = 0.6);
    std::vector<TrackingBox> update(const std::vector<TrackingBox> &detFrameData);

};


#endif // __MULTIOBJECT_TRACKER_H__