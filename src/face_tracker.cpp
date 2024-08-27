/*

*/

#include <opencv2/core/base.hpp>

#include "../include/face_tracker.h"

FaceTracker::FaceTracker(): missingcounter(0)
{

}

FaceTracker::FaceTracker(StateType stateMat) : missingcounter (0) 
{
    int stateNum = 7;
    int measureNum = 4;
    // Initialize Kalman Filter parameters
    kf.init(stateNum, measureNum , 0); // State dimension, measurement dimension, control dimension
    kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) << 1, 0, 0, 0, 1, 0, 0, \
						   0, 1, 0, 0, 0, 1, 0,
						   0, 0, 1, 0, 0, 0, 1,
						   0, 0, 0, 1, 0, 0, 0,
						   0, 0, 0, 0, 1, 0, 0,
						   0, 0, 0, 0, 0, 1, 0,
						   0, 0, 0, 0, 0, 0, 1);

    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    // initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;

}

cv::Rect FaceTracker::getCurrentPosition() const 
{
    return currentPosition;
}

int FaceTracker::getMissingCounter() const 
{
    return missingcounter;
}

void FaceTracker::increaseMissingCounter()
{
    missingcounter++;
}

void FaceTracker::initialize(const cv::Rect& initial_bbox) 
{
    cv::Point2f center = (initial_bbox.br() + initial_bbox.tl()) * 0.5f;
    kf.statePost = (cv::Mat_<float>(4, 1) << center.x, center.y, 0, 0);
    currentPosition = initial_bbox;
}

cv::Rect FaceTracker::predict() 
{   
    std::cout << "1.5" << std::endl;
    cv::Mat prediction = kf.predict();
    std::cout << "1.6" << std::endl;

    cv::Point2f predicted_center(prediction.at<float>(0), prediction.at<float>(1));
    currentPosition = cv::Rect(predicted_center.x - currentPosition.width / 2, predicted_center.y - currentPosition.height / 2, 
            currentPosition.width, currentPosition.height);
    

    return currentPosition;
}

void FaceTracker::update(const cv::Rect& new_bbox) 
{
    cv::Point2f center = (new_bbox.br() + new_bbox.tl()) * 0.5f;
    measurement = (cv::Mat_<float>(2, 1) << center.x, center.y);
    kf.correct(measurement);
    currentPosition = new_bbox; // Update current position
    missingcounter = 0;
}

bool FaceTracker::match(const cv::Rect& detectedBox , double iouThreshold)
{
    double iou = calculateIoU(currentPosition, detectedBox);
    if (iou > iouThreshold) // 优先执行IOU匹配
    {
        std::cout << iou << std::endl;
        return true;
    } else // 若IOU匹配失败才进行模板匹配
    {   
        return false;
    }
}

double FaceTracker::calculateIoU(const cv::Rect& boxA, const cv::Rect& boxB) 
{
        int xA = (std::max)(boxA.x, boxB.x);
        int yA = (std::max)(boxA.y, boxB.y);
        int xB = (std::min)(boxA.x + boxA.width, boxB.x + boxB.width);
        int yB = (std::min)(boxA.y + boxA.height, boxB.y + boxB.height);
        
        int interArea = (std::max)(0, xB - xA) * (std::max)(0, yB - yA);
        
        int boxAArea = boxA.width * boxA.height;
        int boxBArea = boxB.width * boxB.height;
        
        double iou = (double)interArea / (boxAArea + boxBArea - interArea);
        
        return iou;
}

/* * **************************************** * */
/* * **************************************** * */
/* * **************************************** * */

MultiObjectTracker::MultiObjectTracker(double iouThreshold) : iouThreshold(iouThreshold), nextID(0) {}

std::set<int> MultiObjectTracker::getActiveIDs() const 
{
    return activeIDs;
}

void MultiObjectTracker::initialize(const std::vector<cv::Rect>& detected_bboxes) 
{
    for (const auto& rect : detected_bboxes) {
        FaceTracker tracker(rect);
        tracker.initialize(rect);
        int id = nextID++;
        trackers[id] = tracker;
        activeIDs.insert(id);
        previous_faces.emplace_back(rect); // 对初始帧的人脸进行存储
    }
}

void MultiObjectTracker::update(const std::vector<cv::Rect>& detected_bboxes) 
{
    for (auto& [id, tracker] : trackers) {
        tracker.predict(); // 预测所有活动的跟踪器
    }
    std::cout << "2" << std::endl;
    std::set<int> matchTrackers;
    std::vector<std::vector<double>> costMatrix = computeCostMatrix(detected_bboxes);
    std::cout << "3" << std::endl;

    // Apply custom matching
    for (int i = 0; i < costMatrix.size(); ++i) {
        for (int j = 0; j < costMatrix[i].size(); ++j) {
            if (trackers.find(i) != trackers.end()) {
                // Add a custom match check (e.g., IoU threshold)
                if (trackers[i].match(detected_bboxes[j], iouThreshold)) {
                    costMatrix[i][j] = 1.0 - calculateIoU(trackers[i].getCurrentPosition(), detected_bboxes[j]);
                } else {
                    costMatrix[i][j] = 1.0; // Max cost if not matched
                }
            }
        }
    }
    std::cout << "4" << std::endl;

    std::vector<int> assignments = hungarianAlgorithm(costMatrix);
    std::cout << "5" << std::endl;

    for (int idx = 0 ; idx < assignments.size() ; idx++)
    {
        if (assignments[idx] != -1)
        {
            int trackID = assignments[idx];
            trackers[trackID].update(detected_bboxes[idx]);
            matchTrackers.insert(trackID);
        } else {
            FaceTracker newTracker;
            newTracker.initialize(detected_bboxes[idx]);
            int newID = nextID++;
            trackers[newID] = newTracker;
            activeIDs.insert(newID);
        }
    }

    for (auto& [id , tracker] : trackers)
    {
        if (matchTrackers.find(id) == matchTrackers.end())
        {
            // std::cout << id << std::endl;
            tracker.increaseMissingCounter();
        }
    }

    removeInactiveTrackers();
}

std::unordered_map<int, cv::Rect> MultiObjectTracker::getTrackedObjects() const 
{
    std::unordered_map<int, cv::Rect> trackedObjects;
    // std::cout << "activeIDs.size() : " << activeIDs.size() << std::endl;
    for (int id : activeIDs) {
        trackedObjects[id] = trackers.at(id).getCurrentPosition();
    }
    return trackedObjects;
}

double MultiObjectTracker::calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) 
{
    int x1 = (std::max)(rect1.x, rect2.x);
    int y1 = (std::max)(rect1.y, rect2.y);
    int x2 = (std::min)(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = (std::min)(rect1.y + rect1.height, rect2.y + rect2.height);
    
    int interArea = (std::max)(0, x2 - x1) * (std::max)(0, y2 - y1);
    int unionArea = rect1.area() + rect2.area() - interArea;
    
    return static_cast<double>(interArea) / unionArea;
}

std::vector<std::vector<double>> MultiObjectTracker::computeCostMatrix(const std::vector<cv::Rect>& detected_bboxes) 
{
    // 创建一个cost矩阵，大小为numTrackers * numDetections，初始化为1.0，则计算代价为1.0-IoU
    std::vector<std::vector<double>> costMatrix(trackers.size(), std::vector<double>(detected_bboxes.size(), 1.0));
    int i = 0;
    for (int id : activeIDs) {
        cv::Rect predictedBox = trackers[id].predict();
        for (int j = 0; j < detected_bboxes.size(); ++j) {
            costMatrix[i][j] = 1.0 - calculateIoU(predictedBox, detected_bboxes[j]);
        }
        ++i;
    }

    return costMatrix;
}

std::vector<int> MultiObjectTracker::hungarianAlgorithm(const std::vector<std::vector<double>>& costMatrix)
{
    int num_trackers = costMatrix.size();
    int num_detections = costMatrix[0].size();
    std::vector<int> assignment(num_detections , -1);

    cv::Mat costMat(num_trackers , num_detections , CV_64F);
    for (int i = 0 ; i < num_trackers ; ++i)
    {
        double* rowPtr = costMat.ptr<double>(i);
        for (int j = 0 ; j < num_detections ; j++)
        {
            rowPtr[j] = costMatrix[i][j];
        }
    }

    std::vector<bool> assignedTrackers(num_trackers, false); // 跟踪器分配标记
    std::vector<bool> assignedDetections(num_detections, false); // 检测框分配标记

    for (int j = 0 ; j < num_detections ; j++)
    {
        double minCost = std::numeric_limits<double>::max();
        int bestMatch = -1;
        for (int i = 0 ; i < num_trackers ; i++)
        {
            if (!assignedTrackers[i] && costMatrix[i][j] < minCost)
            {
                minCost = costMatrix[i][j];
                bestMatch = i;
            }
        }
        if (bestMatch != -1)
        {
            assignment[j] = bestMatch;
            assignedTrackers[bestMatch] = true;
            assignedDetections[j] = true;
        }
    }

    return assignment;
}


void MultiObjectTracker::removeInactiveTrackers()
{
    std::set<int> idxToRemove;
    for (const auto& id : activeIDs)
    {
        if (trackers[id].getMissingCounter() > maxMissingFrames)
        {
            idxToRemove.insert(id);
        }
    }

    for (const auto& id :idxToRemove)
    {
        trackers.erase(id);
        activeIDs.erase(id);
    }
}

