/*
	Last Change : 2024.09.03
*/

#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "scrfd_detector.h"


cv::Scalar getRandomColor() {
    // Initialize random seed if not already initialized
    static bool seedInitialized = false;
    if (!seedInitialized) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seedInitialized = true;
    }
    
    // Generate random color
    return cv::Scalar(
        std::rand() % 256, // Blue
        std::rand() % 256, // Green
        std::rand() % 256  // Red
    );
}

bool isImageFile(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp");
}

bool isVideoFile(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "avi" || ext == "mp4" || ext == "mov" || ext == "mkv");
}

void printProgressBar(float progress) {
    int barWidth = 70;
    std::cout << "[";
    int pos = static_cast<int>(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

std::vector<cv::Rect> clusterFaces(const std::vector<std::vector<FaceObject>>& faceBoxes, int numClusters) {
    std::vector<cv::Rect> clusteredRects(numClusters);
    if (faceBoxes.empty() || numClusters == 0) {
        return clusteredRects;
    }

    cv::Mat samples(faceBoxes.size(), 2, CV_32F);
    for (size_t i = 0; i < faceBoxes.size(); ++i) {
        for (size_t j = 0; j < faceBoxes[i].size(); ++j) {
            float centerX = (faceBoxes[i][j].rect.x + faceBoxes[i][j].rect.width) / 2.0f;
            float centerY = (faceBoxes[i][j].rect.y + faceBoxes[i][j].rect.height) / 2.0f;
            samples.at<float>(i, 0) = centerX;
            samples.at<float>(i, 1) = centerY;
        }
    }

    samples.convertTo(samples, CV_32F); // 确保数据类型为CV_32F
    if (samples.dims <= 2 && samples.type() == CV_32F && numClusters > 0) {
        // 调用kmeans进行聚类
        cv::Mat labels, centers;
        cv::kmeans(samples , numClusters, labels,
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 1),
                   5, cv::KMEANS_PP_CENTERS, centers);
        
        // 根据聚类结果计算每个聚类的平均位置
        std::vector<int> clusterCounts(numClusters, 0);
        for (int i = 0; i < labels.rows; ++i) {
            int clusterIdx = labels.at<int>(i);
            cv::Point center(centers.at<float>(clusterIdx, 0), centers.at<float>(clusterIdx, 1));
            int w = faceBoxes[i][0].rect.width;
            int h = faceBoxes[i][0].rect.height;

            clusteredRects[clusterIdx] = cv::Rect(center.x - w / 2, center.y - h / 2, w, h);
            clusterCounts[clusterIdx]++;
        }
    } else {
        std::cerr << "K-means input data is not valid." << std::endl;
    }

    return clusteredRects;
}

#endif