/*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

cv::Rect clusterFacesForID(const std::vector<cv::Rect>& boxes, int numClusters) {
    cv::Mat samples(boxes.size(), 1, CV_32FC2);

    int w = 0;
    int h = 0;
    for (size_t i = 0; i < boxes.size(); ++i) {
        float centerX = boxes[i].x + (boxes[i].width / 2.0f);
        float centerY = boxes[i].y + (boxes[i].height / 2.0f);
        w += (boxes[i].width);
        h += (boxes[i].height);
        samples.at<cv::Point2f>(i, 0) = {centerX,centerY};
    }
    cv::Mat labels, centers;
    cv::kmeans(samples, numClusters, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 230, 0.1), // 
               5, cv::KMEANS_PP_CENTERS, centers);

    // 取聚类中心作为标准框的位置
    // cv::Point center(centers.at<float>(0, 0), centers.at<float>(0, 1));
    int result_x = 0 , result_y = 0;
    for (int i = 0; i < centers.rows; ++i) {
        cv::Point2f center = centers.at<cv::Point2f>(i, 0);
        result_x = center.x;
        result_y = center.y;
    }
    // cv::Point2f center = centers.at<cv::Point2f>(i, 0);

    w /= (boxes.size());
    h /= (boxes.size());
    
    cv::Point2f clusterCenter = centers.at<cv::Point2f>(0, 0);
    return cv::Rect(clusterCenter.x - w / 2 ,clusterCenter.y - h / 2, w, h);
}