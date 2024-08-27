/*
    Date: 2024.08.09
*/

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

#include "../include/scrfd_detector.h"
// #include "../include/face_detector.h"
// #include "../include/scrfd_onnxrunner.h"
#include "../include/face_tracker.h"
#include "../include/utils.h"

using namespace std;
namespace fs = std::filesystem;

std::mutex mtx;
bool clustering_done = false;
// std::condition_variable cv_cond;
std::vector<std::vector<FaceObject>> initialFaceBoxes;
std::vector<cv::Rect> clusteredRects;
std::unordered_map<int, FaceTracker> faceTrackers;
std::unordered_map<int, cv::Rect> faceTrackerIDs;

cv::Rect clusterFacesForID(const std::vector<cv::Rect>& boxes, int numClusters) {
    cv::Mat samples(boxes.size(), 1, CV_32FC2);

    int w = 0;
    int h = 0;
    for (size_t i = 0; i < boxes.size(); ++i) {
        float centerX = boxes[i].x + (boxes[i].width / 2.0f);
        float centerY = boxes[i].y + (boxes[i].height / 2.0f);
        // printf("boxes[i].x + boxes[i].width / 2.0f : %d %.2f %.2f \n" , boxes[i].x , (boxes[i].width / 2.0f) , centerX);
        // printf("boxes[i].y + boxes[i].height / 2.0f : %d %.2f %.2f \n" , boxes[i].x , (boxes[i].height / 2.0f) , centerY);
        w += (boxes[i].width);
        h += (boxes[i].height);
        samples.at<cv::Point2f>(i, 0) = {centerX,centerY};
    }
    cv::Mat labels, centers;
    cv::kmeans(samples, numClusters, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 200, 0.1), // 
               5, cv::KMEANS_PP_CENTERS, centers);

    // 取聚类中心作为标准框的位置
    // cv::Point center(centers.at<float>(0, 0), centers.at<float>(0, 1));
    int result_x = 0 , result_y = 0;
    for (int i = 0; i < centers.rows; ++i) {
        cv::Point2f center = centers.at<cv::Point2f>(i, 0);
        // std::cout << "Center " << i + 1 << ": (" << center.x << ", " << center.y << ")" << std::endl;
        result_x = center.x;
        result_y = center.y;
    }
    // cv::Point2f center = centers.at<cv::Point2f>(i, 0);

    w /= (boxes.size());
    h /= (boxes.size());
    
    cv::Point2f clusterCenter = centers.at<cv::Point2f>(0, 0);
    return cv::Rect(clusterCenter.x - w / 2 ,clusterCenter.y - h / 2, w, h);
}


int processFile(const std::string& filePath, SCRFD* detector, const int max_side, const fs::path& resultDir , int standard) {
    if (isImageFile(filePath)) { // Process image
        cv::Mat img = cv::imread(filePath);
        std::cout << "[INFO] Processing image: " << filePath << std::endl;
        if (img.empty()) {
            fprintf(stderr, "[ERROR] cv::imread %s failed\n", filePath.c_str());
            return EXIT_FAILURE;
        }

        std::vector<FaceObject> faceobjects;
        float prob_threshold = 0.5f;  // 人脸置信度阈值
        float nms_threshold = 0.4f;   // 非极大值抑制阈值

        detector->detect(img, faceobjects, prob_threshold, nms_threshold);
        detector->draw(img, faceobjects);  // 绘制人脸框

        std::string output_path = (resultDir / (fs::path(filePath).filename().replace_extension(".png"))).string();
        cv::imwrite(output_path, img);

        std::cout << "[INFO] Image saved to: " << output_path << std::endl;
    } else if (isVideoFile(filePath)) { // Process video
        cv::VideoCapture cap(filePath);
        if (!cap.isOpened()) {
            // fprintf(stderr, "[ERROR] cv::VideoCapture %s failed\n", filePath.c_str());
            return EXIT_FAILURE;
        }

        int frame_width =  static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        if (total_frames <= 0) {
            std::cerr << "[ERROR] Total frames not correctly calculated." << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat frame;
        int interval = 20;
        standard = 20;
        int processed_frames = 0;
        const int update_interval = 10; // Update progress bar every 10 frames
        std::string output_path = (resultDir / (fs::path(filePath).filename().replace_extension(".mp4"))).string();
        cv::VideoWriter video(output_path, cv::VideoWriter::fourcc('X', '2', '6', '4'), cap.get(cv::CAP_PROP_FPS), cv::Size(frame_width, frame_height));
        
        int faceID = 0;
        int numClusters = 0;
        float prob_threshold = 0.3f;  // 人脸置信度阈值
        float nms_threshold = 0.3f;   // 非极大值抑制阈值

        MultiObjectTracker tracker; // Initialize MultiObjectTracker
        std::vector<FaceObject> faceobjects;
        std::map<int, std::vector<cv::Rect>> idToBoxes; // 用于存储每个ID对应的检测框
        std::map<int, cv::Rect> idToStandardBox;        // 用于存储每个ID的标准框

        while (cap.read(frame)) {
            if (frame.empty()) 
            {
                // fprintf(stderr, "[ERROR] cv::read frame failed\n");
                break;
            }

            if (processed_frames <= standard)
            {

                detector->detect(frame, faceobjects, prob_threshold , nms_threshold);
                detector->draw(frame, faceobjects);
                
                std::vector<cv::Rect> detected_boxes;
                for (auto& face : faceobjects)
                {
                    detected_boxes.push_back(face.rect);
                }
                std::cout << "detected_boxes.size()" << detected_boxes.size() << std::endl;

            
                if (!detected_boxes.empty())
                {
                    if (tracker.getActiveIDs().empty())
                    {
                        tracker.initialize(detected_boxes);
                    } else {
                        std::cout << "1" << std::endl;
                        tracker.update(detected_boxes);
                    }
                }

                auto tracked_objects = tracker.getTrackedObjects();
                std::cout << "tracked_objects.size() : " << tracked_objects.size() << std::endl;
                for (const auto& [id, rect] : tracked_objects) {
                    cv::rectangle(frame, rect, (255,0,255), 2);
                    std::string label = "ID: " + std::to_string(id); // 显示跟踪器的 ID
                    int baseline = 0;
                    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Point textOrigin(rect.x, rect.y - baseline);
                    cv::putText(frame, label, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1);

                    idToBoxes[id].emplace_back(rect);
                }

                std::cout << "idToBoxes.size() : " << idToBoxes.size() << std::endl;

                initialFaceBoxes.push_back(faceobjects);
                if (processed_frames == standard) 
                {
                    for (const auto& [id , boxes] : idToBoxes)
                    {
                        int numClusters = 1;
                        cv::Rect standardBox = clusterFacesForID(boxes , numClusters);
                        std::cout << "boxes.size() : " << boxes.size() << " - " << standardBox << std::endl;

                        idToStandardBox[id] = standardBox;
                    }
                    clustering_done = true;
                }
            } else {
                if ((processed_frames - standard) % interval == 0)
                {
                    detector->detect(frame, faceobjects, prob_threshold , nms_threshold);
                    detector->draw(frame, faceobjects);
                }
                if (clustering_done)
                {
                    for (const auto& [id , standardBox] : idToStandardBox) {
                        cv::rectangle(frame, standardBox, cv::Scalar(0, 255, 0), 2);
                        std::string label = "Standard ID: " + std::to_string(id);
                        int baseline = 0;
                        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                        cv::Point textOrigin(standardBox.x, standardBox.y - baseline);
                        cv::putText(frame, label, textOrigin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                    }
                }
            }

            video.write(frame);

            // Update and print progress bar
            processed_frames++;
            if (processed_frames % update_interval == 0) {
                float progress = (float)processed_frames / total_frames;
                printProgressBar(progress);
            }
        }

        cap.release();
        video.release();

        std::cout << "[INFO] Video saved to: " << output_path << std::endl;
        std::cout << std::endl; // Move to the next line after the progress bar

        return EXIT_SUCCESS;
    }
} 

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [directory_path max_side] \n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string dirPath = argv[1];
    const int max_side = atoi(argv[2]);
    const int standard = 100;

    std::string param_path = "$YourParamPath";
    param_path = "C:\\Users\\Administrator\\Desktop\\yolo-face-with-landmark-master\\ncnn_project\\models\\assets\\scrfd_10g-opt2.param";
    std::string model_path = "$YourModelPath";
    model_path = "C:\\Users\\Administrator\\Desktop\\yolo-face-with-landmark-master\\ncnn_project\\models\\assets\\scrfd_10g-opt2.bin";

    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "[ERROR] File does not exist: " << dirPath << std::endl;
        return EXIT_FAILURE;
    }
    if (!fs::exists(param_path) || !fs::exists(model_path)) {
        std::cerr << "[ERROR] Model File or Param File does not exist" << std::endl;
        return EXIT_FAILURE;
    }
    
    fs::path resultDir = fs::path(dirPath) / "result";
    if (!fs::exists(resultDir)) {
        fs::create_directory(resultDir); 
    }

    // FaceDetector *detector = new FaceDetector(param_path, model_path);
    SCRFD *detector = new SCRFD();
    detector->load(param_path, model_path, false);

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            processFile(filePath, detector, max_side, resultDir , standard);
        }
    }

    delete detector;

    return EXIT_SUCCESS;
}



