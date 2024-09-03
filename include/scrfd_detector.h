/*

*/

#ifndef SCRFD_H
#define SCRFD_H


#include <net.h>
#include <opencv2/core/core.hpp>

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
    bool match = false;
};

class SCRFD
{
public:
    int load(std::string parampath , std::string modelpath , bool use_gpu);

    int detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, \
        float prob_threshold = 0.5f, float nms_threshold = 0.45f);

    int draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects);

private:
    ncnn::Net scrfd;
    bool has_kps;
};

#endif // SCRFD_H