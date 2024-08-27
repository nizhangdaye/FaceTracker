/*
*/

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <iostream>

struct Configuration
{
    std::string modelpath;
    float conf_threshold = 0.30f;
    float nms_threshold = 0.45f;
    std::string device;
};

#endif // CONFIGURATION_H