#pragma once
#include "opencv2/opencv.hpp"

class ImageFilter
{
public:
	virtual void filter(cv::Mat &image) = 0;


};

