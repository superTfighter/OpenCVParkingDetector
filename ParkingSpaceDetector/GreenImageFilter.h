#pragma once
#include "opencv2/opencv.hpp"
#include "ImageFilter.h"


class GreenImageFilter : public ImageFilter
{

public:
	void filter(cv::Mat& image);

private:



};

