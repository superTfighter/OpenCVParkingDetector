#pragma once
#include "opencv2/opencv.hpp"
#include "ImageFilter.h"
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>


class Detector
{
public:

	Detector();

	virtual void detect() = 0;
	virtual void stop() = 0;

	virtual cv::Mat getDoneImage() = 0;


protected:
	// To have fresh image from framebuffer, running on other thread, important to cleanup
	void runImageCapturing();
	void loadClassNames();

	virtual void loadNet() = 0;

	std::vector<ImageFilter*> filters;
	std::vector<std::string> class_names;
	cv::VideoCapture source;
	int classCount;

	std::vector<cv::String> output_names;
	cv::dnn::dnn4_v20220524::Net net;

	cv::Mat freshImage;
	cv::Mat doneImage;
	bool running;
	bool isReading;
};

