#pragma once
#include "Detector.h"
#include <string>
#include <format>


struct Detection
{
	int class_id;
	float confidence;
	cv::Rect box;
};

//TODO: Refactor
class YoloV5Detector : public Detector
{

public:
	YoloV5Detector();

	virtual void detect() override;
	virtual void stop() override;
	virtual cv::Mat getDoneImage() override;

private:
	const float NMS_THRESHOLD = 0.4;
	const float CONFIDENCE_THRESHOLD = 0.4;
	const float SCORE_THRESHOLD = 0.5;
	const float BLOB_SCALE_FACTOR = 1.0 / 255.0;
	const cv::Size BLOB_SIZE = cv::Size(640, 640); //Was 640

	// Standard values for 640x640 model size
	const int DIMENSIONS = 85;
	const int DATA_ROWS = 25200;


	virtual void loadNet() override;

	cv::Mat postProcess(cv::Mat input, std::vector<cv::Mat>& detections);
};

