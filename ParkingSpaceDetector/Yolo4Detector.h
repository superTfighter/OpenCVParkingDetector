#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include "ImageFilter.h"
#include "GreenImageFilter.h"
#include <memory>
#include "Detector.h"


class Yolo4Detector : public Detector
{

public:
	Yolo4Detector();

	void detect();
	void stop();

	cv::Mat getDoneImage();

private:
	float NMS_THRESHOLD = 0.51;
	float CONFIDENCE_THRESHOLD = 0.51;
	float BLOB_SCALE_FACTOR = 1.0 / 255.0;
	cv::Size BLOB_SIZE = cv::Size(416, 416);


	void processDetections(std::vector<cv::Mat> detections, std::vector<std::vector<float>>& scores, std::vector<std::vector<cv::Rect>>& boxes, cv::Mat frame);
	void drawBoxesOnImage(cv::Mat &frame, std::vector<std::vector<cv::Rect>> boxes, std::vector<std::vector<int>> indices, std::vector<std::vector<float>> scores);

	// Inherited via Detector
	virtual void loadNet() override;
};

