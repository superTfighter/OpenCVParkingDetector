#include "CarHaarCascadeDetector.h"

CarHaarCascadeDetector::CarHaarCascadeDetector() : Detector()
{
    this->loadNet();
}

void CarHaarCascadeDetector::detect()
{
	cv::Mat frame, frame_gray;

	std::vector<cv::Rect> detections;

	frame = this->freshImage;

	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

	carCascade.detectMultiScale(frame_gray, detections);

	for(auto &detection : detections)
	{
		cv::rectangle(frame, cv::Point(detection.x, detection.y), cv::Point(detection.x + detection.width, detection.y + detection.height), cv::Scalar(0,0,255,0), 3);
	}

	this->doneImage = frame;
	detections.clear();
}

void CarHaarCascadeDetector::stop()
{
	running = false;

	if (!isReading)
		source.release();
	else
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		stop();
	}

}

cv::Mat CarHaarCascadeDetector::getDoneImage()
{
	return this->doneImage;
}

void CarHaarCascadeDetector::loadNet()
{
    this->carCascade = cv::CascadeClassifier("config_files/car.xml");
}
