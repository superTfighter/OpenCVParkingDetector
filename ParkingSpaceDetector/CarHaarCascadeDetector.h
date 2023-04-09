#pragma once
#include "Detector.h"


class CarHaarCascadeDetector : public Detector
{

public:
	CarHaarCascadeDetector();


	// Inherited via Detector
	virtual void detect() override;
	virtual void stop() override;
	virtual cv::Mat getDoneImage() override;


private:
	virtual void loadNet() override;

	cv::CascadeClassifier carCascade;


};

