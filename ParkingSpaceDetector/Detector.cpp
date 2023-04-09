#include "Detector.h"

Detector::Detector()
{
	this->loadClassNames();

	this->source = cv::VideoCapture("http://192.168.0.45:8080/video"); // Load IP webcam source

	if (!this->source.isOpened())
	{
		throw "Source cannot be opened!";
	}

	running = true;
	isReading = false;

	std::thread a(&Detector::runImageCapturing, this);

	a.detach();
}

void Detector::runImageCapturing()
{
	while (running)
	{
		isReading = true; // Poormans lock
		source.read(this->freshImage);
		isReading = false;
	}

}

void Detector::loadClassNames()
{
	this->classCount = 0;
	std::ifstream class_file("config_files/classes.txt");

	if (!class_file)
	{
		throw "failed to open classes.txt\n";
	}

	std::string line;
	while (std::getline(class_file, line))
	{
		class_names.push_back(line);

		classCount++;
	}

}
