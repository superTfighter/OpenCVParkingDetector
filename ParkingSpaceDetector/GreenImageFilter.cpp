#include "GreenImageFilter.h"

void GreenImageFilter::filter(cv::Mat& image)
{
	cv::Mat copyImage;

	image.copyTo(copyImage);

	// Use HSV color to threshold the image
	cv::Mat3b hsv;
	cvtColor(copyImage, hsv, cv::COLOR_BGR2HSV);

	cv::Mat1b res;
	inRange(hsv, cv::Scalar(100, 80, 100), cv::Scalar(120, 255, 255), res);

	res = ~res;

	// Apply morphology 
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	morphologyEx(res, res, cv::MORPH_ERODE, element, cv::Point(-1, -1), 2);
	morphologyEx(res, res, cv::MORPH_OPEN, element);

	// Blending
	cv::Mat3b green(res.size(), cv::Vec3b(0, 0, 0));
	for (int r = 0; r < res.rows; ++r) {
		for (int c = 0; c < res.cols; ++c) {
			if (res(r, c)) { green(r, c)[1] = uchar(255); }
		}
	}

	cv::imshow("output", res);

	cv::waitKey();

	res.copyTo(image);
}
