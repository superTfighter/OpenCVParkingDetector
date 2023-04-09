#include "Yolo4Detector.h"

Yolo4Detector::Yolo4Detector() : Detector()
{
	//filters.push_back(new GreenImageFilter());
	this->loadNet();
}

void Yolo4Detector::detect()
{
	cv::Mat frame, blob;
	std::vector<cv::Mat> detections;

	frame = this->freshImage;

	// Applying optional filters
	for(auto filter : filters)
	{
		filter->filter(frame);
	}

	if (frame.empty())
		return;

	// Detecting
	cv::dnn::blobFromImage(frame, blob, BLOB_SCALE_FACTOR, BLOB_SIZE, cv::Scalar(), true, false, CV_32F);

	net.setInput(blob);
	net.forward(detections, output_names);

	std::vector<std::vector<int>> indices(classCount, std::vector<int>(0));
	std::vector<std::vector<cv::Rect>> boxes(classCount);
	std::vector<std::vector<float>> scores(classCount, std::vector<float>(0));

	this->processDetections(detections, scores, boxes, frame);
	
	for (int c = 0; c < classCount; c++)
		cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

	this->drawBoxesOnImage(frame, boxes, indices, scores);

	this->doneImage = frame;

	detections.clear();
}

void Yolo4Detector::stop()
{
	running = false;

	if(!isReading)
		source.release();
	else
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		stop();
	}

}

cv::Mat Yolo4Detector::getDoneImage()
{
	return this->doneImage;
}

void Yolo4Detector::loadNet()
{
	net = cv::dnn::readNetFromDarknet("config_files/yolov4-tiny.cfg", "config_files/yolov4-tiny.weights");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

	output_names = net.getUnconnectedOutLayersNames();
}

void Yolo4Detector::processDetections(std::vector<cv::Mat> detections, std::vector<std::vector<float>>& scores, std::vector<std::vector<cv::Rect>>& boxes, cv::Mat frame)
{
	for (auto& output : detections)
	{
		const auto num_boxes = output.rows;
		for (int i = 0; i < num_boxes; i++)
		{
			for (int c = 0; c < classCount; c++)
			{
				auto confidence = *output.ptr<float>(i, 5 + c);
				if (confidence >= CONFIDENCE_THRESHOLD)
				{
					auto x = output.at<float>(i, 0) * frame.cols;
					auto y = output.at<float>(i, 1) * frame.rows;
					auto width = output.at<float>(i, 2) * frame.cols;
					auto height = output.at<float>(i, 3) * frame.rows;
					cv::Rect rect(x - width / 2, y - height / 2, width, height);

					boxes[c].push_back(rect);
					scores[c].push_back(confidence);
				}
			}
		}
	}

}

void Yolo4Detector::drawBoxesOnImage(cv::Mat& frame, std::vector<std::vector<cv::Rect>> boxes, std::vector<std::vector<int>> indices, std::vector<std::vector<float>> scores)
{
	for (int c = 0; c < classCount; c++)
	{
		for (size_t i = 0; i < indices[c].size(); ++i)
		{
			const cv::Scalar color = { 0,0,255,0 };

			int idx = indices[c][i];
			const cv::Rect& rect = boxes[c][idx];
			cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

			std::ostringstream label_ss;
			label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
			std::string label = label_ss.str();

			int baseline;
			cv::Size label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
			cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
			cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
		}
	}

}


