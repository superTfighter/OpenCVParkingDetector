#include "YoloV5Detector.h"

YoloV5Detector::YoloV5Detector() : Detector()
{
	this->loadNet();
}

void YoloV5Detector::detect()
{
	cv::Mat input_image, blob;
	std::vector<cv::Mat> detections;

	input_image = this->freshImage;

	cv::resize(input_image,input_image, BLOB_SIZE);

	cv::dnn::blobFromImage(input_image, blob, BLOB_SCALE_FACTOR, BLOB_SIZE, cv::Scalar(), true, false);

	net.setInput(blob);
	net.forward(detections, output_names);

	this->doneImage = this->postProcess(input_image, detections);

	detections.clear();
}

void YoloV5Detector::stop()
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

cv::Mat YoloV5Detector::getDoneImage()
{
	return this->doneImage;
}

void YoloV5Detector::loadNet()
{
	net = cv::dnn::readNetFromONNX("config_files/yolov5s.onnx");

	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);

	output_names = net.getUnconnectedOutLayersNames();
}

cv::Mat YoloV5Detector::postProcess(cv::Mat input, std::vector<cv::Mat>& detections)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	// Resizing factor.
	float x_factor = input.cols / BLOB_SIZE.width;
	float y_factor = input.rows / BLOB_SIZE.height;

	float* data = (float*)detections[0].data;

	for (int i = 0; i < DATA_ROWS; ++i)
	{
		float confidence = data[4];

		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float* classes_scores = data + 5;
			// Create a 1x85 Mat and store class scores of 80 classes.
			cv::Mat scores(1, this->classCount, CV_32FC1, classes_scores);
			// Perform minMaxLoc and acquire the index of best class  score.
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			// Continue if the class score is above the threshold.
			if (max_class_score > SCORE_THRESHOLD)
			{
				// Store class ID and confidence in the pre-defined respective vectors.
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);
				// Center.
				float cx = data[0];
				float cy = data[1];
				// Box dimension.
				float w = data[2];
				float h = data[3];
				// Bounding box coordinates.
				int left = int((cx - 0.5 * w) * x_factor);
				int top = int((cy - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				// Store good detections in the boxes vector.
				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
		// Jump to the next row.
		data += DIMENSIONS;
	}

	// Perform Non-Maximum Suppression and draw predictions.
	std::vector<int> indices;

	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		int left = box.x;
		int top = box.y;
		int width = box.width;
		int height = box.height;
		// Draw bounding box.
		cv::rectangle(input, cv::Point(left, top), cv::Point(left + width, top + height), {0,0,255,0}, 3);

		std::ostringstream label_ss;
		label_ss << class_names[class_ids[idx]] << ": " << std::fixed << std::setprecision(2) << confidences[idx];
		std::string label = label_ss.str();

		int baseline;
		cv::Size label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
		cv::rectangle(input, cv::Point(left, top - label_bg_sz.height - baseline - 10), cv::Point(left + label_bg_sz.width, top), {0,0,255,0}, cv::FILLED);
		cv::putText(input, label.c_str(), cv::Point(left, top - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
	}

	cv::resize(input, input, cv::Size(1920, 1080));

	return input;
}

