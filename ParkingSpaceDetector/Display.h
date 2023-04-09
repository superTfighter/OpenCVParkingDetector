#pragma once
#include "Yolo4Detector.h"
#include "SFML/Graphics.hpp"
#include "opencv2/opencv.hpp"


class Display
{

public:
	Display(sf::RenderWindow* _window, Detector* _detector);

	void startRendering();

private:

	sf::RenderWindow* window;

	sf::Image image;
	sf::Texture texture;
	cv::Mat cvImage;
	sf::Sprite detectorOutputSprite;

	Detector* detector;

	void handleInput();
	void setRenderSprite();

};

