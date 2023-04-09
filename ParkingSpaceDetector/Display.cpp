#include "Display.h"

Display::Display(sf::RenderWindow* _window, Detector* _detector)
{
	this->window = _window;
	this->detector = _detector;
}

void Display::startRendering()
{

	while (window->isOpen())
	{

		//Run detection
		detector->detect();
		this->setRenderSprite();

		this->handleInput();

		window->clear();

		window->draw(detectorOutputSprite);

		window->display();
	}

	detector->stop(); // Cleanup after yourselfs
}

void Display::handleInput()
{
	// Event processing
	sf::Event event;
	while (window->pollEvent(event))
	{
		// Request for closing the window
		if (event.type == sf::Event::Closed)
			window->close();
	}

}

void Display::setRenderSprite()
{
	detector->getDoneImage().copyTo(cvImage);

	// Convert image to SFML format
	cv::cvtColor(cvImage, cvImage, cv::COLOR_BGR2RGBA);
	image.create(cvImage.cols, cvImage.rows, cvImage.ptr());

	if (!texture.loadFromImage(image))
	{
		//TODO: Refactor this
		std::cout << "Cannot get image" << std::endl;

		throw "Image not found exception";
	}

	detectorOutputSprite.setTexture(texture);
	detectorOutputSprite.setScale(window->getView().getSize().x / detectorOutputSprite.getLocalBounds().width, window->getView().getSize().y / detectorOutputSprite.getLocalBounds().height);
}
