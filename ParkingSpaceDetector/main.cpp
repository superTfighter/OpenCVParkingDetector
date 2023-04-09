//#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include "Yolo4Detector.h"
#include <SFML/Graphics.hpp>
#include "Display.h"
#include "YoloV5Detector.h"
#include "CarHaarCascadeDetector.h"

int main(int argc, char** argv)
{
	Yolo4Detector detector; // This is the faster model 
	
	//YoloV5Detector detector;

	sf::RenderWindow window(sf::VideoMode(1280, 720), "Parking detector!");

	Display display(&window,&detector);

	display.startRendering();
	
	return 0;
}