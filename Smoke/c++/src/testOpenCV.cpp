#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void drawText(Mat & image);

int main()
{
	Mat image, tmp[4];
    image = imread("../../data/0040.png", IMREAD_UNCHANGED);
	imshow("tmp", image);

  cout << "OpenCV version : " << CV_VERSION << endl;
  cout << "Major version : " << CV_MAJOR_VERSION << endl;
  cout << "Minor version : " << CV_MINOR_VERSION << endl;
  cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;
 

	
	waitKey();   
}


