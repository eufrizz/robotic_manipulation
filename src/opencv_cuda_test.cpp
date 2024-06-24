#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
  // Rest of your code here
	#ifdef HAVE_OPENCV_CUDA
	  std::cout << "OpenCV is built with CUDA support!" << std::endl;
	#else
	  std::cout << "OpenCV is not built with CUDA support." << std::endl;
	#endif
  return 0;
}
