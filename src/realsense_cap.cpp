#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp> // For image processing (optional)
#include <opencv2/imgcodecs.hpp>
#include <fstream> // For file operations

int main() {
  // Define pipeline and config objects
  rs2::pipeline pipe;
  rs2::config cfg;

  // Configure depth and color streams (adjust resolution and fps as needed)
  //cfg.enable_stream(rs2::stream::depth, 640, 480, rs2::format::z16, 30);
  //cfg.enable_stream(rs2::stream::color, 640, 480, rs2::format::bgr8, 30);
  //cfg.enable_stream(RS2_STREAM_COLOR);

  // Start the pipeline
  pipe.start();

  // Wait for frames
  rs2::frameset frames = pipe.wait_for_frames();

  // Get depth and color frames (modify if you only need one type)
  //rs2::depth_frame depth_frame = frames.get_depth_frame();
  rs2::video_frame color_frame = frames.get_color_frame();

  // Save color frame as PNG (modify filename and format as needed)
  cv::Mat color_image(cv::Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
  
  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  cv::imwrite("captured_frame.png", color_image, compression_params);

  // Optionally, save depth frame as a custom format (refer to librealsense documentation for depth data format)
  //std::ofstream depth_file("depth_frame.bin", std::ios::binary);
  //depth_file.write(reinterpret_cast<const char*>(depth_frame.get_data()), depth_frame.get_data_size());
  //depth_file.close();

  // Stop the pipeline
  pipe.stop();

  return 0;
}

