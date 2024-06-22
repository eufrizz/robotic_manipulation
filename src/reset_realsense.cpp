
#include <librealsense2/rs.hpp>

int main()
{

  rs2::context ctx;
  rs2::device dev = ctx.query_devices().front(); // Reset the first device
  dev.hardware_reset();
  //rs2::device_hub hub(ctx);
  //dev = hub.wait_for_device();
}
