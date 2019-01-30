#include <darknet_ros/YoloROSTracker.hpp>
#include <ros/ros.h>
#include <signal.h>

sig_atomic_t volatile done = 0;

void mySigintHandler(int sig)
{
  ros::shutdown();
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "darknet_ros");
  ros::NodeHandle nodeHandle("~");
  darknet_ros::YoloROSTracker yoloROSTracker(nodeHandle);
  signal(SIGINT, mySigintHandler);
  return 0;
}
