#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int8.h>
#include <queue>

// Uncomment for Optical Flow
// #define TRACK_OPTFLOW

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>

#include "yolo_v2_class.hpp"
#include "Extrapolate_Coordinates.hpp"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace darknet_ros {
  class YoloROSTracker
  {
  public:
    explicit YoloROSTracker(ros::NodeHandle nh);
    ~YoloROSTracker();
  private:

    void initROS();
    void initdarknet();
    void cameraCallback(const sensor_msgs::ImageConstPtr& msg);
    bool readParameters();
    void captureThread();
    void trackThread();
    void darknetThread();
    void publishResult();

    // ROS
    ros::NodeHandle nodeHandle;
    image_transport::ImageTransport imageTransport;
    image_transport::Subscriber imageSubscriber;
    ros::Publisher boundingBoxesPublisher;
    ros::Publisher detectionImagePublisher;

    // Yolo Object Detector API

    Detector *detector;
    float thresh;

    // ROS Parameters
    bool viewImage_;
    bool enableConsoleOutput_;
    int waitKeyDelay_;

    //! Class labels.
    std::vector<std::string> classLabels;


    //! Detected objects.
    darknet_ros_msgs::BoundingBox boundingBox;
    darknet_ros_msgs::BoundingBoxes boundingBoxes;

    // Tracking
    extrapolate_coords_t extrapolate_coords;
#ifdef TRACK_OPTFLOW
    Tracker_optflow tracker_flow;
#endif
    bool extrapolate_flag = false;
    float cur_time_extrapolate = 0, old_time_extrapolate = 0;
    bool show_small_boxes = false;
    std::queue<cv::Mat> track_optflow_queue;
    int passed_flow_frames = 0;

    // Detection
    cv::Mat cap_frame, cur_frame, det_frame;
    cv::Size frame_size;
    std::shared_ptr<image_t> det_image;
    std::vector<bbox_t> result_vec, thread_result_vec;
    std::atomic<bool> consumed;
    bool exit_flag;
    std::atomic<int> fps_det_counter, fps_cap_counter;
    int current_det_fps, current_cap_fps = 0;
    std::thread t_detect, t_cap;
    std::mutex mtx;
    std::condition_variable cv_detected, cv_pre_tracked;
    std::chrono::steady_clock::time_point steady_start, steady_end;

  };
};
