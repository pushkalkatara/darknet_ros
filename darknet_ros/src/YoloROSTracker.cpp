#include "darknet_ros/YoloROSTracker.hpp"

#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

YoloROSTracker::YoloROSTracker(ros::NodeHandle nh)
  : nodeHandle(nh),
    imageTransport(nodeHandle)
  {
    ROS_INFO("[YoloROSDetector] Node started.");

    if(!readParameters()) ros::requestShutdown();
    initROS();
    initdarknet();
    while(true){
      captureThread();
      trackThread();
      darknetThread();
      publishResult();
      if(!ros::ok()) break;
    }
  }
YoloROSTracker::~YoloROSTracker(){
    exit_flag = true;
    if(t_cap.joinable()) t_cap.join();
    if(t_detect.joinable()) t_detect.join();
}

bool YoloROSTracker::readParameters()
    {
    // Load common parameters.
    nodeHandle.param("image_view/enable_opencv", viewImage_, true);
    nodeHandle.param("image_view/wait_key_delay", waitKeyDelay_, 3);
    nodeHandle.param("image_view/enable_console_output", enableConsoleOutput_, false);

    // Check if Xserver is running on Linux.
    if (XOpenDisplay(NULL)) {
      // Do nothing!
      ROS_INFO("[YoloObjectDetector] Xserver is running.");
    } else {
      ROS_INFO("[YoloObjectDetector] Xserver is not running.");
      viewImage_ = false;
    }
    // Set vector sizes.
    nodeHandle.param("yolo_model/detection_classes/names", classLabels,
                      std::vector<std::string>(0));
    return true;
  }
void YoloROSTracker::initROS()
  {
  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  nodeHandle.param("subscribers/camera_reading/topic", cameraTopicName,
                  std::string("/camera/image_raw"));
  nodeHandle.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle.param("publishers/object_detector/topic", objectDetectorTopicName,
                  std::string("found_object"));
  nodeHandle.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle.param("publishers/bounding_boxes/topic", boundingBoxesTopicName,
                  std::string("bounding_boxes"));
  nodeHandle.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle.param("publishers/detection_image/topic", detectionImageTopicName,
                    std::string("detection_image"));
  nodeHandle.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle.param("publishers/detection_image/latch", detectionImageLatch, true);

  imageSubscriber = imageTransport.subscribe(cameraTopicName, cameraQueueSize,
                                             &YoloROSTracker::cameraCallback, this);

  boundingBoxesPublisher = nodeHandle.advertise<darknet_ros_msgs::BoundingBoxes>(
                                             boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
                                             detectionImagePublisher = nodeHandle.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                             detectionImageQueueSize,
                                             detectionImageLatch);

  detectionImagePublisher = nodeHandle.advertise<sensor_msgs::Image>(detectionImageTopicName,
                                              detectionImageQueueSize,
                                              detectionImageLatch);
}
void YoloROSTracker::initdarknet()
  {
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  nodeHandle.param("yolo_model/threshold/value", thresh, (float)0.3);
  nodeHandle.param("yolo_model/weight_file/name", weightsModel, std::string("yolov3.weights"));
  nodeHandle.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;

  nodeHandle.param("yolo_model/config_file/name", configModel, std::string("yolov3.cfg"));
  nodeHandle.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;

  dataPath = darknetFilePath_;
  dataPath += "/data";

  detector = new Detector(configPath, weightsPath);
  detector->nms = 0.02;
 #ifdef TRACK_OPTFLOW
  detector->wait_stream = true;
 #endif
  passed_flow_frames = 0;
  exit_flag = false;
  consumed = true;
  fps_det_counter = 0;
  fps_cap_counter = 0;
  current_cap_fps = 0;
  current_det_fps = 0;
  }
void YoloROSTracker::darknetThread(){
  if (!t_detect.joinable()){
  t_detect = std::thread([&]() {
    auto current_image = det_image;
    consumed = true;
    ROS_INFO("Started darknet thread");
    while (current_image.use_count() > 0 && !exit_flag) {
           ROS_INFO("Reference Count > 0");
           auto result = detector->detect_resized(*current_image, frame_size.width, frame_size.height, thresh, false);
           ++fps_det_counter;
           std::unique_lock<std::mutex> lock(mtx);
           thread_result_vec = result;
           consumed = true;
           ROS_INFO("Started darknet thread");
           cv_detected.notify_all();
           if (detector->wait_stream) while (consumed && !exit_flag) cv_pre_tracked.wait(lock);
           current_image = det_image;
      }
    });
  }
}
void YoloROSTracker::cameraCallback(const sensor_msgs::ImageConstPtr& msg)
  {
  ROS_DEBUG("[YoloROSDetector] ROS image received.");

  cv_bridge::CvImagePtr cv_ptr;

  try {
    ROS_INFO("Callback Called");
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cap_frame = cv_ptr->image.clone();
    frame_size = cap_frame.size();
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  return;
}
void YoloROSTracker::captureThread(){
  t_cap = std::thread([&]() { ros::spinOnce(); });
  ++cur_time_extrapolate;
  if(t_cap.joinable()){
    t_cap.join();
    ++fps_cap_counter;
    cur_frame = cap_frame.clone();
  }
}
void YoloROSTracker::trackThread(){
  if(consumed){
    ROS_INFO("Inside tracking");
      std::unique_lock<std::mutex> lock(mtx);
      det_image = detector->mat_to_image_resize(cur_frame);
      auto old_result_vec = detector->tracking_id(result_vec);
      auto detected_result_vec = thread_result_vec;
      result_vec = detected_result_vec;
 #ifdef TRACK_OPTFLOW
      if(track_optflow_queue.size() > 0){
        cv::Mat first_frame = track_optflow_queue.front();
        tracker_flow.update_tracking_flow(track_optflow_queue.front(), result_vec);
        while (track_optflow_queue.size() > 1) {
          track_optflow_queue.pop();
          result_vec = tracker_flow.tracking_flow(track_optflow_queue.front(), true);
        }
      track_optflow_queue.pop();
      passed_flow_frames = 0;

      result_vec = detector->tracking_id(result_vec);
      auto tmp_result_vec = detector->tracking_id(detected_result_vec, false);
      extrapolate_coords.new_result(tmp_result_vec, old_time_extrapolate);
      old_time_extrapolate = cur_time_extrapolate;
      }
 #else
      result_vec = detector->tracking_id(result_vec);
      extrapolate_coords.new_result(result_vec, cur_time_extrapolate - 1);
 #endif
    for (auto &i : old_result_vec) {
      auto it = std::find_if(result_vec.begin(), result_vec.end(),
                [&i](bbox_t const& b) { return b.track_id == i.track_id && b.obj_id == i.obj_id; });
      bool track_id_absent = (it == result_vec.end());
      if (track_id_absent)
        if (i.frames_counter-- > 1)
            result_vec.push_back(i);
        else it->frames_counter = std::min((unsigned)3, i.frames_counter + 1);
    }

 #ifdef TRACK_OPTFLOW
    tracker_flow.update_cur_bbox_vec(result_vec);
    result_vec = tracker_flow.tracking_flow(cur_frame, true);
 #endif
    consumed = false;
    ROS_INFO("Consumder Done");
    cv_pre_tracked.notify_all();
  }
}
void YoloROSTracker::publishResult(){
  if (!cur_frame.empty()) {
    steady_end = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(steady_end - steady_start).count() >= 1) {
        current_det_fps = fps_det_counter;
        current_cap_fps = fps_cap_counter;
        steady_start = steady_end;
        fps_det_counter = 0;
        fps_cap_counter = 0;
      }
 #ifdef TRACK_OPTFLOW
    ++passed_flow_frames;
    track_optflow_queue.push(cur_frame.clone());
    result_vec = tracker_flow.tracking_flow(cur_frame);
    extrapolate_coords.update_result(result_vec, cur_time_extrapolate);
 #endif
    auto result_vec_draw = result_vec;
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    for(auto &i : result_vec){
      boundingBox.Class = classLabels[i.obj_id];
      boundingBox.prob = i.prob;
      boundingBox.x = i.x;
      boundingBox.y = i.y;
      boundingBox.w = i.w;
      boundingBox.h = i.h;
      boundingBoxes.bounding_boxes.push_back(boundingBox);
      cv::Scalar color = obj_id_to_color(i.obj_id);
      cv::rectangle(cur_frame, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
      if (classLabels.size() > i.obj_id) {
            std::string obj_name = classLabels[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            cv::rectangle(cur_frame, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
                cv::Point2f(std::min((int)i.x + max_width, cur_frame.cols-1), std::min((int)i.y, cur_frame.rows-1)),
                color, CV_FILLED, 8, 0);
            putText(cur_frame, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
        }
    }

    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps);
        putText(cur_frame, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
    boundingBoxesPublisher.publish(boundingBoxes);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cur_frame).toImageMsg();
    detectionImagePublisher.publish(msg);
    ROS_INFO("Published");

    }
}

}
