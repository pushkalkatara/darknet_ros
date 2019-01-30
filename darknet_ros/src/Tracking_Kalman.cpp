#include "Tracking_Kalman.hpp"

track_kalman::track_kalman(int _state_size, int _meas_size, int _contr_size)
  : state_size(_state_size),
    meas_size(_meas_size),
    contr_size(_contr_size)
  {
    kf.init(state_size, meas_size, contr_size, CV_32F);

    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-5));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1e-2));
    cv::setIdentity(kf.transitionMatrix);
  }

void track_kalman::set(std::vector<bbox_t> result_vec)
{
  for (size_t i = 0; i < result_vec.size() && i < state_size*2; ++i) {
      kf.statePost.at<float>(i * 2 + 0) = result_vec[i].x;
      kf.statePost.at<float>(i * 2 + 1) = result_vec[i].y;
  }
}

std::vector<bbox_t> track_kalman::correct(std::vector<bbox_t> result_vec)
{
  cv::Mat measurement(meas_size, 1, CV_32F);
  for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
      measurement.at<float>(i * 2 + 0) = result_vec[i].x;
      measurement.at<float>(i * 2 + 1) = result_vec[i].y;
  }
  cv::Mat estimated = kf.correct(measurement);
  for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
      result_vec[i].x = estimated.at<float>(i * 2 + 0);
      result_vec[i].y = estimated.at<float>(i * 2 + 1);
  }
  return result_vec;
}

std::vector<bbox_t> track_kalman::predict()
{
  std::vector<bbox_t> result_vec;
  cv::Mat control;
  cv::Mat prediction = kf.predict(control);
  for (size_t i = 0; i < prediction.rows && i < state_size * 2; ++i) {
      result_vec[i].x = prediction.at<float>(i * 2 + 0);
      result_vec[i].y = prediction.at<float>(i * 2 + 1);
  }
  return result_vec;
}
