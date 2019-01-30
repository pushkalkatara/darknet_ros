#include "yolo_v2_class.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/highgui/highgui.hpp>

class track_kalman{
public:
  track_kalman(int _state_size, int _meas_size, int _contr_size);

private:
  cv::KalmanFilter kf;
  int state_size;
  int meas_size;
  int contr_size;

  void set(std::vector<bbox_t> result_vec);

  std::vector<bbox_t> correct(std::vector<bbox_t> result_vec);

  std::vector<bbox_t> predict();
};
