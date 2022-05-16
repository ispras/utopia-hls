#include "hls/scheduler/optimizers/simple_optimizer.h"

namespace eda::hls::scheduler::optimizers {
  simple_optimizer::simple_optimizer(std::function<float(const std::vector<float>&)> tar_fun) 
  : target_function(tar_fun) {}

void simple_optimizer::optimize(std::vector<float>& params) {
  int y1, y2;
  int x2 = params[0];
  y2 = target_function(params);
  int x1 = max_freq - (max_freq - min_freq) / 10;
  target_function(params);
  y1 = cur_area;
  float k = float(y1 - y2) / float(x1 - x2);
  float b = float(y1 - x1 * k);
  params[0] = (limitation - b) / k;
  target_function(params);
  int sign;
  if(cur_area > limitation) {
    sign = -1;
  } else {
    sign = 1;
  }
  int grid = (max_freq - min_freq) / 100;

  // Optimization loop.
  const unsigned N = 5;
  for (unsigned i = 0; i < N; i++) {

    params[0] += sign * grid;
    cur_area = target_function(params);
    // Check the constraints.
    if (cur_area < limitation) {
      break;
    }
  }

}

} // namespace eda::hls::scheduler::optimizers
