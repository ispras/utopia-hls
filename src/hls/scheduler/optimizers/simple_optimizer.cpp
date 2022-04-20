#include "hls/scheduler/optimizers/simple_optimizer.h"

namespace eda::hls::scheduler::optimizers {
    simple_optimizer::simple_optimizer()
     : abstract_optimizer() {}

     void simple_optimizer::optimize(std::vector<float>& param) {
           /*int y1, y2;
  int x2 = cur_f;
  y2 = indicators.area;
  int x1 = criteria.frequency.max - (criteria.frequency.max - criteria.frequency.min) / 10;
  count_params(model, params, indicators, x1, defaultParams);
  y1 = indicators.area;
  float k = float(y1 - y2) / float(x1 - x2);
  float b = float(y1 - x1 * k);
  cur_f = (criteria.area.max - b) / k;
  count_params(model, params, indicators, cur_f, defaultParams);
  int sign;
  if(indicators.area > criteria.area.max) {
    sign = -1;
  } else {
    sign = 1;
  }
  int grid = (criteria.frequency.max - criteria.frequency.min) / 100;

  // Optimization loop.
  const unsigned N = 5;
  for (unsigned i = 0; i < N; i++) {

    cur_f += sign * grid;
    count_params(model, params, indicators, cur_f, defaultParams);
    // Check the constraints.
    if (criteria.check(indicators)) {
      break;
    }
    // Reset to the initial model state.
    model.undo();
  }*/

     }

} // namespace eda::hls::scheduler::optimizers
