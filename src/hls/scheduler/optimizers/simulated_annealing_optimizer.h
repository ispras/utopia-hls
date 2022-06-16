#pragma once

#include "hls/scheduler/optimizers/abstract_optimizer.h"

#include <functional>

namespace eda::hls::scheduler::optimizers {

class simulated_annealing_optimizer : public eda::hls::scheduler::optimizers::abstract_optimizer {
public:
  using target = std::function<float(const std::vector<float>&)>;
  using condition = std::function<float(const std::vector<float>&)>;
  using step = std::function<void(std::vector<float>&, const std::vector<float>&, float, float)>;
  using temp = std::function<float(int, float)>;
 
  simulated_annealing_optimizer(float init_temp, float fin_temp, float lim,
                                target tar_fun, condition condition_fun, step step_fun, temp temp_fun);
  simulated_annealing_optimizer(const simulated_annealing_optimizer& optimizer) = default;
  
  void optimize(std::vector<float>& param) override;

private:
  float get_probability(float prev_f, float cur_f, float cur_lim, float temp, const std::vector<float> &params);

  float temperature;
  float init_temp;
  float final_temp;
  float limitation;

  target target_function;
  condition condition_function;
  step step_function;
  temp temp_function;
};

} // namespace eda::hls::scheduler::optimizers
