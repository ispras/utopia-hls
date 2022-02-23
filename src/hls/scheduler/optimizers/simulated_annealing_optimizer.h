#pragma once

#include "hls/scheduler/optimizers/abstract_optimizer.h"

#include <functional>

namespace eda::hls::scheduler::optimizers {

    class simulated_annealing_optimizer {
      public:
        simulated_annealing_optimizer(float init_temp, float fin_temp,
                        std::function<float(const std::vector<float>&)> tar_fun,
                        std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_fun,
                        std::function<float(int, float)> temp_fun);
        simulated_annealing_optimizer(const simulated_annealing_optimizer& optimizer) = default;
        
        void optimize(std::vector<float>& param);
      private:
        float get_probabiliy(const float& prev_f, const float& cur_f, const float& temp);

        float temperature;
        float final_temp;

        std::function<float(const std::vector<float>&)> target_function;
        std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_function;
        std::function<float(int, float)> temp_function;
    };

} // namespace eda::hls::scheduler::optimizers
