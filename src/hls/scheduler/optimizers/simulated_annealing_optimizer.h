#pragma once

#include "hls/scheduler/optimizers/abstract_optimizer.h"

#include <functional>

namespace eda::hls::scheduler::optimizers {

    class simulated_annealing_optimizer : public eda::hls::scheduler::optimizers::abstract_optimizer {
      public:
        simulated_annealing_optimizer(float init_temp, float fin_temp, float lim,
                        std::function<float(const std::vector<float>&)> tar_fun,
                        std::function<float(const std::vector<float>&)> condition_fun,
                        std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_fun,
                        std::function<float(int, float)> temp_fun);
        simulated_annealing_optimizer(const simulated_annealing_optimizer& optimizer) = default;
        
        void optimize(std::vector<float>& param) override;
      private:
        float get_probabiliy(const float& prev_f, const float& cur_f, const float& cur_lim, const float& temp, const std::vector<float>& params);

        float temperature;
        float final_temp;
        float limitation;

        std::function<float(const std::vector<float>&)> target_function;
        std::function<float(const std::vector<float>&)> condition_function;
        std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_function;
        std::function<float(int, float)> temp_function;
    };

} // namespace eda::hls::scheduler::optimizers