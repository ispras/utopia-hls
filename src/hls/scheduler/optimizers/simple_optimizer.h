#pragma once

#include "hls/scheduler/optimizers/abstract_optimizer.h"

#include <functional>
#include <vector>

namespace eda::hls::scheduler::optimizers {

    class simple_optimizer : public eda::hls::scheduler::optimizers::abstract_optimizer {
      public:
        simple_optimizer(std::function<float(const std::vector<float>&)> tar_fun);
        simple_optimizer(const simple_optimizer& optimizer) = default;

        void optimize(std::vector<float>& params) override;
      private:
        std::function<float(const std::vector<float>&)> target_function;

        float max_freq;
        float min_freq;
        float limitation;
        float cur_area;
    };

} // namespace eda::hls::scheduler::optimizers
