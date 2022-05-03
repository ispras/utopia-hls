#pragma once

#include "hls/scheduler/optimizers/abstract_optimizer.h"

namespace eda::hls::scheduler::optimizers {

    class simple_optimizer : public abstract_optimizer {
      public:
        simple_optimizer();
        simple_optimizer(const simple_optimizer& optimizer) = default;
        void optimize(std::vector<float>& param) override;
    };

} // namespace eda::hls::scheduler::optimizers
