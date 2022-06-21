#pragma once

// C++ STL
#include <vector>

namespace eda::hls::scheduler::optimizers {

class AbstractOptimizer {
  public:
    AbstractOptimizer();
    virtual void optimize(std::vector<float>& parameterValues) = 0;
};

} // namespace eda::hls::scheduler::optimizers
