#pragma once

// C++ STL
#include <vector>

namespace eda::hls::scheduler::optimizers {

class abstract_optimizer {
public:
  abstract_optimizer();
  virtual void optimize(std::vector<float> &param) = 0;
};

} // namespace eda::hls::scheduler::optimizers
