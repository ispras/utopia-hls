//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#pragma once

#include <vector>
#include <functional>

namespace eda::hls::scheduler {
class simulated_annealing {
  public:
    simulated_annealing(float init_temp, float fin_temp,
                        std::function<float(const std::vector<float>&)> tar_fun,
                        std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_fun,
                        std::function<float(int, float)> temp_fun);
    void optimize(std::vector<float>& param);

  private:
    float get_probabiliy(const float& prev_f, const float& cur_f, const float& temp);

    float temperature;
    float final_temp;

    std::function<float(const std::vector<float>&)> target_function;
    std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_function;
    std::function<float(int, float)> temp_function;
};
} // namespace eda::hls::scheduler
