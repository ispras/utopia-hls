#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>

namespace eda::hls::scheduler::optimizers {
    simulated_annealing_optimizer::simulated_annealing_optimizer(float init_temp, float fin_temp,
                                            std::function<float(const std::vector<float>&)> tar_fun,
                                            std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_fun,
                                            std::function<float(int, float)> temp_fun)
      : temperature(init_temp)
      , final_temp(fin_temp)
      , target_function(tar_fun)
      , step_function(step_fun)
      , temp_function(temp_fun) {}

    void simulated_annealing_optimizer::optimize(std::vector<float>& param) {
        std::vector<float> cur_param = param;
        float prev_f = target_function(param), cur_f, transition;
        srand (1);
        

        int i = 1;
        while(temperature > final_temp && i < 10000) {
            step_function(cur_param, param, temperature);
            cur_f = target_function(cur_param);
            transition = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if(transition < get_probabiliy(prev_f, cur_f, temperature)) {
                param = cur_param;
                prev_f = cur_f;
            } else {
                cur_f = prev_f;
            }
            i++;
            temperature = temp_function(i, temperature);
        }
    }

    float simulated_annealing_optimizer::get_probabiliy(const float& prev_f, const float& cur_f, const float& temp) {
        if(prev_f > cur_f) {
            return 1.0;
        }
        return exp((prev_f - cur_f) / temp);
    }
} //namespace eda::hls::scheduler::optimizers
