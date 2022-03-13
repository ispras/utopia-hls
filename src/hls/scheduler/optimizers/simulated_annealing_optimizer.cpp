#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace eda::hls::scheduler::optimizers {
    simulated_annealing_optimizer::simulated_annealing_optimizer(float init_temp, float fin_temp, float lim,
                                            std::function<float(const std::vector<float>&)> tar_fun,
                                            std::function<float(const std::vector<float>&, float)> condition_fun,
                                            std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_fun,
                                            std::function<float(int, float)> temp_fun)
      : temperature(init_temp)
      , final_temp(fin_temp)
      , limitation(lim)
      , target_function(tar_fun)
      , condition_function(condition_fun)
      , step_function(step_fun)
      , temp_function(temp_fun) {}

    void simulated_annealing_optimizer::optimize(std::vector<float>& param) {
        std::vector<float> cur_param = param;
        float prev_f = target_function(param), cur_f, transition;
        srand (1);
        
        std::ofstream ostrm("annealing.txt");
        int i = 1;
        while(temperature > final_temp && i < 10000) {
            step_function(cur_param, param, temperature);
            cur_f = target_function(cur_param);
            auto cur_lim = condition_function(cur_param, cur_f);
            transition = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            ostrm << "Diff freq: " << (prev_f - cur_f) << std::endl;
            ostrm << "Exp freq: " << ((prev_f - cur_f) / temperature) << std::endl;
            if(transition < get_probabiliy(prev_f, cur_f, cur_lim, temperature)) {
                param = cur_param;
                prev_f = cur_f;
            } else {
                cur_f = prev_f;
            }
            i++;
            temperature = temp_function(i, temperature);
        }
        ostrm.close();
    }

    float simulated_annealing_optimizer::get_probabiliy(const float& prev_f, const float& cur_f,
                                                        const float& cur_lim, const float& temp) {
        if((prev_f > cur_f) && (cur_lim <= limitation)) {
            return 1.0;
        }
        return exp((prev_f - cur_f) / temp);
    }
} //namespace eda::hls::scheduler::optimizers
