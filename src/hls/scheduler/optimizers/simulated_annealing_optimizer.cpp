#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace eda::hls::scheduler::optimizers {
    simulated_annealing_optimizer::simulated_annealing_optimizer(float init_temp, float fin_temp, float lim,
                                                                 target tar_fun, condition condition_fun, step step_fun, temp temp_fun)
      : temperature(init_temp)
      , init_temp(init_temp)
      , final_temp(fin_temp)
      , limitation(lim)
      , target_function(tar_fun)
      , condition_function(condition_fun)
      , step_function(step_fun)
      , temp_function(temp_fun) {}

    void simulated_annealing_optimizer::optimize(std::vector<float>& param) {
        std::vector<float> cur_param = param;
        float prev_f = 0, cur_f, transition;
        
        std::ofstream ostrm("annealing.txt");
        int i = 1;
        while(temperature > final_temp && i < 10000) {
            cur_f = target_function(cur_param);
            auto cur_lim = condition_function(cur_param);
            transition = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            auto proba = get_probability(prev_f, cur_f, cur_lim, temperature, param);
            ostrm << "Transition: " << transition << std::endl;
            ostrm << "Probability: " << proba << std::endl;
            ostrm << "Current frequency: " << cur_f << std::endl;
            ostrm << "Current limitation: " << cur_lim << std::endl;
            ostrm << "Current params: " << cur_param[0] << " " << cur_param[1] << std::endl;
            ostrm << "Previous params: " << param[0] << " " << param[1] << std::endl << std::endl;
            if(transition < proba) {
                param = cur_param;
                prev_f = cur_f;
            } else {
                cur_param = param;
            }
            i++;
            step_function(cur_param, param, temperature, init_temp);
            temperature = temp_function(i, temperature);
        }
        ostrm.close();
    }

    float simulated_annealing_optimizer::get_probability(float prev_f, float cur_f,
                                                         float cur_lim, float temp,
                                                         const std::vector<float>& params) {
        /*bool check_limits = true;
        for(const auto& param : params) {
            if(param < 0 || param > 1) {
                check_limits = false;
                break;
            }
        }*/
        if(cur_lim < limitation) {
            return -1.0;
        }
        if(prev_f < cur_f) {
            return 1.0;
        }
        return exp((cur_f - prev_f) / temp);
    }
} //namespace eda::hls::scheduler::optimizers
