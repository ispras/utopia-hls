#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace eda::hls::scheduler::optimizers {
    simulated_annealing_optimizer::simulated_annealing_optimizer(float init_temp, float fin_temp, float lim,
                                            std::function<float(const std::vector<float>&)> tar_fun,
                                            std::function<float(const std::vector<float>&)> condition_fun,
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
        
        std::ofstream ostrm("annealing.txt");
        int i = 1;
        while(temperature > final_temp && i < 10000) {
            step_function(cur_param, param, temperature);
            cur_f = target_function(cur_param);
            auto cur_lim = condition_function(cur_param);
            transition = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            auto proba = get_probabiliy(prev_f, cur_f, cur_lim, temperature, param);
            ostrm << "Transition: " << transition << std::endl;
            ostrm << "Probability: " << proba << std::endl;
            ostrm << "Current params: " << cur_param[0] << " " << cur_param[1] << std::endl;
            ostrm << "Previous params: " << param[0] << " " << param[1] << std::endl << std::endl;
            if(transition < proba) {
                param = cur_param;
                prev_f = cur_f;
            } else {
                cur_param = param;
                cur_f = prev_f;
            }
            i++;
            temperature = temp_function(i, temperature);
        }
        ostrm.close();
    }

    float simulated_annealing_optimizer::get_probabiliy(const float& prev_f, const float& cur_f,
                                                        const float& cur_lim, const float& temp,
                                                        const std::vector<float>& params) {
        auto compar = (cur_lim > limitation);
        std::ofstream ostrm("probability.txt", std::ios_base::app);
        ostrm << "Current limitation: " << cur_lim << std::endl;
        ostrm << "Limitation: " << limitation << std::endl;
        ostrm << "Cur lim > lim: " << compar << std::endl;
        
        bool check_limits = true;
        if((prev_f < cur_f) && (cur_lim <= limitation) && check_limits) {
            ostrm << "Need to transit" << std::endl << std::endl;
            ostrm.close();
            return 1.0;
        }
        if((cur_lim > limitation) || !check_limits) {
            ostrm << "Transition forbiden to transit" << std::endl << std::endl;
            ostrm.close();
            return -1.0;
        }
        return exp((prev_f - cur_f) / temp);
    }
} //namespace eda::hls::scheduler::optimizers