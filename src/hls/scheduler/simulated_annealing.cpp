#include "hls/scheduler/simulated_annealing.h"

#include <cmath>
#include <cstdlib>
#include <stdlib.h> 
#include <ctime>

#include <iostream>
#include <fstream>

namespace eda::hls::scheduler {
    simulated_annealing::simulated_annealing(float init_temp, float fin_temp,
                                            std::function<float(std::vector<float>)> tar_fun,
                                            std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_fun,
                                            std::function<float(int, float)> temp_fun)
      : temperature(init_temp)
      , final_temp(fin_temp)
      , target_function(tar_fun)
      , step_function(step_fun)
      , temp_function(temp_fun) {}
    
    void simulated_annealing::optimize(std::vector<float>& param) {
        std::ofstream ostrm("annealing_file.txt", std::ios::out);
        auto begin_time = std::clock();
        std::vector<float> prev_param = param, cur_param = param;
        float prev_f = target_function(param), cur_f, transition;
        srand (1);
        for(auto j = 0; j < param.size(); j++) {
            cur_param[j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
        }
        int i = 1;
        ostrm << "Optimizing" << std::endl;
        while(temperature > final_temp && i < 10000) {
            step_function(cur_param, param, temperature);
            cur_f = target_function(cur_param);
            prev_param = cur_param;
            transition = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if(transition < get_probabiliy(prev_f, cur_f, temperature)) {
                param = cur_param;
                prev_f = cur_f;
            }
            i++;
            temperature = temp_function(i, temperature);
            ostrm << "Temperature: " << temperature << std::endl;
        }
        auto end_time = std::clock();
        ostrm << "Number of iterations: " << i << std::endl;
        ostrm << "Result: " << std::endl;
        ostrm << param[0] << std::endl;
        ostrm << "Time: " << 1000.0 * (end_time - begin_time) / CLOCKS_PER_SEC << std::endl;
    }

    float simulated_annealing::get_probabiliy(float prev_f, float cur_f, float temp) {
        if(prev_f > cur_f) {
            return 1.0;
        }
        return exp((prev_f - cur_f) / temp);
    }

} // namespace eda::hls::scheduler
