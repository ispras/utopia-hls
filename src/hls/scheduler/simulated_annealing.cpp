#include "hls/scheduler/simulated_annealing.h"

#include <cmath>
#include <cstdlib>
#include <stdlib.h> 

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
    
    void simulated_annealing::optimize(std::vector<float>& x) {
        std::ofstream ostrm("annealing_file.txt", std::ios::out);
        std::vector<float> prev_x = x, cur_x = x;
        float prev_f = target_function(x), cur_f, transition;
        srand (1);
        for(auto j = 0; j < x.size(); j++) {
            cur_x[j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10));
        }
        int i = 1;
        ostrm << "Optimizing" << std::endl;
        while(temperature > final_temp && i < 10000) {
            step_function(cur_x, x, temperature);
            cur_f = target_function(cur_x);
            prev_x = cur_x;
            transition = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            if(transition < get_probabiliy(prev_f, cur_f, temperature)) {
                x = cur_x;
                prev_f = cur_f;
            }
            i++;
            temperature = temp_function(i, temperature);
            ostrm << "Temperature: " << temperature << std::endl;
        }
        ostrm << "Number of iterations: " << i << std::endl;
        ostrm << "Result: " << std::endl;
        for(auto i = 0; i < x.size(); i++) {
            ostrm << x[i] << std::endl;
        }
    }

    float simulated_annealing::get_probabiliy(float prev_f, float cur_f, float temp) {
        if(prev_f > cur_f) {
            return 1.0;
        }
        return exp((prev_f - cur_f) / temp);
    }

} // namespace eda::hls::scheduler
