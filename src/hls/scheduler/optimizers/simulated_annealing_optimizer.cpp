#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace eda::hls::scheduler::optimizers {
<<<<<<< HEAD
simulated_annealing_optimizer::simulated_annealing_optimizer(
    float init_temp, float fin_temp, float lim, target tar_fun,
    condition condition_fun, step step_fun, temp temp_fun)
    : temperature(init_temp), init_temp(init_temp), final_temp(fin_temp),
      limitation(lim), target_function(tar_fun),
      condition_function(condition_fun), step_function(step_fun),
      temp_function(temp_fun) {}

void simulated_annealing_optimizer::optimize(std::vector<float> &param) {
  std::vector<float> cur_param = param;
  float prev_f = -1000000.0, cur_f, transition;

  std::ofstream ostrm("annealing.txt");
  int i = 1;
  while (temperature > final_temp && i < 10000) {
    cur_f = target_function(cur_param);
    auto cur_lim = condition_function(cur_param);
    transition = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    auto proba = get_probability(prev_f, cur_f, cur_lim, temperature, param);
    ostrm << "Temperature: " << temperature << std::endl;
    ostrm << "Transition: " << transition << std::endl;
    ostrm << "Probability: " << proba << std::endl;
    ostrm << "Best target: " << prev_f << std::endl;
    ostrm << "Current target: " << cur_f << std::endl;
    ostrm << "Current limitation: " << cur_lim << std::endl;
    ostrm << "Current params: " << cur_param[0] << " " << cur_param[1]
          << std::endl;
    ostrm << "Previous params: " << param[0] << " " << param[1] << std::endl
          << std::endl;
    if (transition < proba) {
      param = cur_param;
      prev_f = cur_f;
    }
    i++;
    step_function(cur_param, param, temperature, init_temp);
    temperature = temp_function(i, init_temp /*temperature*/);
  }
  ostrm.close();
}

float simulated_annealing_optimizer::get_probability(
    float prev_f, float cur_f, float cur_lim, float temp,
    const std::vector<float> &params) {
  /*bool check_limits = true;
  for(const auto& param : params) {
      if(param < 0 || param > 1) {
          check_limits = false;
          break;
      }
  }*/
  //        if(cur_lim < limitation) {
  //            return -1.0;
  //        }
  if (prev_f < cur_f) {
    return 1.0;
  }
  return exp((cur_f - prev_f) / temp);
}
} // namespace eda::hls::scheduler::optimizers
=======
SimulatedAnnealingOptimizer::SimulatedAnnealingOptimizer(
    float initialTemperature, float finalTemperature, target targetFunction, 
    condition conditionFunction, step stepFunction, temperature temperatureFunction) 
  : currentTemperature(initialTemperature), initialTemperature(initialTemperature), 
  finalTemperature(finalTemperature), targetFunction(targetFunction), 
  conditionFunction(conditionFunction), stepFunction(stepFunction), 
  temperatureFunction(temperatureFunction) {}

void SimulatedAnnealingOptimizer::optimize(std::vector<float>& currentValues) {
  std::vector<float> candidateValues = currentValues;
  float currentTarget = -1000000.0, candidateTarget, transitionProbability;
        
  std::ofstream ostrm("annealing.txt");
  int i = 1;
  while(currentTemperature > finalTemperature && i < 10000) {
    candidateTarget = targetFunction(candidateValues);
    auto currentLimit = conditionFunction(candidateValues);
    transitionProbability = 
      static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    auto stepProbability = 
      getProbability(currentTarget, candidateTarget, currentTemperature);
    ostrm << "Temperature: " << currentTemperature << std::endl;
    ostrm << "Transition: " << transitionProbability << std::endl;
    ostrm << "Probability: " << stepProbability << std::endl;
    ostrm << "Best target: " << currentTarget << std::endl;
    ostrm << "Current target: " << candidateTarget << std::endl;
    ostrm << "Current limitation: " << currentLimit << std::endl;
    ostrm << "Current params: " << candidateValues[0] << " " << candidateValues[1] << std::endl;
    ostrm << "Previous params: " << currentValues[0] << " " << currentValues[1] << std::endl << std::endl;
    if(transitionProbability < stepProbability) {
      currentValues = candidateValues;
      currentTarget = candidateTarget;
    }
    i++;
    stepFunction(candidateValues, currentValues, currentTemperature, 
      initialTemperature);
    currentTemperature = temperatureFunction(i, initialTemperature);
  }
  ostrm.close();
}
   
float SimulatedAnnealingOptimizer::getProbability(float currentTarget, 
    float candidateTarget, float temperature) {
  if(candidateTarget > currentTarget) {
    return 1.0;
  }
  return exp((candidateTarget - currentTarget) / temperature);
}

} //namespace eda::hls::scheduler::optimizers
>>>>>>> d433088eec51812ef77758095ee36f81a8554680
