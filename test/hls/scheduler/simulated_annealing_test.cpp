#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>
#include <cstdlib>
#include <math.h>
#include <random>

#include "gtest/gtest.h"

namespace {
  float initialTemperature = 1000.0;
  float finalTemperature = 1.0;

  auto conditionStub = [](const std::vector<float>&) -> float {
    return -1.0;
  };

  auto rosenbrock = [](std::vector<float> x) -> float  {
    float result = 0.0;
    for(std::size_t i = 0; i < x.size() - 1; i++) {
      result += 100 * pow((x[i+1] - pow(x[i], 2)), 2) + pow((x[i] - 1), 2);
    }
    return result;
  };

  auto sphere = [](std::vector<float> x) -> float  {
    float result = 0.0;
    for(std::size_t i = 0; i < x.size() - 1; i++) {
      result += pow(x[i], 2);
    }
    return result;
  };

  auto rastrigin = [](std::vector<float> x) -> float  {
    float result = 10 * x.size();
    for(std::size_t i = 0; i < x.size() - 1; i++) {
      result += (pow(x[i], 2) - 10 * cos(2 * M_PI * x[i]));
    }
    return result;
  };

  auto stepFunction = [](std::vector<float>& x, const std::vector<float>& prev, 
      float temperature, float initialTemperature) -> void {
    srand(8);
    std::random_device randomDevice{};
    std::mt19937 generator{randomDevice()};
    std::normal_distribution<double> distribution{0, 1};

    for(std::size_t i = 0; i < x.size(); i++) {
      x[i] += (temperature / initialTemperature) * distribution(generator);
    }
  };

  auto temperatureFunction = [](int i, float temperature) -> float {
    return temperature / i; //log(i + 1);
  };
}

TEST(SimulatedAnnealing, Sphere) {
    srand(5);
    std::vector<float> x = {6.0, 6.0, 6.0, 6.0};
    eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer 
      test(initialTemperature, finalTemperature, sphere, conditionStub, 
      stepFunction, temperatureFunction);
    test.optimize(x);
    std::cout << "Sphere: " << x[0] << std::endl;
    ASSERT_TRUE(abs(x[0]) < 0.05);
}

TEST(SimulatedAnnealing, Rosenbrock) {
    srand(5);
    std::vector<float> x = {9.0, 9.0, 9.0, 9.0};
    eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer 
      test(initialTemperature, finalTemperature, rosenbrock, conditionStub, 
      stepFunction, temperatureFunction);
    test.optimize(x);
    std::cout << "Rosenbrock: " << x[0] << std::endl;
    ASSERT_TRUE(abs(x[0]) < 1.05 && abs(x[0]) > 0.95);
}

TEST(SimulatedAnnealing, Rastrigin) {
    srand(5);
    std::vector<float> x = {3.0, 3.0, 3.0, 3.0};
    eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer 
      test(initialTemperature, finalTemperature, rastrigin, conditionStub, 
      stepFunction, temperatureFunction);
    test.optimize(x);
    std::cout << "Rastrigin: " << x[0] << std::endl;
    ASSERT_TRUE(abs(x[0]) < 0.05);
}
