#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <math.h>
#include <random>

#include "gtest/gtest.h"

namespace {
float initialTemperature = 5000.0;
float finalTemperature = 1.0;

std::random_device numberGenerator{};

std::mt19937 generator{3217459755};
std::mt19937 generatorRos{1800691079};
std::mt19937 generatorRastr{194940063};

auto conditionStub = [](const std::vector<float> &) -> float { return -1.0; };

auto rosenbrock = [](std::vector<float> x) -> float {
  float result = 0.0;
  for (std::size_t i = 0; i < x.size() - 1; i++) {
    result += 100 * pow((x[i + 1] - pow(x[i], 2)), 2) + pow((x[i] - 1), 2);
  }
  return -1.0 * result;
};

auto sphere = [](std::vector<float> x) -> float {
  float result = 0.0;
  for (std::size_t i = 0; i < x.size(); i++) {
    result += pow(x[i], 2);
  }
  return -1.0 * result;
};

auto rastrigin = [](std::vector<float> x) -> float {
  float result = 10 * x.size();
  for (std::size_t i = 0; i < x.size(); i++) {
    result += (pow(x[i], 2) - 10 * cos(2 * M_PI * x[i]));
  }
  return -1.0 * result;
};

auto stepFunction = [](std::vector<float> &currentValues,
                       const std::vector<float> &previousValues,
                       float currentTemperature,
                       float initialTemperature) -> void {
  std::cauchy_distribution<double> distribution{0.0, 1.0};
  for (std::size_t i = 0; i < currentValues.size(); i++) {
    auto diff = distribution(generator);
    currentValues[i] = previousValues[i] + diff;
  }
};

auto stepFunctionRos = [](std::vector<float> &currentValues,
                          const std::vector<float> &previousValues,
                          float currentTemperature,
                          float initialTemperature) -> void {
  std::cauchy_distribution<double> distribution{0.0, 1.0};
  for (std::size_t i = 0; i < currentValues.size(); i++) {
    auto diff = distribution(generatorRos);
    currentValues[i] = previousValues[i] + diff;
  }
};

auto stepFunctionRastr = [](std::vector<float> &currentValues,
                            const std::vector<float> &previousValues,
                            float currentTemperature,
                            float initialTemperature) -> void {
  std::cauchy_distribution<double> distribution{0.0, 1.0};
  for (std::size_t i = 0; i < currentValues.size(); i++) {
    auto diff = distribution(generatorRastr);
    currentValues[i] = previousValues[i] + diff;
  }
};

auto temperatureFunction = [](int i, float temperature) -> float {
  return initialTemperature / i;
};

void init(std::vector<float> &parameterValues) {
  std::cauchy_distribution<double> distribution{0, 1};
  for (std::size_t i = 0; i < parameterValues.size(); i++) {
    parameterValues[i] = distribution(generator);
  }
}

void initRos(std::vector<float> &parameterValues) {
  std::cauchy_distribution<double> distribution{0, 1};
  for (std::size_t i = 0; i < parameterValues.size(); i++) {
    parameterValues[i] = distribution(generatorRos);
  }
}

void initRastr(std::vector<float> &parameterValues) {
  std::cauchy_distribution<double> distribution{0, 1};
  for (std::size_t i = 0; i < parameterValues.size(); i++) {
    parameterValues[i] = distribution(generatorRastr);
  }
}
} // namespace

TEST(SimulatedAnnealing, Sphere) {
  std::vector<float> optimizedParameters(3);
  init(optimizedParameters);
  eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer test(
      initialTemperature, finalTemperature, sphere, conditionStub, stepFunction,
      temperatureFunction);
  test.optimize(optimizedParameters);
  auto functionValue = sphere(optimizedParameters);
  ASSERT_TRUE(abs(functionValue) < 0.25);
}

/*TEST(SimulatedAnnealing, Rosenbrock) {
  std::vector<float> optimizedParameters(3);
  initRos(optimizedParameters);
  eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer test(
      initialTemperature, finalTemperature, rosenbrock, conditionStub,
      stepFunctionRos, temperatureFunction);
  test.optimize(optimizedParameters);
  auto functionValue = rosenbrock(optimizedParameters);
  ASSERT_TRUE(abs(functionValue) < 0.15);
}

TEST(SimulatedAnnealing, Rastrigin) {
  std::vector<float> optimizedParameters(3);
  initRastr(optimizedParameters);
  eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer test(
      initialTemperature, finalTemperature, rastrigin, conditionStub,
      stepFunctionRastr, temperatureFunction);
  test.optimize(optimizedParameters);
  auto functionValue = rastrigin(optimizedParameters);
  ASSERT_TRUE(abs(functionValue) < 2);
}*/
