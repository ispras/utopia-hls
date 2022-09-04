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

std::size_t seed = 828399055;
std::mt19937 generator{seed};

auto conditionStub = [](const std::vector<float> &) -> float { return -1.0; };

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

auto temperatureFunction = [](int i, float temperature) -> float {
  return initialTemperature / i;
};

void init(std::vector<float> &parameterValues) {
  std::cauchy_distribution<double> distribution{0, 1};
  for (std::size_t i = 0; i < parameterValues.size(); i++) {
    parameterValues[i] = distribution(generator);
  }
}

float distance(float x, float y) { return sqrt(pow((x - y), 2)); }

float getMaxDist(const std::vector<float> &first,
                 const std::vector<float> &second) {
  assert(first.size() == second.size());
  assert(first.size() != 0);
  auto maxDist = distance(first[0], second[0]);
  for (std::size_t i = 0; i < first.size(); i++) {
    auto tmp = distance(first[i], second[i]);
    if (maxDist < tmp) {
      maxDist = tmp;
    }
  }
  return maxDist;
}

} // namespace

TEST(SimulatedAnnealing, Sphere) {
  auto sphere = [](std::vector<float> x) -> float {
    float result = 0.0;
    for (std::size_t i = 0; i < x.size(); i++) {
      result += pow(x[i], 2);
    }
    return -1.0 * result;
  };

  std::vector<float> optimizedParameters(5);
  std::vector<float> perciseParameters(5, 0.0);
  init(optimizedParameters);
  eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer test(
      initialTemperature, finalTemperature, sphere, conditionStub, stepFunction,
      temperatureFunction);
  test.optimize(optimizedParameters);
  auto maxDistance = getMaxDist(optimizedParameters, perciseParameters);
  ASSERT_TRUE(maxDistance < 2.5);
}

TEST(SimulatedAnnealing, Rosenbrock) {
  auto rosenbrock = [](std::vector<float> x) -> float {
    float result = 0.0;
    for (std::size_t i = 0; i < x.size() - 1; i++) {
      result += 100 * pow((x[i + 1] - pow(x[i], 2)), 2) + pow((x[i] - 1), 2);
    }
    return -1.0 * result;
  };

  std::vector<float> optimizedParameters(5);
  std::vector<float> perciseParameters(5, 1.0);
  init(optimizedParameters);
  eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer test(
      initialTemperature, finalTemperature, rosenbrock, conditionStub,
      stepFunction, temperatureFunction);
  test.optimize(optimizedParameters);
  auto maxDistance = getMaxDist(optimizedParameters, perciseParameters);
  ASSERT_TRUE(maxDistance < 2.5);
}

TEST(SimulatedAnnealing, Rastrigin) {
  auto rastrigin = [](std::vector<float> x) -> float {
    float result = 10 * x.size();
    for (std::size_t i = 0; i < x.size(); i++) {
      result += (pow(x[i], 2) - 10 * cos(2 * M_PI * x[i]));
    }
    return -1.0 * result;
  };

  std::vector<float> optimizedParameters(5);
  std::vector<float> perciseParameters(5, 0.0);
  init(optimizedParameters);
  eda::hls::scheduler::optimizers::SimulatedAnnealingOptimizer test(
      initialTemperature, finalTemperature, rastrigin, conditionStub,
      stepFunction, temperatureFunction);
  test.optimize(optimizedParameters);
  auto maxDistance = getMaxDist(optimizedParameters, perciseParameters);
  ASSERT_TRUE(maxDistance < 2.5);
}
