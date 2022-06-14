#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>
#include <cstdlib>
#include <math.h>
#include <random>

#include "gtest/gtest.h"

namespace {
  float init = 1000000000.0;
  float end = 1.0;
  float lim = 0.0;

   std::function<float(const std::vector<float>&)> condition_stub = [](const std::vector<float>&) -> float {
    return -1.0;
   };

  std::function<float(std::vector<float>)> ros_fun = [](std::vector<float> x) -> float  {
    float result = 0.0;
    for(std::size_t i = 0; i < x.size() - 1; i++) {
      result += 100 * pow((x[i+1] - pow(x[i], 2)), 2) + pow((x[i] - 1), 2);
    }
    return result;
  };

  std::function<float(std::vector<float>)> sphere_fun = [](std::vector<float> x) -> float  {
    float result = 0.0;
    for(std::size_t i = 0; i < x.size() - 1; i++) {
      result += pow(x[i], 2);
    }
    return result;
  };

  std::function<float(std::vector<float>)> rastr_fun = [](std::vector<float> x) -> float  {
    float result = 10 * x.size();
    for(std::size_t i = 0; i < x.size() - 1; i++) {
      result += (pow(x[i], 2) - 10 * cos(2 * M_PI * x[i]));
    }
    return result;
  };

  std::function<void(std::vector<float>&, const std::vector<float>&, float)> step_fun = 
            [](std::vector<float>& x, const std::vector<float>& prev, float temp) -> void {
                  srand(8);
                  std::random_device rand_dev{};
                  std::mt19937 gen{rand_dev()};
                  std::normal_distribution<double> distr{0, 1};

                  auto step = distr(gen);

                  for(std::size_t i = 0; i < x.size(); i++) {
                    x[i] += step;
                  }
            };

  std::function<float(int, float)> temp_fun = [](int i, float temp) -> float {
    return temp / log(i + 1);
  };

}

TEST(SimulatedAnnealing, SphereFunct) {
    srand(5);
    std::vector<float> x = {6.0, 6.0, 6.0, 6.0};
    eda::hls::scheduler::optimizers::simulated_annealing_optimizer test(init, end, lim, sphere_fun, condition_stub, step_fun, temp_fun);
    test.optimize(x);
    std::cout << "Sphere: " << x[0] << std::endl;
    ASSERT_TRUE(abs(x[0]) < 0.05);
}

TEST(SimulatedAnnealing, RosenbrockFunct) {
    srand(5);
    std::vector<float> x = {9.0, 9.0, 9.0, 9.0};
    eda::hls::scheduler::optimizers::simulated_annealing_optimizer test(init, end, lim, ros_fun, condition_stub, step_fun, temp_fun);
    test.optimize(x);
    std::cout << "Rosenbroc: " << x[0] << std::endl;
    ASSERT_TRUE(abs(x[0]) < 1.05 && abs(x[0]) > 0.95);
}

TEST(SimulatedAnnealing, RastriginFunct) {
    srand(5);
    std::vector<float> x = {3.0, 3.0, 3.0, 3.0};
    eda::hls::scheduler::optimizers::simulated_annealing_optimizer test(init, end, lim, rastr_fun, condition_stub, step_fun, temp_fun);
    test.optimize(x);
    std::cout << "Rastrigin: " << x[0] << std::endl;
    ASSERT_TRUE(abs(x[0]) < 0.05);
}
