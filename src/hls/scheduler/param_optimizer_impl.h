//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

template<typename T>
std::map<std::string, model::Parameters> ParametersOptimizer<T>::optimize(
    const model::Criteria &criteria, model::Model &model, 
    model::Indicators &indicators) const {
  float initialTemperature = 100.0; // TODO: Why this value?
  float finalTemperature = 1.5;

  std::ofstream ostrm("real.txt");

  std::map<std::string, model::Parameters> parameters;

  // Get the main dataflow graph.
  auto *graph = model.main();
  uassert(graph, "Main graph is not found");

  std::random_device numberGenerator{};
  std::mt19937 generator{numberGenerator()};

  std::vector<float> parameterValues, minValues, maxValues;

  auto temperatureFunction = [](int i, float temperature) -> float {
    return temperature / i;
  };

  auto stepFunction =
      [&](std::vector<float> &currentValues,
          const std::vector<float> &previousValues, // TODO: Why denormalized?
          float currentTemperature, float initialTemperature) -> void {
    std::normal_distribution<> distribution{
        0.0, (currentTemperature / initialTemperature)};

    for (std::size_t i = 0; i < currentValues.size(); i++) {
      auto diff = distribution(generator);
      currentValues[i] = std::clamp(previousValues[i] + diff, 0.0, 1.0);
    }
  };

  auto targetFunction = [&](const std::vector<float> &currentValues) -> float {
    std::vector<float> denormalizedValues;
    std::size_t index = 0;
    for (const auto &value : currentValues) {
      denormalizedValues.push_back(
          denormalize(value, minValues[index], maxValues[index]));
      index++;
    }

    estimate(model, parameters, indicators, denormalizedValues);
    model.undo();

    float a = 10000.0 *
              ((float)criteria.area.getMax() - (float)indicators.area) /
              (float)criteria.area.getMax();
    return a;
  };

  auto constraintFunction = [&](const std::vector<float> &parameters) -> float {
    return indicators.freq();
  };

  init(graph, parameters, parameterValues, minValues, maxValues);

  optimizers::SimulatedAnnealingOptimizer optimizer(
      initialTemperature, finalTemperature, targetFunction, constraintFunction,
      stepFunction, temperatureFunction);

  optimizer.optimize(parameterValues);

  auto result = targetFunction(parameterValues);
  auto limitation = constraintFunction(parameterValues);

  ostrm << std::endl << "After optimization" << std::endl;
  ostrm << "Parameters values" << std::endl;
  for (auto val : parameterValues) {
    ostrm << val << " ";
  }
  ostrm << std::endl;

  ostrm << "Freq: " << indicators.freq() << std::endl;
  ostrm << "Perf: " << indicators.perf() << std::endl;
  ostrm << "Ticks: " << indicators.ticks << std::endl;
  ostrm << "Power: " << indicators.power << std::endl;
  ostrm << "Area: " << indicators.area << std::endl;

  ostrm << "Target function: " << result << std::endl;
  ostrm << "Limitation: " << limitation << std::endl;

  ostrm.close();
  return parameters;
}

template<typename T>
void ParametersOptimizer<T>::init(const model::Graph *graph, 
    std::map<std::string, model::Parameters> &parameters, 
    std::vector<float> &parameterValues, std::vector<float> &minValues, 
    std::vector<float> &maxValues) const {

  std::random_device numberGenerator{};
  std::mt19937 generator{numberGenerator()};
  std::normal_distribution<> distribution{0.5, 0.25};

  for (const auto *node : graph->nodes) {
    auto metaElement = node->map;
    parameters.insert({node->name, metaElement->params});

    for (const auto &iter : metaElement->params.getAll()) {
      float value = std::clamp(distribution(generator), 0.0, 1.0);
      parameterValues.push_back(value);
      minValues.push_back(iter.second.getMin());
      maxValues.push_back(iter.second.getMax());
    }
  }
}

template<typename T>
void ParametersOptimizer<T>::estimate(model::Model &model, 
    std::map<std::string, model::Parameters> &parameters, 
    model::Indicators &indicators,
    const std::vector<float> &parameterValues) const {
  std::ofstream ostrm("estimation.txt", std::ios_base::app);
  // Update the values of the parameters & apply to nodes.
  auto *graph = model.main();
  std::size_t index = 0;
  for (auto *node : graph->nodes) {
    auto nodeParams = parameters.find(node->name);
    if (nodeParams == parameters.end()) {
      continue;
    }
    nodeParams->second.setValue("stages", parameterValues[index]);
    mapper::Mapper::get().apply(*node, nodeParams->second);
    index++;
  }

  // Balance flows and align times.
  T::get().balance(model);
  
  // Estimate overall design indicators
  mapper::Mapper::get().estimate(model);
  indicators = model.ind;
}

template<typename T>
double ParametersOptimizer<T>::normalize(double value, double min, double max) const {
  return (value - min) / (max - min);
}

template<typename T>
double ParametersOptimizer<T>::denormalize(double value, double min, double max) const {
  return value * (max - min) + min;
}