#ifndef DFCXX_CONTROL_H
#define DFCXX_CONTROL_H

#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/varbuilders/builder.h"

#include <initializer_list>

namespace dfcxx {

class Kernel;

class Control {
  friend Kernel;
private:
  Graph &graph;
  GraphHelper helper;
  VarBuilder &varBuilder;
  KernStorage &storage;

  Control(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
          KernStorage &storage);

public:
  DFVariable mux(DFVariable ctrl, std::initializer_list<DFVariable> args);
};

} // namespace dfcxx

#endif // DFCXX_CONTROL_H
