#ifndef DFCXX_CONSTANT_H
#define DFCXX_CONSTANT_H

#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/varbuilders/builder.h"

namespace dfcxx {

class Kernel;

class Constant {
  friend Kernel;
private:
  Graph &graph;
  GraphHelper helper;
  VarBuilder &varBuilder;
  KernStorage &storage;

  Constant(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
           KernStorage &storage);

public:
  DFVariable var(const DFType &type, int64_t value);

  DFVariable var(const DFType &type, uint64_t value);

  DFVariable var(const DFType &type, double value);
};

} // namespace dfcxx

#endif // DFCXX_CONSTANT_H
