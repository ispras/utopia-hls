#ifndef DFCXX_OFFSET_H
#define DFCXX_OFFSET_H

#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/varbuilders/builder.h"

namespace dfcxx {

class Kernel;

class Offset {
  friend Kernel;
private:
  Graph &graph;
  GraphHelper helper;
  VarBuilder &varBuilder;
  KernStorage &storage;

  Offset(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &builder, KernStorage &storage);

public:
  DFVariable operator()(DFVariable &stream, int64_t offset);
};

} // namespace dfcxx

#endif // DFCXX_OFFSET_H