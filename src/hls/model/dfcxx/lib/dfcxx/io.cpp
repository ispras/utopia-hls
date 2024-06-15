#include "dfcxx/io.h"

namespace dfcxx {

using IODirection = dfcxx::IODirection;

IO::IO(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
       KernStorage &storage) : graph(graph), helper(graph,
                                                    typeBuilder,
                                                    varBuilder,
                                                    storage),
       varBuilder(varBuilder), storage(storage) {}

DFVariable IO::input(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildStream(name, IODirection::INPUT, helper, type);
  storage.addVariable(var);
  graph.addNode(var, OpType::IN, NodeData{});
  return var;
}

DFVariable IO::inputScalar(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildScalar(name, IODirection::INPUT, helper, type);
  storage.addVariable(var);
  graph.addNode(var, OpType::IN, NodeData{});
  return var;
}

DFVariable IO::output(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildStream(name, IODirection::OUTPUT, helper, type);
  storage.addVariable(var);
  graph.addNode(var, OpType::OUT, NodeData{});
  return var;
}

DFVariable IO::outputScalar(const std::string &name, const DFType &type) {
  DFVariable var = varBuilder.buildScalar(name, IODirection::OUTPUT, helper, type);
  storage.addVariable(var);
  graph.addNode(var, OpType::OUT, NodeData{});
  return var;
}

} // namespace dfcxx