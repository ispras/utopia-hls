#ifndef DFCXX_IR_BUILDER_H
#define DFCXX_IR_BUILDER_H

#include "dfcir/DFCIROperations.h"
#include "dfcxx/IRbuilders/converter.h"
#include "dfcxx/kernel.h"
#include "dfcxx/typedefs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <stack>

namespace dfcxx {

class DFCIRBuilder {
private:
  mlir::MLIRContext ctx;
  const DFLatencyConfig &config;
  mlir::OpBuilder builder;
  std::unordered_map<Node, mlir::Value> map;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  DFCIRTypeConverter conv;

  std::unordered_map<Node, llvm::SmallVector<mlir::Value>> muxMap;

  void buildKernelBody(Graph *graph, mlir::OpBuilder &builder);

  mlir::dfcir::KernelOp buildKernel(Kernel *kern, mlir::OpBuilder &builder);

  mlir::ModuleOp buildModule(Kernel *kern, mlir::OpBuilder &builder);

  std::stack<Node> topSortNodes(Graph *graph);

  void translate(Node node, Graph *graph, mlir::OpBuilder &builder);

public:
  explicit DFCIRBuilder(const DFLatencyConfig &config);

  mlir::ModuleOp buildModule(Kernel *kern);
};

} // namespace dfcxx

#endif // DFCXX_IR_BUILDER_H
