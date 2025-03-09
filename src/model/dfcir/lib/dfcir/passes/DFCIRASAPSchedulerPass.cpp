//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include <queue>
#include <utility>
// TODO: Replace with a normal logger.
// Issue #13 (https://github.com/ispras/utopia-hls/issues/13).
#include <iostream>

namespace mlir::dfcir {
#define GEN_PASS_DECL_DFCIRASAPSCHEDULERPASS
#define GEN_PASS_DEF_DFCIRASAPSCHEDULERPASS

#include "dfcir/passes/DFCIRPasses.h.inc"

class DFCIRASAPSchedulerPass
    : public impl::DFCIRASAPSchedulerPassBase<DFCIRASAPSchedulerPass> {
  using Node = utils::Node;
  using Channel = utils::Channel;
  using Graph = utils::Graph;

private:
  std::pair<Buffers, int32_t> schedule(Graph &graph) {
    Latencies map;
    std::vector<Node *> sorted = topSortNodes(graph);

    for (Node *node : sorted) {
      for (Channel *channel : graph.outputs.at(node)) {
        int32_t latency = map[node] + node->latency + channel->offset;

        if (latency > map[channel->target]) {
          map[channel->target] = latency;
        }
      }
    }

    Buffers buffers;

    for (Node *node: graph.nodes) {
      for (Channel *channel: graph.inputs[node]) {
        int32_t delta = map[channel->target] -
                       (map[channel->source] +
                        channel->source->latency +
                        channel->offset);

        using mlir::dfcir::utils::hasConstantInput;
        if (delta && !hasConstantInput(channel->source->op)) {
          buffers[channel] = delta;
        }
      }
    }

    int32_t maxOutLatency = calculateOverallLatency(graph, buffers, &map);

    return std::make_pair(buffers, maxOutLatency);
  }

public:
  explicit DFCIRASAPSchedulerPass(const DFCIRASAPSchedulerPassOptions &options)
      : impl::DFCIRASAPSchedulerPassBase<DFCIRASAPSchedulerPass>(options) {}

  void runOnOperation() override {
    // Convert kernel into graph.
    Graph graph(llvm::cast<ModuleOp>(getOperation()));

    // Apply latency config to the graph.
    graph.applyConfig(*latencyConfig);

    // Execute scheduler.
    auto [buffers, latency] = schedule(graph);
    latencyStatistic = latency;

    // Check whether the scheduling finished successfully.
    if (latency < 0) {
      std::cout << "Scheduling failed." << std::endl;
      return;
    } else {
      std::cout << "Top-level kernel overall latency: " << latency << std::endl;
    }

    // Insert buffers.
    mlir::dfcir::utils::insertBuffers(
        this->getContext(),
        buffers,
        graph.connectionMap
    );

    // Erase old "dfcir.offset" operations.
    mlir::dfcir::utils::eraseOffsets(getOperation());
  }
};

std::unique_ptr<mlir::Pass> createDFCIRASAPSchedulerPass(LatencyConfig *config) {
  DFCIRASAPSchedulerPassOptions options;
  options.latencyConfig = config;
  return std::make_unique<DFCIRASAPSchedulerPass>(options);
}

} // namespace mlir::dfcir
