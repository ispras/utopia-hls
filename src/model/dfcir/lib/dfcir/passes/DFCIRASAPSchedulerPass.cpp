//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"
#include "dfcir/passes/DFCIRSchedulingUtils.h"
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
  using SchedNode = utils::SchedNode;
  using SchedChannel = utils::SchedChannel;
  using SchedGraph = utils::SchedGraph;

private:
  std::pair<SchedGraph::Buffers, int32_t> schedule(SchedGraph &graph) {
    SchedGraph::Latencies map;

    int32_t maxLatency = 0;

    for (SchedNode *node : graph.nodes) {
      for (SchedChannel *channel : node->outputs) {
        int32_t latency = map[node] + node->latency + channel->offset;

        if (latency > map[channel->target]) {
          map[channel->target] = latency;
        }

        if (graph.isOutput(channel->target) && latency > maxLatency) {
          maxLatency = latency;
        }
      }
    }

    SchedGraph::Buffers buffers;

    for (SchedNode *node: graph.nodes) {
      for (SchedChannel *channel: node->inputs) {
        int32_t delta = map[channel->target] -
                       (map[channel->source] +
                        channel->source->latency +
                        channel->offset);

        if (delta && !graph.isConstantInput(channel->source)) {
          buffers[channel] = delta;
        }
      }
    }

    return std::make_pair(buffers, maxLatency);
  }

public:
  explicit DFCIRASAPSchedulerPass(const DFCIRASAPSchedulerPassOptions &options)
      : impl::DFCIRASAPSchedulerPassBase<DFCIRASAPSchedulerPass>(options) {}

  void runOnOperation() override {
    // Convert kernel into graph.
    SchedGraph graph;
    graph.constructFrom(llvm::cast<ModuleOp>(getOperation()));

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
    graph.insertBuffers(this->getContext(), buffers);

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
