//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"
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

#include "dfcir/conversions/DFCIRPasses.h.inc"

typedef std::unordered_map<mlir::dfcir::utils::Node *, int32_t> Latencies;

class DFCIRASAPSchedulerPass
    : public impl::DFCIRASAPSchedulerPassBase<DFCIRASAPSchedulerPass> {
  using Node = utils::Node;
  using Channel = utils::Channel;
  using Graph = utils::Graph;

  class ChannelComp final {
  private:
    Latencies &map;

  public:
    explicit ChannelComp(Latencies &map) : map(map) {}

    bool operator()(Channel *lhs, const Channel *rhs) const {
      return map[lhs->source] + lhs->source->latency + lhs->offset <
             map[rhs->source] + rhs->source->latency + rhs->offset;
    }
  };

private:
  std::pair<Buffers, int32_t> schedule(Graph &graph) {
    Latencies map;
    using ChannelQueue = 
        std::priority_queue<Channel *, std::vector<Channel *>, ChannelComp>;
    ChannelQueue chanQueue((ChannelComp(map)));

    auto visitChannel =
      [&](Channel *channel) {
        map[channel->target] = std::max(map[channel->target],
                                        map[channel->source] +
                                        channel->source->latency +
                                        channel->offset);
    };

    auto visitNode = [&](Node *node) {
      for (Channel *out: graph.outputs[node]) {
        chanQueue.push(out);
        visitChannel(out);
      }
    };

    for (Node *node: graph.startNodes) {
      visitNode(node);
    }

    while (!chanQueue.empty()) {
      Node *outNode = chanQueue.top()->target;
      chanQueue.pop();
      visitNode(outNode);
    }

    Buffers buffers;

    int32_t maxOutLatency = 0;

    for (Node *node: graph.nodes) {
      if (llvm::isa<OutputOpInterface>(node->op) &&
          map[node] > maxOutLatency) {
        maxOutLatency = map[node];
      }
    }

    for (Node *node: graph.nodes) {
      if (llvm::isa<OutputOpInterface>(node->op)) {
        map[node] = maxOutLatency;
      }

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
    return std::make_pair(buffers, calculateOverallLatency(graph, buffers));
  }

public:
  explicit DFCIRASAPSchedulerPass(const DFCIRASAPSchedulerPassOptions &options)
      : impl::DFCIRASAPSchedulerPassBase<DFCIRASAPSchedulerPass>(options) {}

  void runOnOperation() override {
    // Convert kernel into graph.
    Graph graph(llvm::dyn_cast<ModuleOp>(getOperation()));

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
    mlir::dfcir::utils::insertBuffers(this->getContext(), buffers);

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
