//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"
#include "dfcir/conversions/DFCIRLPUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include <unordered_set>
#include <unordered_map>
#include <utility>
// TODO: Replace with a normal logger.
// Issue #13 (https://github.com/ispras/utopia-hls/issues/13).
#include <iostream>  

namespace mlir::dfcir {

#define GEN_PASS_DECL_DFCIRLINEARSCHEDULERPASS
#define GEN_PASS_DEF_DFCIRLINEARSCHEDULERPASS

#include "dfcir/conversions/DFCIRPasses.h.inc"

class DFCIRLinearSchedulerPass
    : public impl::DFCIRLinearSchedulerPassBase<DFCIRLinearSchedulerPass> {
  using Node = utils::Node;
  using Channel = utils::Channel;
  using Graph = utils::Graph;
  using LPProblem = utils::lp::LPProblem;
  using OpType = utils::lp::OpType;
  using Status = utils::lp::Status;

private:
  void synchronizeInput(Node *node) {
    int *var = new int[1]{nodeMap[node]};
    double *coeff = new double[1]{1.0};

    // t_source = 0
    problem.addConstraint(1, var, coeff, OpType::Equal, 0);
  }

  void addLatencyConstraint(Channel *chan) {
    int *vars = new int[2]{nodeMap[chan->target], nodeMap[chan->source]};
    double *coeffs = new double[2]{1.0, -1.0};

    // t_next >= t_prev + prev_latency + next_prev_offset
    problem.addConstraint(2, vars, coeffs, OpType::GreaterOrEqual,
                          int(chan->source->latency) + chan->offset);
  }

  int addDeltaConstraint(Channel *chan) {
    int deltaID = problem.addVariable();
    int *vars = new int[3]{deltaID, nodeMap[chan->target],
                           nodeMap[chan->source]};
    double *coeffs = new double[3]{1.0, -1.0, 1.0};

    // delta_t = t_next - t_prev
    problem.addConstraint(3, vars, coeffs, OpType::Equal, 0);
    return deltaID;
  }

  void addBufferConstraint(Channel *chan) {
    int bufID = problem.addVariable();
    bufMap[chan] = bufID;
    int *vars = new int[3]{bufID,
                           nodeMap[chan->target],
                           nodeMap[chan->source]};
    double *coeffs = new double[3]{1.0, -1.0, 1.0};

    // buf_next_prev = t_next - (t_prev + prev_latency + next_prev_offset)
    problem.addConstraint(3, vars, coeffs, OpType::Equal,
                          -1.0 * (int(chan->source->latency) + chan->offset));

    // buf_next_prev >= 0
    problem.addConstraint(1, new int[1]{bufID}, new double[1]{1.0},
                          OpType::GreaterOrEqual, 0);
  }

  LPProblem problem;
  std::unordered_map<Node *, int> nodeMap;
  std::unordered_map<Channel *, int> bufMap;

  Node *prevOutputNode = nullptr;

  std::pair<Buffers, int32_t> schedule(Graph &graph) {
    size_t chanCount = 0;
    for (Node *node: graph.nodes) {
      chanCount += graph.inputs[node].size();
      nodeMap[node] = problem.addVariable();
      if (graph.startNodes.find(node) != graph.startNodes.end()) {
        synchronizeInput(node);
      }
    }

    int *deltaIDs = new int[chanCount];
    double *deltaCoeffs = new double[chanCount];
    int curr_id = 0;
    for (Node *node: graph.nodes) {
      for (Channel *chan: graph.inputs[node]) {
        addLatencyConstraint(chan);
        deltaCoeffs[curr_id] = 1.0;
        deltaIDs[curr_id++] = addDeltaConstraint(chan);
        addBufferConstraint(chan);
      }
    }

    // Minimize deltas.
    problem.setMin();
    problem.setObjective(chanCount, deltaIDs, deltaCoeffs);
    problem.lessMessages();
    int status = problem.solve();

    Buffers buffers;

    if (status == Status::Optimal || status == Status::Suboptimal) {
      double *result;
      problem.getResults(&result);
      for (const auto &[chan, id]: bufMap) {
        // lp_solve positions start with 1.
        int32_t latency = result[id - 1];

        using mlir::dfcir::utils::hasConstantInput;
        if (latency && !hasConstantInput(chan->source->op)) {
          buffers[chan] = latency;
        }
      }
      delete []result;
    } else {
      // TODO: Replace with a legit logger?
      // Issue #13 (https://github.com/ispras/utopia-hls/issues/13).
      std::cout << status;
    }

    delete []deltaIDs;
    delete []deltaCoeffs;

    int32_t maxOutLatency = calculateOverallLatency(graph, buffers);

    return std::make_pair(buffers, maxOutLatency);
  }

public:
  explicit DFCIRLinearSchedulerPass(const DFCIRLinearSchedulerPassOptions &options)
      : impl::DFCIRLinearSchedulerPassBase<DFCIRLinearSchedulerPass>(options) {}

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

std::unique_ptr<mlir::Pass> createDFCIRLinearSchedulerPass(LatencyConfig *config) {
  DFCIRLinearSchedulerPassOptions options;
  options.latencyConfig = config;
  return std::make_unique<DFCIRLinearSchedulerPass>(options);
}

} // namespace mlir::dfcir
