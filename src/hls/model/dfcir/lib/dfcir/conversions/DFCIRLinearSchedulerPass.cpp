#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRLPUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include <unordered_set>
#include <unordered_map>

// TODO: Figure how to avoid using std::cout.
#include <iostream>

namespace mlir::dfcir {
#define GEN_PASS_DECL_DFCIRLINEARSCHEDULERPASS
#define GEN_PASS_DEF_DFCIRLINEARSCHEDULERPASS
#include "dfcir/conversions/DFCIRPasses.h.inc"

    using namespace circt::firrtl::utils;
    class DFCIRLinearSchedulerPass : public impl::DFCIRLinearSchedulerPassBase<DFCIRLinearSchedulerPass> {
        using Node = utils::Node;
        using Channel = utils::Channel;
        using Graph = utils::Graph;
        using LPProblem = utils::lp::LPProblem;
        using OpType = utils::lp::OpType;
        using Status = utils::lp::Status;

    private:

        void synchronizeInput(const Node &node) {
            int *var = new int[1] { _nodeMap[node] };
            double *coeff = new double[1] { 1.0 };

            // t_source = 0
            _problem.addConstraint(1, var, coeff, OpType::Equal, 0);
        }

        void addLatencyConstraint(const Channel &chan) {
            int *vars = new int[2] { _nodeMap[chan.target], _nodeMap[chan.source] };
            double *coeffs = new double[2] { 1.0, -1.0 };

            // t_next >= t_prev + prev_latency + next_prev_offset
            _problem.addConstraint(2, vars, coeffs, OpType::GreaterOrEqual,
                                   (int(chan.source.latency) + chan.offset));
        }

        int addDeltaConstraint(const Channel &chan) {
            int delta_id = _problem.addVariable();
            int *vars = new int[3] { delta_id, _nodeMap[chan.target], _nodeMap[chan.source] };
            double *coeffs = new double[3] { 1.0, -1.0, 1.0 };

            // delta_t = t_next - t_prev
            _problem.addConstraint(3, vars, coeffs, OpType::Equal, 0);
            return delta_id;
        }

        void addBufferConstraint(const Channel &chan) {
            int buf_id = _problem.addVariable();
            _bufMap[chan] = buf_id;
            int *vars = new int[3] { buf_id, _nodeMap[chan.target], _nodeMap[chan.source] };
            double *coeffs = new double[3] { 1.0, -1.0, 1.0 };

            // buf_next_prev = t_next - (t_prev + prev_latency + next_prev_offset)
            _problem.addConstraint(3, vars, coeffs, OpType::Equal, -1.0 * (int(chan.source.latency) + chan.offset));

            // buf_next_prev >= 0
            _problem.addConstraint(1, new int[1] { buf_id }, new double[1] { 1.0 }, OpType::GreaterOrEqual, 0);
        }

        LPProblem _problem;
        std::unordered_map<Node, int> _nodeMap;
        std::unordered_map<Channel, int> _bufMap;

        Buffers schedule(Graph &graph) {
            size_t chanCount = 0;
            for (const Node &node : graph.nodes) {
                chanCount += graph.inputs[node].size();
                _nodeMap[node] = _problem.addVariable();
                if (graph.start_nodes.find(node) != graph.start_nodes.end()) {
                    synchronizeInput(node);
                }
            }
            int *delta_ids = new int[chanCount];
            double *delta_coeffs = new double[chanCount];
            int curr_id = 0;
            for (const Node &node : graph.nodes) {
                for (const Channel &chan : graph.inputs[node]) {
                    addLatencyConstraint(chan);
                    delta_coeffs[curr_id] = 1.0;
                    delta_ids[curr_id++] = addDeltaConstraint(chan);
                    addBufferConstraint(chan);
                }
            }

            // Minimize deltas.
            _problem.setMin();
            _problem.setObjective(chanCount, delta_ids, delta_coeffs);
            _problem.lessMessages();
            int status = _problem.solve();

            Buffers buffers;

            if (status == Status::Optimal || status == Status::Suboptimal) {
                double *result;
                int count = _problem.getResults(&result);
                for (const auto &[chan, id] : _bufMap) {
                    // lp_solve positions start with 1.
                    // TODO: Can we make such a cast?
                    int latency = result[id - 1];
                    if (!chan.source.is_const && latency) {
                        buffers[chan] = latency;
                    }
                }
                delete []result;
            } else {
                // TODO: Replace with a legit logger?
                std::cout << status;
            }

            delete []delta_ids;
            delete []delta_coeffs;
            return buffers;
        }


    public:

        void runOnOperation() override {

            // Convert kernel into graph.
            Graph graph(llvm::dyn_cast<ModuleOp>(getOperation()));

            // Execute scheduler.
            auto buffers = schedule(graph);

            // Insert buffers.
            mlir::dfcir::utils::insertBuffers(this->getContext(), buffers);
        }
    };

    std::unique_ptr<mlir::Pass> createDFCIRLinearSchedulerPass() {
        return std::make_unique<DFCIRLinearSchedulerPass>();
    }

} // namespace mlir::dfcir
