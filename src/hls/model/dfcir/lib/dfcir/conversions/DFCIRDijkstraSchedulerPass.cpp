#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include <queue>


namespace mlir::dfcir {
#define GEN_PASS_DECL_DFCIRDIJKSTRASCHEDULERPASS
#define GEN_PASS_DEF_DFCIRDIJKSTRASCHEDULERPASS
#include "dfcir/conversions/DFCIRPasses.h.inc"

    using namespace circt::firrtl::utils;
    class DFCIRDijkstraSchedulerPass : public impl::DFCIRDijkstraSchedulerPassBase<DFCIRDijkstraSchedulerPass> {
        using Node = utils::Node;
        using Channel = utils::Channel;
        using Graph = utils::Graph;

        class ChannelComp final {
        private:
            Latencies &map;
        public:
            explicit ChannelComp(Latencies &map) : map(map) {}
            bool operator() (const Channel &lhs, const Channel &rhs) const {
                return map[lhs.source] + lhs.source.latency < map[rhs.source] + rhs.source.latency;
            }
        };

    private:

        Buffers schedule(Graph &graph) {
            Latencies map;
            std::priority_queue<Channel, std::vector<Channel>, ChannelComp> chanQueue((ChannelComp(map)));

            auto visitChannel = [&] (const Channel &channel) {
                map[channel.target] = std::max(map[channel.target], map[channel.source] + channel.source.latency);
            };

            auto visitNode = [&] (const Node &node) {
                for (const Channel &out : graph.outputs[node]) {
                    chanQueue.push(out);
                    visitChannel(out);
                }
            };

            for (const Node &node : graph.start_nodes) {
                visitNode(node);
            }

            while (!chanQueue.empty()) {
                Node outNode = chanQueue.top().target;
                chanQueue.pop();
                visitNode(outNode);
            }

            Buffers buffers;

            for (const Node &node : graph.nodes) {
                for (const Channel &channel : graph.inputs[node]) {
                    unsigned delta = map[channel.target] - (map[channel.source] + channel.source.latency);
                    if (delta) {
                        buffers[channel] = delta;
                    }
                }
            }

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

    std::unique_ptr<mlir::Pass> createDFCIRDijkstraSchedulerPass() {
        return std::make_unique<DFCIRDijkstraSchedulerPass>();
    }

} // namespace mlir::dfcir
