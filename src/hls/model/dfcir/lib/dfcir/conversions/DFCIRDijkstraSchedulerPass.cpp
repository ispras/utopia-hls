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
            LatencyConfig &config;
        public:
            explicit ChannelComp(Latencies &map, LatencyConfig &config) : map(map), config(config) {}
            bool operator() (const Channel &lhs, const Channel &rhs) const {
                return map[lhs.source] + config[lhs.source.type] < map[rhs.source] + config[rhs.source.type];
            }
        };

    private:

        Buffers schedule(Graph &graph) {
            Latencies map;
            std::priority_queue<Channel, std::vector<Channel>, ChannelComp> queue(ChannelComp(map, *latencyConfig));

            auto visitChannel = [&] (const Channel &channel) {
                map[channel.target] = std::max(map[channel.target], map[channel.source] + (*latencyConfig)[channel.source.type]);
            };

            auto visitNode = [&] (const Node &node) {
                for (const Channel &out : graph.outputs[node]) {
                    queue.push(out);
                    visitChannel(out);
                }
            };

            for (const Node &node : graph.start_nodes) {
                visitNode(node);
            }

            while (!queue.empty()) {
                Node outNode = queue.top().target;
                queue.pop();
                visitNode(outNode);
            }

            Buffers buffers;

            for (const Node &node : graph.nodes) {
                for (const Channel &channel : graph.inputs[node]) {
                    unsigned delta = map[channel.target] - (map[channel.source] + (*latencyConfig)[channel.source.type]);
                    if (delta) {
                        buffers[channel] = delta;
                    }
                }
            }

            return buffers;
        }

    public:
        explicit DFCIRDijkstraSchedulerPass(const DFCIRDijkstraSchedulerPassOptions &options)
                : impl::DFCIRDijkstraSchedulerPassBase<DFCIRDijkstraSchedulerPass>(options) { }

        void runOnOperation() override {

            // Convert kernel into graph.
            Graph graph(llvm::dyn_cast<ModuleOp>(getOperation()));

            // Execute scheduler.
            auto buffers = schedule(graph);

            // Insert buffers.
            mlir::dfcir::utils::insertBuffers(this->getContext(), buffers);
        }
    };

    std::unique_ptr<mlir::Pass> createDFCIRDijkstraSchedulerPass(LatencyConfig *config) {
        DFCIRDijkstraSchedulerPassOptions options;
        options.latencyConfig = config;
        return std::make_unique<DFCIRDijkstraSchedulerPass>(options);
    }

} // namespace mlir::dfcir
