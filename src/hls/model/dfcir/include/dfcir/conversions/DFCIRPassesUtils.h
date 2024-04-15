#ifndef DFCIR_PASSES_UTILS_H
#define DFCIR_PASSES_UTILS_H

#include "dfcir/DFCIROperations.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Iterators.h"
#include "dfcir/conversions/DFCIRPasses.h"
#include <memory>
#include <unordered_set>

#define ADD_MODULE "__ADD"
#define SUB_MODULE "__SUB"
#define DIV_MODULE "__DIV"
#define MUL_MODULE "__MUL"

#define FLOAT_SPEC "FLOAT"
#define INT_SPEC "INT"
#define UINT_SPEC "UINT"
#define SINT_SPEC "SINT"


#define BUF_MODULE "__FIFO"
#define STAGES_PARAM "stages"
#define CLOCK_ARG "clk"

#define TYPE_SIZE_PARAM "size"
#define INSTANCE_LATENCY_ATTR "__latency"
#define CONNECT_OFFSET_ATTR "__offset"

namespace mlir::dfcir::utils {
    struct Node;
    struct Channel;
    struct Graph;
} // namespace mlir::dfcir::utils

typedef std::unordered_map<mlir::dfcir::utils::Node, int> Latencies;
typedef std::unordered_map<mlir::dfcir::utils::Channel, int> Buffers;
typedef std::unordered_map<mlir::Operation *, unsigned> ModuleArgMap;

namespace circt::firrtl::utils {

    template<typename OpTy>
    inline CircuitOp findCircuit(const OpTy &op) {
        return op->template getParentOfType<CircuitOp>();
    }



    inline FExtModuleOp createBufferModule(OpBuilder &builder, llvm::StringRef name, Type type, Location loc, unsigned stages);
    inline FExtModuleOp createBufferModuleWithTypeName(OpBuilder &builder, Type type, Location loc, unsigned stages);
    FExtModuleOp findOrCreateBufferModule(OpBuilder &builder, Type type, Location loc, unsigned stages);
    bool isAStartWire(Operation *op);
    std::pair<Value, Operation *> unrollConnectChain(Value value);
    Value getBlockArgument(Block *block, unsigned ind);
    Value getBlockArgumentFromOpBlock(Operation *op, unsigned ind);
    Value getClockVar(Block *block);
    Value getClockVarFromOpBlock(Operation *op);
    ConnectOp createConnect(OpBuilder &builder, Value destination, Value source, int offset = 0);
    int getConnectOffset(ConnectOp connect);
    int setConnectOffset(ConnectOp connect, int offset);
} // namespace circt::firrtl::utils

namespace mlir::utils {

    template<typename OpTy>
    inline OpTy findFirstOccurence(Operation *op) {
        Operation *result = nullptr;
        op->template walk<mlir::WalkOrder::PreOrder>([&](Operation *found) -> mlir::WalkResult {
            if (llvm::dyn_cast<OpTy>(found)) {
                result = found;
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
        });
        return llvm::dyn_cast<OpTy>(result);
    }
} // namespace mlir::utils

namespace mlir::dfcir::utils {

    struct Node {
        Operation *op;
        unsigned latency;
        bool is_const;
        long arg_ind;
        explicit Node(Operation *op, unsigned latency = 0, bool is_const = false, long arg_ind = -1);
        Node();
        Node(const Node &node) = default;
        ~Node() = default;
        bool operator ==(const Node &node) const;
    };

    struct Channel {
        Node source;
        Node target;
        Value val;
        unsigned val_ind;
        Operation *connect_op;
        int offset;
        Channel(Node source, Node target, Value val, unsigned val_ind, Operation *connect_op, int offset = 0);
        Channel();
        Channel(const Channel &) = default;
        ~Channel() = default;
        bool operator ==(const Channel &channel) const;
    };

} // namespace mlir::dfcir::utils

template<>
struct std::hash<mlir::dfcir::utils::Node> {
    size_t operator() (const mlir::dfcir::utils::Node &node) const noexcept {
        return std::hash<mlir::Operation *>()(node.op) + node.arg_ind;
    }
};

template<>
struct std::hash<mlir::dfcir::utils::Channel> {
    using Node = mlir::dfcir::utils::Node;
    size_t operator() (const mlir::dfcir::utils::Channel &channel) const noexcept {
        return std::hash<Node>()(channel.target) + 13 + channel.val_ind;
    }
};

namespace mlir::dfcir::utils {
    struct Graph {
        using FModuleOp = circt::firrtl::FModuleOp;
        using CircuitOp = circt::firrtl::CircuitOp;
        using StringRef = llvm::StringRef;

        std::unordered_set<Node> nodes;
        std::unordered_set<Node> start_nodes;
        std::unordered_map<Node, std::unordered_set<Channel>> inputs;
        std::unordered_map<Node, std::unordered_set<Channel>> outputs;

        explicit Graph();

        auto findNode(const std::pair<Value, Operation *> &connectInfo);

        explicit Graph(FModuleOp module);

        explicit Graph(CircuitOp circuit, StringRef name = StringRef());

        explicit Graph(ModuleOp op, StringRef name = StringRef());
    };

    void insertBuffer(OpBuilder &builder, circt::firrtl::InstanceOp buf, const Channel &channel);
    void insertBuffers(mlir::MLIRContext &ctx, const Buffers &buffers);

} // namespace mlir::dfcir::utils

#endif // DFCIR_PASSES_UTILS_H
