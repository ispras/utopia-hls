#include "dfcir/conversions/DFCIRPassesUtils.h"

#include <iostream>

namespace circt::firrtl::utils {
    inline FExtModuleOp createBufferModule(OpBuilder &builder, llvm::StringRef name, Type type, Location loc, unsigned stages) {
        SmallVector<circt::firrtl::PortInfo> ports = {
                circt::firrtl::PortInfo(
                        mlir::StringAttr::get(builder.getContext(), "out"),
                        type,
                        circt::firrtl::Direction::Out),
                circt::firrtl::PortInfo(
                        mlir::StringAttr::get(builder.getContext(), "in"),
                        type,
                        circt::firrtl::Direction::In)
        };
        IntegerType attrType = mlir::IntegerType::get(builder.getContext(), 32, mlir::IntegerType::Unsigned);
        return builder.create<FExtModuleOp>(
                loc,
                mlir::StringAttr::get(builder.getContext(), name),
                circt::firrtl::ConventionAttr::get(builder.getContext(), Convention::Internal),
                ports,
                StringRef(BUF_MODULE),
                mlir::ArrayAttr(),
                mlir::ArrayAttr::get(
                        builder.getContext(),
                        ParamDeclAttr::get(builder.getContext(),
                                           mlir::StringAttr::get(builder.getContext(), BUF_MODULE_STAGES),
                                           attrType,
                                           mlir::IntegerAttr::get(attrType, stages))));
    }

    inline FExtModuleOp createBufferModuleWithTypeName(OpBuilder &builder, Type type, Location loc, unsigned stages) {
        std::string name = BUF_MODULE;
        llvm::raw_string_ostream nameStream(name);
        type.print(nameStream);
        nameStream << "##" << stages;
        return createBufferModule(builder, name, type, loc, stages);
    }

    FExtModuleOp findOrCreateBufferModule(OpBuilder &builder, Type type, Location loc, unsigned stages) {
        CircuitOp circuit = findCircuit(builder.getInsertionPoint());
        Block *block = circuit.getBodyBlock();
        std::string name = BUF_MODULE;
        llvm::raw_string_ostream nameStream(name);
        type.print(nameStream);
        nameStream << "##" << stages;

        for (auto op = block->op_begin<FExtModuleOp>(); op != block->op_end<FExtModuleOp>(); ++op) {
            if ((*op).getModuleName() == name) {
                return *op;
            }
        }
        auto save = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(block);
        FExtModuleOp result = createBufferModuleWithTypeName(builder, type, loc, stages);
        builder.restoreInsertionPoint(save);
        return result;
    }

    bool isAStartWire(Operation *op) {
        auto casted = llvm::dyn_cast<circt::firrtl::WireOp>(op);
        if (!casted) return false;

        for (const auto &val : op->getUses()) {
            auto found = llvm::dyn_cast<circt::firrtl::ConnectOp>(val.getOwner());
            if (found && found.getDest() == casted.getResult()) return false;
        }
        return true;
    }

    std::pair<Operation *, Operation *> unrollConnectChain(Value value) {
        Value cur_val = value;
        Operation *initial_op = nullptr;
        Operation *connect_op = nullptr;
        bool flag;
        do {
            flag = false;
            for (const auto &operand : cur_val.getUses()) {
                auto found = llvm::dyn_cast<circt::firrtl::ConnectOp>(operand.getOwner());
                if (found && found.getDest() == operand.get()) {
                    cur_val = found.getSrc();
                    if (!connect_op) {
                        connect_op = found.getOperation();
                    }
                    initial_op = cur_val.getDefiningOp();
                    flag = true;
                }
            }
        } while (flag);
        return std::make_pair(initial_op, connect_op);
    }

} // namespace circt::firrtl::utils

#include <iostream>

namespace mlir::dfcir::utils {

    Node::Node(Operation *op, Ops type) : op(op), type(type) {}

    Node::Node() : Node(nullptr, Ops::UNDEFINED) {}

    bool Node::operator ==(const Node &node) const {
        return this->op == node.op && this->type == node.type;
    }

    Channel::Channel(Node source, Node target, Value val, unsigned val_ind, Operation *connect_op)
                    : source(source), target(target),
                      val(val), val_ind(val_ind), connect_op(connect_op) {}

    Channel::Channel() : source(), target(), val(), val_ind(0), connect_op() {}

    bool Channel::operator ==(const Channel &channel) const {
        return this->source == channel.source && this->target == channel.target &&
               this->val == channel.val && this->val_ind == channel.val_ind;
    }

     Graph::Graph() {
        nodes = std::unordered_set<Node>();
        start_nodes = std::unordered_set<Node>();
        inputs = std::unordered_map<Node, std::unordered_set<Channel>>();
        outputs = std::unordered_map<Node, std::unordered_set<Channel>>();
    }

     Graph::Graph(FModuleOp module) : Graph() {
        using circt::firrtl::InstanceOp;
        using circt::firrtl::WireOp;
        using circt::firrtl::utils::isAStartWire;

        assert(module);

        for (WireOp wire : module.getBodyBlock()->getOps<WireOp>()) {
            Operation *op = wire.operator Operation *();
            Node newNode(op, Ops::WIRE);
            nodes.insert(newNode);
            if (isAStartWire(op)) {
                start_nodes.insert(newNode);
                continue;
            }
        }

        for (InstanceOp instance : module.getBodyBlock()->getOps<InstanceOp>()) {
            Operation *op = instance.operator Operation *();
            Ops type = UNDEFINED;

            auto name = instance.getModuleName();
            if (name.starts_with(ADD_MODULE)) {
                type = Ops::ADD;
            } else if (name.starts_with(SUB_MODULE)) {
                type = Ops::SUB;
            } else if (name.starts_with(DIV_MODULE)) {
                type = Ops::DIV;
            } else if (name.starts_with(MUL_MODULE)) {
                type = Ops::MUL;
            }

            Node newNode(op, type);
            nodes.insert(newNode);

            auto directions = instance.getPortDirections();
            for (size_t index = 0, count = instance.getResults().size(); index < count; ++index) {
                if (!directions.isOneBitSet(index)) {
                    // TODO: Handle the case of block arguments.
                    auto operand = instance.getResult(index);
                    auto connectInfo = circt::firrtl::utils::unrollConnectChain(operand);
                    auto found = std::find_if(nodes.begin(), nodes.end(), [&] (const Node &n) {return n.op == connectInfo.first;});
                    if (found == nodes.end()) continue;
                    Channel newChan(*found, newNode, operand, index, connectInfo.second);
                    outputs[*found].insert(newChan);
                    inputs[newNode].insert(newChan);
                }
            }
        }
    }

    Graph::Graph(CircuitOp circuit, StringRef name)
            : Graph(llvm::dyn_cast<FModuleOp>(name.empty() ? circuit.getMainModule() : circuit.lookupSymbol(name))) {}

    Graph::Graph(ModuleOp op, StringRef name)
            : Graph(mlir::utils::findFirstOccurence<CircuitOp>(op), name) {}

    void insertBuffers(mlir::MLIRContext &ctx, const Buffers &buffers) {
        using circt::firrtl::FExtModuleOp;
        using circt::firrtl::InstanceOp;
        using circt::firrtl::ConnectOp;
        OpBuilder builder(&ctx);
        for (auto &[channel, latency] : buffers) {
            builder.setInsertionPoint(channel.target.op);
            FExtModuleOp bufModule = circt::firrtl::utils::findOrCreateBufferModule(builder, channel.val.getType(), builder.getUnknownLoc(), latency);
            InstanceOp instance = builder.create<InstanceOp>(builder.getUnknownLoc(), bufModule, "placeholder");
            ConnectOp castedConnect = llvm::dyn_cast<ConnectOp>(channel.connect_op);
            // Take the original connectee operand and bind it to a newly created 'ConnectOp'
            Value connecteeVal = castedConnect.getSrc();
            builder.create<ConnectOp>(builder.getUnknownLoc(), instance.getResult(1), connecteeVal);
            // Set the original connectee operand to the newly created 'ConnectOp''s result.
            castedConnect.setOperand(1, instance.getResult(0));
        }
    }

} // namespace mlir::dfcir::utils