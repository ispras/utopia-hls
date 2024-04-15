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
                            circt::firrtl::Direction::In),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(builder.getContext(), "clk"),
                            circt::firrtl::ClockType::get(builder.getContext()),
                            circt::firrtl::Direction::In)
            };
            IntegerType attrType = mlir::IntegerType::get(builder.getContext(), 32, mlir::IntegerType::Unsigned);
            auto typeWidth = circt::firrtl::getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(type));
            assert(typeWidth.has_value());
            return builder.create<FExtModuleOp>(
                    loc,
                    mlir::StringAttr::get(builder.getContext(), name),
                    circt::firrtl::ConventionAttr::get(builder.getContext(), Convention::Internal),
                    ports,
                    StringRef(BUF_MODULE),
                    mlir::ArrayAttr(),
                    mlir::ArrayAttr::get(
                            builder.getContext(),
                            {
                            ParamDeclAttr::get(builder.getContext(),
                                               mlir::StringAttr::get(builder.getContext(), STAGES_PARAM),
                                               attrType,
                                               mlir::IntegerAttr::get(attrType, stages)),
                            ParamDeclAttr::get(builder.getContext(),
                                               mlir::StringAttr::get(builder.getContext(), "in_" TYPE_SIZE_PARAM),
                                               attrType,
                                               mlir::IntegerAttr::get(attrType, *typeWidth))
                            }));
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

        std::pair<Value, Operation *> unrollConnectChain(Value value) {
            Value cur_val = value;
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
                        flag = true;
                    }
                }
            } while (flag);
            return std::make_pair(cur_val, connect_op);
        }

        Value getBlockArgument(Block *block, unsigned ind) {
            return block->getArgument(ind);
        }

        Value getBlockArgumentFromOpBlock(Operation *op, unsigned ind) {
            return getBlockArgument(op->getBlock(), ind);
        }

        Value getClockVar(Block *block) {

            Value arg = getBlockArgument(block, block->getNumArguments() - 1);
            if (arg.getType().isa<ClockType>()) {
                return arg;
            }
            return nullptr;
        }

        Value getClockVarFromOpBlock(Operation *op) {
            return getClockVar(op->getBlock());
        }

        ConnectOp createConnect(OpBuilder &builder, Value destination, Value source, int offset) {
            auto connect = builder.create<ConnectOp>(builder.getUnknownLoc(), destination, source);
            connect->setAttr(CONNECT_OFFSET_ATTR, builder.getI32IntegerAttr(offset));
            return connect;
        }

        int getConnectOffset(ConnectOp connect) {
            return (connect && connect->hasAttr(CONNECT_OFFSET_ATTR)) ? connect->getAttr(CONNECT_OFFSET_ATTR).cast<IntegerAttr>().getInt() : 0;
        }

        int setConnectOffset(ConnectOp connect, int offset) {
            connect->setAttr(CONNECT_OFFSET_ATTR, mlir::IntegerAttr::get(mlir::IntegerType::get(connect.getContext(), 32, mlir::IntegerType::Signed), offset));
            return getConnectOffset(connect);
        }

    } // namespace circt::firrtl::utils

    namespace mlir::dfcir::utils {

        Node::Node(Operation *op, unsigned latency, bool is_const, long arg_ind) : op(op), latency(latency), is_const(is_const), arg_ind(arg_ind) {}

        Node::Node() : Node(nullptr) {}

        bool Node::operator ==(const Node &node) const {
            return this->op == node.op && this->latency == node.latency && this->is_const == node.is_const && this->arg_ind == node.arg_ind;
        }

        Channel::Channel(Node source, Node target, Value val, unsigned val_ind, Operation *connect_op, int offset)
                        : source(source), target(target),
                          val(val), val_ind(val_ind), connect_op(connect_op), offset(offset) {}

        Channel::Channel() : source(), target(), val(), val_ind(0), connect_op(nullptr), offset(0) {}

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

        auto Graph::findNode(const std::pair<Value, Operation *> &connectInfo) {
            bool isArg = connectInfo.first.isa<BlockArgument>();
            return std::find_if(nodes.begin(), nodes.end(),
                                [&] (const Node &n) {
                                     return (isArg) ? (n.arg_ind == connectInfo.first.cast<BlockArgument>().getArgNumber()) : (n.op == connectInfo.first.getDefiningOp());
                                });
        }

         Graph::Graph(FModuleOp module) : Graph() {
            using circt::firrtl::ClockType;
            using circt::firrtl::InstanceOp;
            using circt::firrtl::ConnectOp;
            using circt::firrtl::MultibitMuxOp;
            using circt::firrtl::ConstantOp;
            using circt::firrtl::utils::isAStartWire;
            using circt::firrtl::utils::getConnectOffset;
            using circt::firrtl::utils::unrollConnectChain;

            assert(module);

            for (BlockArgument arg : module.getArguments()) {
                if (!(arg.getType().isa<circt::firrtl::ClockType>())) {
                    Node newNode(nullptr, 0, false, arg.getArgNumber());
                    nodes.insert(newNode);
                    if (module.getPortDirection(arg.getArgNumber()) == circt::firrtl::Direction::In) {
                        start_nodes.insert(newNode);
                    }
                }
            }

            for (Operation &op : module.getBodyBlock()->getOperations()) {

                if (auto constOp = llvm::dyn_cast<ConstantOp>(&op)) {
                    Node newNode(constOp, 0, true);
                    nodes.insert(newNode);
                    start_nodes.insert(newNode);
                } else if (auto muxOp = llvm::dyn_cast<MultibitMuxOp>(&op)) {
                    Node newNode(muxOp);
                    nodes.insert(newNode);
                    for (size_t index = 0, count = muxOp.getOperands().size(); index < count; ++index) {
                        auto operand = muxOp.getOperand(index);
                        if (!llvm::isa<ClockType>(operand.getType())) {
                            auto connectInfo = unrollConnectChain(operand);

                            auto found = findNode(connectInfo);

                            if (found == nodes.end()) continue;
                            Channel newChan(*found, newNode, operand, index, connectInfo.second, 0);
                            outputs[*found].insert(newChan);
                            inputs[newNode].insert(newChan);
                        }
                    }
                } else if (auto instanceOp = llvm::dyn_cast<InstanceOp>(&op)) {
                    unsigned latency = instanceOp.getReferencedModule()->getAttr(INSTANCE_LATENCY_ATTR).cast<IntegerAttr>().getUInt();

                    Node newNode(instanceOp, latency);
                    nodes.insert(newNode);

                    auto directions = instanceOp.getPortDirections();
                    for (size_t index = 0, count = instanceOp.getResults().size(); index < count; ++index) {
                        auto operand = instanceOp.getResult(index);
                        if (!directions.isOneBitSet(index) && !(operand.getType().isa<ClockType>())) {
                            // TODO: Check that all block arguments cases are handled properly.
                            auto connectInfo = unrollConnectChain(operand);

                            auto found = findNode(connectInfo);

                            if (found == nodes.end()) continue;
                            Channel newChan(*found, newNode, operand, index, connectInfo.second,
                                            getConnectOffset(llvm::cast<ConnectOp>(connectInfo.second)));
                            outputs[*found].insert(newChan);
                            inputs[newNode].insert(newChan);
                        }
                    }
                }
            }
        }

        Graph::Graph(CircuitOp circuit, StringRef name)
                : Graph(llvm::dyn_cast<FModuleOp>(name.empty() ? circuit.getMainModule() : circuit.lookupSymbol(name))) {}

        Graph::Graph(ModuleOp op, StringRef name)
                : Graph(mlir::utils::findFirstOccurence<CircuitOp>(op), name) {}

        void insertBuffer(OpBuilder &builder, circt::firrtl::InstanceOp buf, const Channel &channel) {
            using circt::firrtl::InstanceOp;
            using circt::firrtl::ConnectOp;
            using circt::firrtl::MultibitMuxOp;
            using circt::firrtl::utils::createConnect;
            using circt::firrtl::utils::getClockVarFromOpBlock;
            assert(!channel.source.is_const);

            if (llvm::isa<InstanceOp>(channel.target.op)) {
                ConnectOp castedConnect = llvm::dyn_cast<ConnectOp>(channel.connect_op);
                // Take the original connectee operand and bind it to a newly created 'ConnectOp'
                Value connecteeVal = castedConnect.getSrc();
                createConnect(builder, buf.getResult(1), connecteeVal);
                createConnect(builder, buf.getResult(2), getClockVarFromOpBlock(buf));
                // Set the original connectee operand to the newly created 'ConnectOp''s result.
                castedConnect.setOperand(1, buf.getResult(0));
            } else if (auto muxOp = llvm::dyn_cast<MultibitMuxOp>(channel.target.op)) {
                // Just bind the newly created 'ConnectOp'.
                createConnect(builder, buf.getResult(1), channel.val);
                createConnect(builder, buf.getResult(2), getClockVarFromOpBlock(buf));
                muxOp.setOperand(channel.val_ind, buf.getResult(0));
            }
        }

        void insertBuffers(mlir::MLIRContext &ctx, const Buffers &buffers) {
            using circt::firrtl::InstanceOp;
            using circt::firrtl::utils::findOrCreateBufferModule;
            OpBuilder builder(&ctx);
            for (auto &[channel, latency] : buffers) {
                assert(!channel.source.is_const);
                builder.setInsertionPoint(channel.target.op);
                auto bufModule = findOrCreateBufferModule(builder, channel.val.getType(), builder.getUnknownLoc(), latency);
                auto instance = builder.create<InstanceOp>(builder.getUnknownLoc(), bufModule, "placeholder");
                insertBuffer(builder, instance, channel);
            }
        }

    } // namespace mlir::dfcir::utils