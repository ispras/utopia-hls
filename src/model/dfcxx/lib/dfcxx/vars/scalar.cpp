#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/varbuilders/builder.h"
#include "dfcxx/vars/scalar.h"

namespace dfcxx {

DFScalar::DFScalar(const std::string &name, IODirection direction,
                   GraphHelper &helper, dfcxx::DFTypeImpl &type) :
        DFVariableImpl(name, direction, helper),
        type(type) {}

DFTypeImpl &DFScalar::getType() {
  return type;
}

DFVariableImpl &DFScalar::operator+(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::ADD, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator-(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::SUB, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator*(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::MUL, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator/(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::DIV, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator&(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::AND, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator|(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::OR, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator^(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::XOR, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator!() {
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::NOT, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator-() {
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         type);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::NEG, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator<(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::LESS, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator<=(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::LESS_EQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator>(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::MORE, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator>=(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::MORE_EQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator==(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::EQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator!=(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::NEQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator<<(uint8_t bits) {
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildShiftedType(type, bits));
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::SHL, NodeData{.bitShift=bits});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

DFVariableImpl &DFScalar::operator>>(uint8_t bits) {
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildShiftedType(type, bits));
  DFVariableImpl *newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                                         *newType);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::SHR, NodeData{.bitShift=bits});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

bool DFScalar::isScalar() const {
  return true;
}

} // namespace dfcxx