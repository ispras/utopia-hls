#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/types/types.h"
#include "dfcxx/varbuilders/builder.h"
#include "dfcxx/vars/constant.h"

namespace dfcxx {

DFConstant::DFConstant(GraphHelper &helper, dfcxx::DFTypeImpl &type,
                       ConstantTypeKind kind, ConstantValue value) :
        DFVariableImpl("", IODirection::NONE, helper),
        type(type),
        kind(kind),
        value(value) {}

DFTypeImpl &DFConstant::getType() {
  return type;
}

DFVariableImpl &DFConstant::operator+(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ + casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ + casted.value.uint_;
        break;
      case FLOAT:
        val.double_ = value.double_ + casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper, type);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::ADD, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator-(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ - casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ - casted.value.uint_;
        break;
      case FLOAT:
        val.double_ = value.double_ - casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper, type);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::SUB, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator*(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ * casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ * casted.value.uint_;
        break;
      case FLOAT:
        val.double_ = value.double_ * casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper, type);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::MUL, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator/(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ / casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ / casted.value.uint_;
        break;
      case FLOAT:
        val.double_ = value.double_ / casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper, type);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::DIV, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator&(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ & casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ & casted.value.uint_;
        break;
      case FLOAT:
        // TODO: FIX.
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper, type);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::AND, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator|(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ | casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ | casted.value.uint_;
        break;
      case FLOAT:
        // TODO: FIX.
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper, type);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::OR, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator^(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ | casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ | casted.value.uint_;
        break;
      case FLOAT:
        // TODO: FIX.
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper, type);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::XOR, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator!() {
  DFVariableImpl *newVar;
  ConstantValue val{};
  switch (kind) {
    case INT:
      val.int_ = ~value.int_;
      break;
    case UINT:
      val.uint_ = ~value.uint_;
      break;
    case FLOAT:
      // TODO: FIX.
      break;
  }
  newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::NOT, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator-() {
  DFVariableImpl *newVar;
  ConstantValue val{};
  switch (kind) {
    case INT:
      val.int_ = -value.int_;
      break;
    case UINT:
      val.uint_ = -value.uint_;
      break;
    case FLOAT:
      val.double_ = -value.double_;
      break;
  }
  newVar = helper.varBuilder.buildConstant(helper, type, kind, val);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::NEG, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator<(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.uint_ = value.int_ < casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ < casted.value.uint_;
        break;
      case FLOAT:
        val.uint_ = value.double_ < casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper,
                                             *newType,
                                             kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                           *newType);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::LESS, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator<=(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.uint_ = value.int_ <= casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ <= casted.value.uint_;
        break;
      case FLOAT:
        val.uint_ = value.double_ <= casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper,
                                             *newType,
                                             kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                           *newType);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::LESS_EQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator>(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.uint_ = value.int_ > casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ > casted.value.uint_;
        break;
      case FLOAT:
        val.uint_ = value.double_ > casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper,
                                             *newType,
                                             kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                           *newType);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::MORE, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator>=(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.uint_ = value.int_ >= casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ >= casted.value.uint_;
        break;
      case FLOAT:
        val.uint_ = value.double_ >= casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper,
                                             *newType,
                                             kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                           *newType);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::MORE_EQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator==(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.uint_ = value.int_ == casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ == casted.value.uint_;
        break;
      case FLOAT:
        val.uint_ = value.double_ == casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper,
                                             *newType,
                                             kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                           *newType);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::EQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator!=(DFVariableImpl &rhs) {
  //if (type != rhs.getType()) { throw std::exception(); }
  DFVariableImpl *newVar;
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    ConstantValue val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.uint_ = value.int_ != casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ != casted.value.uint_;
        break;
      case FLOAT:
        val.uint_ = value.double_ != casted.value.double_;
        break;
    }
    newVar = helper.varBuilder.buildConstant(helper,
                                             *newType,
                                             kind, val);
  } else {
    newVar = helper.varBuilder.buildStream("", IODirection::NONE, helper,
                                           *newType);
  }
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::NEQ, NodeData{});
  helper.addChannel(this, newVar, 0, false);
  helper.addChannel(&rhs, newVar, 1, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator<<(uint8_t bits) {
  DFVariableImpl *newVar;
  ConstantValue val{};
  switch (kind) {
    case INT:
      val.int_ = value.int_ << bits;
      break;
    case UINT:
      val.uint_ = value.uint_ << bits;
      break;
    case FLOAT:
    // TODO: FIX.
      break;
  }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildShiftedType(type, bits));
  newVar = helper.varBuilder.buildConstant(helper, *newType, kind, val);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::SHL, NodeData{.bitShift=bits});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

DFVariableImpl &DFConstant::operator>>(uint8_t bits) {
  DFVariableImpl *newVar;
  ConstantValue val{};
  switch (kind) {
    case INT:
      val.int_ = value.int_ >> bits;
      break;
    case UINT:
      val.uint_ = value.uint_ >> bits;
      break;
    case FLOAT:
    // TODO: FIX.
      break;
  }
  DFTypeImpl *newType = helper.storage.addType(helper.typeBuilder.buildShiftedType(type, int8_t(bits) * -1));
  newVar = helper.varBuilder.buildConstant(helper, *newType, kind, val);
  helper.storage.addVariable(newVar);
  helper.addNode(newVar, OpType::SHR, NodeData{.bitShift=bits});
  helper.addChannel(this, newVar, 0, false);
  return *newVar;
}

ConstantTypeKind DFConstant::getKind() const {
  return kind;
}

int64_t DFConstant::getInt() const {
  return value.int_;
}

uint64_t DFConstant::getUInt() const {
  return value.uint_;
}

double DFConstant::getDouble() const {
  return value.double_;
}

bool DFConstant::isConstant() const {
  return true;
}

} // namespace dfcxx