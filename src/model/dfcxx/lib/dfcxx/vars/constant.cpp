//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernmeta.h"
#include "dfcxx/vars/constant.h"

namespace dfcxx {

DFConstant::DFConstant(KernMeta &meta, DFTypeImpl *type, Value value) :
                       DFVariableImpl("", IODirection::NONE, meta),
                       type(*type), value(value) {
  if (this->type.isFixed()) {
    if (((FixedType &) this->type).isSigned()) {
      kind = TypeKind::INT;
    } else {
      kind = TypeKind::UINT;
    }
  } else {
    kind = TypeKind::FLOAT;
  }
}

DFVariableImpl *DFConstant::clone() const {
  return new DFConstant(meta, &type, value);
}

DFTypeImpl *DFConstant::getType() {
  return &type;
}

DFVariableImpl *DFConstant::operator+(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta, &type, val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::ADD, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator-(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta, &type, val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SUB, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator*(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta, &type, val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::MUL, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator/(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta, &type, val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::DIV, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator&(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    Value val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ & casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ & casted.value.uint_;
        break;
      case FLOAT:
        // TODO: For discussion.
        // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
        break;
    }
    newVar = meta.varBuilder.buildConstant(meta, &type, val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::AND, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator|(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    Value val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ | casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ | casted.value.uint_;
        break;
      case FLOAT:
        // TODO: For discussion.
        // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
        break;
    }
    newVar = meta.varBuilder.buildConstant(meta, &type, val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::OR, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator^(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  if (rhs.isConstant()) {
    Value val{};
    DFConstant &casted = (DFConstant &) (rhs);
    switch (kind) {
      case INT:
        val.int_ = value.int_ | casted.value.int_;
        break;
      case UINT:
        val.uint_ = value.uint_ | casted.value.uint_;
        break;
      case FLOAT:
        // TODO: For discussion.
        // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
        break;
    }
    newVar = meta.varBuilder.buildConstant(meta, &type, val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::XOR, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator!() {
  DFVariableImpl *newVar;
  Value val{};
  switch (kind) {
    case INT:
      val.int_ = ~value.int_;
      break;
    case UINT:
      val.uint_ = ~value.uint_;
      break;
    case FLOAT:
      // TODO: For discussion.
      // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
      break;
  }
  newVar = meta.varBuilder.buildConstant(meta, &type, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::NOT, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator-() {
  DFVariableImpl *newVar;
  Value val{};
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
  newVar = meta.varBuilder.buildConstant(meta, &type, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::NEG, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator<(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta,
                                             newType,
                                             val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta,
                                           newType);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::LESS, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator<=(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta,
                                             newType,
                                             val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta,
                                           newType);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::LESSEQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator>(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta,
                                             newType,
                                             val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta,
                                           newType);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::GREATER, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator>=(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta,
                                             newType,
                                             val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta,
                                           newType);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::GREATEREQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator==(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta,
                                             newType,
                                             val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta,
                                           newType);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::EQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator!=(DFVariableImpl &rhs) {
  DFVariableImpl *newVar;
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  if (rhs.isConstant()) {
    Value val{};
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
    newVar = meta.varBuilder.buildConstant(meta,
                                             newType,
                                             val);
  } else {
    newVar = meta.varBuilder.buildStream("", IODirection::NONE, meta,
                                           newType);
  }
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::NEQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator<<(uint8_t bits) {
  DFVariableImpl *newVar;
  Value val{};
  switch (kind) {
    case INT:
      val.int_ = value.int_ << bits;
      break;
    case UINT:
      val.uint_ = value.uint_ << bits;
      break;
    case FLOAT:
      // TODO: For discussion.
      // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
      break;
  }
  DFTypeImpl *newType = meta.storage.addType(
      meta.typeBuilder.buildShiftedType(&type, bits));
  newVar = meta.varBuilder.buildConstant(meta, newType, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SHL, NodeData{.bitShift=bits});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFConstant::operator>>(uint8_t bits) {
  DFVariableImpl *newVar;
  Value val{};
  switch (kind) {
    case INT:
      val.int_ = value.int_ >> bits;
      break;
    case UINT:
      val.uint_ = value.uint_ >> bits;
      break;
    case FLOAT:
      // TODO: For discussion.
      // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
      break;
  }
  DFTypeImpl *newType = meta.storage.addType(
      meta.typeBuilder.buildShiftedType(&type, int8_t(bits) * -1));
  newVar = meta.varBuilder.buildConstant(meta, newType, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SHR, NodeData{.bitShift=bits});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFConstant::TypeKind DFConstant::getKind() const {
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
