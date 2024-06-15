#include "dfcxx/varbuilders/builder.h"

namespace dfcxx {

DFVariableImpl *
VarBuilder::buildStream(const std::string &name, IODirection direction,
                        GraphHelper &helper, DFTypeImpl &type) {
  return new DFStream(name, direction, helper, type);
}

DFVariableImpl *
VarBuilder::buildScalar(const std::string &name, IODirection direction,
                        GraphHelper &helper, DFTypeImpl &type) {
  return new DFScalar(name, direction, helper, type);
}

DFVariableImpl *
VarBuilder::buildConstant(GraphHelper &helper, DFTypeImpl &type,
                          ConstantTypeKind kind, ConstantValue value) {
  return new DFConstant(helper, type, kind, value);
}

DFVariable
VarBuilder::buildStream(const std::string &name, IODirection direction,
                        GraphHelper &helper, const DFType &type) {
  return DFVariable(buildStream(name, direction, helper, *(type.getImpl())));
}

DFVariable
VarBuilder::buildScalar(const std::string &name, IODirection direction,
                        GraphHelper &helper, const DFType &type) {
  return DFVariable(buildScalar(name, direction, helper, *(type.getImpl())));
}

DFVariable
VarBuilder::buildConstant(GraphHelper &helper, const DFType &type,
                          ConstantTypeKind kind, ConstantValue value) {
  return DFVariable(buildConstant(helper, *(type.getImpl()), kind, value));
}

DFVariable
VarBuilder::buildMuxCopy(const DFVariable &var, GraphHelper &helper) {
  if (var.isConstant()) {
    return buildConstant(helper, var.getType(),
                         ((DFConstant *) (var.getImpl()))->getKind(),
                         ConstantValue{});
  } else if (var.isScalar()) {
    return buildScalar("", IODirection::NONE, helper,
                       var.getType());
  } else /* if (var.isStream()) */ {
    return buildStream("", IODirection::NONE, helper,
                       var.getType());
  }
}

} // namespace dfcxx