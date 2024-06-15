#include "dfcxx/kernstorage.h"

#include <algorithm>

namespace dfcxx {

dfcxx::DFTypeImpl *KernStorage::addType(dfcxx::DFTypeImpl *type) {
  auto found = std::find_if(types.begin(), types.end(), [&] (dfcxx::DFTypeImpl *t) { return t->operator==(*type); });
  if (found != types.end()) {
    delete type;
    return *found;
  } else {
    return *(types.insert(type).first);
  }
}

DFType KernStorage::addType(const DFType &type) {
  return DFType(addType(type.getImpl()));
}

DFVariableImpl *KernStorage::addVariable(DFVariableImpl *var) {
  return *(variables.insert(var).first);
}

DFVariable KernStorage::addVariable(const DFVariable &var) {
  return DFVariable(addVariable(var.getImpl()));
}

KernStorage::~KernStorage() {
  for (DFTypeImpl *type: types) {
    delete type;
  }
  for (DFVariableImpl *var: variables) {
    delete var;
  }
}

} // namespace dfcxx