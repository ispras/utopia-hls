#ifndef DFCXX_KERNSTORAGE_H
#define DFCXX_KERNSTORAGE_H

#include "dfcxx/types/type.h"
#include "dfcxx/vars/var.h"

#include <unordered_set>

namespace dfcxx {

class KernStorage {
private:
  std::unordered_set<DFTypeImpl *> types;
  std::unordered_set<DFVariableImpl *> variables;
public:
  DFTypeImpl *addType(DFTypeImpl *type);

  DFType addType(const DFType &type);

  DFVariableImpl *addVariable(DFVariableImpl *var);

  DFVariable addVariable(const DFVariable &var);

  ~KernStorage();
};

} // namespace dfcxx

#endif // DFCXX_KERNSTORAGE_H
