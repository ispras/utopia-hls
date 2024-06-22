#ifndef DFCXX_TYPE_BUILDER_H
#define DFCXX_TYPE_BUILDER_H

#include "dfcxx/types/types.h"

namespace dfcxx {

class TypeBuilder {
public:
  DFTypeImpl *buildFixed(SignMode mode, uint8_t intBits, uint8_t fracBits);

  DFTypeImpl *buildBool();

  DFTypeImpl *buildFloat(uint8_t expBits, uint8_t fracBits);

  DFTypeImpl *buildShiftedType(DFTypeImpl &type, int8_t shift);

  DFType buildShiftedType(const DFType &type, int8_t shift);
};

} // namespace dfcxx

#endif // DFCXX_TYPE_BUILDER_H