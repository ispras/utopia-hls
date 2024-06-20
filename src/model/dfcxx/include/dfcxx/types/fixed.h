#ifndef DFCXX_FIXED_H
#define DFCXX_FIXED_H

#include "dfcxx/types/type.h"

namespace dfcxx {

enum SignMode {
  UNSIGNED = 0,
  SIGNED
};

class TypeBuilder;

class FixedType : DFTypeImpl {
  friend TypeBuilder;
private:
  SignMode mode;
  uint8_t intBits;
  uint8_t fracBits;

  FixedType(SignMode mode, uint8_t intBits, uint8_t fracBits);

public:
  SignMode getSign() const;

  uint8_t getIntBits() const;

  uint8_t getFracBits() const;

  uint16_t getTotalBits() const override;

  ~FixedType() override = default;

  bool operator==(const DFTypeImpl &rhs) const override;

  bool isFixed() const override;

  bool isInt() const;

  bool isSigned() const;

  bool isUnsigned() const;

  bool isBool() const;
};

} // namespace dfcxx

#endif // DFCXX_FIXED_H
