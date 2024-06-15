#ifndef DFCXX_CONST_H
#define DFCXX_CONST_H

#include "dfcxx/vars/var.h"

namespace dfcxx {

enum ConstantTypeKind : uint8_t {
  INT = 0,
  UINT,
  FLOAT
};

union ConstantValue {
  int64_t int_;
  uint64_t uint_;
  double double_;
};

class VarBuilder;
class DFCIRBuilder;

class DFConstant : DFVariableImpl {
  friend VarBuilder;
  friend DFCIRBuilder;

private:
  DFTypeImpl &type;
  ConstantTypeKind kind;
  ConstantValue value;

  DFConstant(GraphHelper &helper, DFTypeImpl &type, ConstantTypeKind kind,
             ConstantValue value);

public:
  ~DFConstant() override = default;

protected:
  DFTypeImpl &getType() override;

  DFVariableImpl &operator+(DFVariableImpl &rhs) override;

  DFVariableImpl &operator-(DFVariableImpl &rhs) override;

  DFVariableImpl &operator*(DFVariableImpl &rhs) override;

  DFVariableImpl &operator/(DFVariableImpl &rhs) override;

  DFVariableImpl &operator&(DFVariableImpl &rhs) override;

  DFVariableImpl &operator|(DFVariableImpl &rhs) override;

  DFVariableImpl &operator^(DFVariableImpl &rhs) override;

  DFVariableImpl &operator!() override;

  DFVariableImpl &operator-() override;

  DFVariableImpl &operator<(DFVariableImpl &rhs) override;

  DFVariableImpl &operator<=(DFVariableImpl &rhs) override;

  DFVariableImpl &operator>(DFVariableImpl &rhs) override;

  DFVariableImpl &operator>=(DFVariableImpl &rhs) override;

  DFVariableImpl &operator==(DFVariableImpl &rhs) override;

  DFVariableImpl &operator!=(DFVariableImpl &rhs) override;

  DFVariableImpl &operator<<(uint8_t bits) override;

  DFVariableImpl &operator>>(uint8_t bits) override;

  ConstantTypeKind getKind() const;

  int64_t getInt() const;

  uint64_t getUInt() const;

  double getDouble() const;

  bool isConstant() const override;
};

} // namespace dfcxx

#endif // DFCXX_CONST_H
