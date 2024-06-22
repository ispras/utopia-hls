#ifndef DFCXX_STREAM_H
#define DFCXX_STREAM_H

#include "dfcxx/vars/var.h"

namespace dfcxx {

class VarBuilder;
class DFCIRBuilder;

class DFStream : DFVariableImpl {
  friend VarBuilder;
  friend DFCIRBuilder;
private:
  DFTypeImpl &type;

  DFStream(const std::string &name, IODirection direction, GraphHelper &helper,
           DFTypeImpl &type);

public:
  ~DFStream() override = default;

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

  bool isStream() const override;
};

} // namespace dfcxx

#endif // DFCXX_STREAM_H