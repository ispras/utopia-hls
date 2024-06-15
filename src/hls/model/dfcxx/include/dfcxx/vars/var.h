#ifndef DFCXX_VAR_H
#define DFCXX_VAR_H

#include "dfcxx/types/type.h"

#include <string_view>
#include <string>

namespace dfcxx {

enum IODirection{
  NONE = 0,
  INPUT,
  OUTPUT
};

class GraphHelper;
class DFVariable;
class VarBuilder;
class DFCIRTypeConverter;
class DFCIRBuilder;

class DFVariableImpl {
  friend DFVariable;
  friend VarBuilder;
  friend DFCIRTypeConverter;
  friend DFCIRBuilder;

private:
  IODirection direction;
  std::string name;

public:
  virtual ~DFVariableImpl() = default;

  virtual bool isStream() const;

  virtual bool isScalar() const;

  virtual bool isConstant() const;

protected:
  GraphHelper &helper;

  DFVariableImpl(const std::string &name, IODirection direction, GraphHelper &helper);

  std::string_view getName() const;

  IODirection getDirection() const;

  virtual DFTypeImpl &getType() = 0;

  virtual DFVariableImpl &operator+(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator-(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator*(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator/(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator&(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator|(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator^(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator!() = 0;

  virtual DFVariableImpl &operator-() = 0;

  virtual DFVariableImpl &operator<(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator<=(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator>(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator>=(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator==(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator!=(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl &operator<<(uint8_t bits) = 0;

  virtual DFVariableImpl &operator>>(uint8_t bits) = 0;

  void connect(DFVariableImpl &connectee);
};

class DFVariable {
private:
  DFVariableImpl *impl;

public:
  DFVariable(DFVariableImpl *impl);

  DFVariableImpl *getImpl() const;

  std::string_view getName() const;

  IODirection getDirection() const;

  DFType getType() const;

  DFVariable operator+(const DFVariable &rhs);

  DFVariable operator-(const DFVariable &rhs);

  DFVariable operator*(const DFVariable &rhs);

  DFVariable operator/(const DFVariable &rhs);

  DFVariable operator&(const DFVariable &rhs);

  DFVariable operator|(const DFVariable &rhs);

  DFVariable operator^(const DFVariable &rhs);

  DFVariable operator!();

  DFVariable operator-();

  DFVariable operator<(const DFVariable &rhs);

  DFVariable operator<=(const DFVariable &rhs);

  DFVariable operator>(const DFVariable &rhs);

  DFVariable operator>=(const DFVariable &rhs);

  DFVariable operator==(const DFVariable &rhs);

  DFVariable operator!=(const DFVariable &rhs);

  DFVariable operator<<(uint8_t bits);

  DFVariable operator>>(uint8_t bits);

  bool isStream() const;

  bool isScalar() const;

  bool isConstant() const;

  void connect(const DFVariable &connectee);
};

} // namespace dfcxx

#endif // DFCXX_VAR_H
