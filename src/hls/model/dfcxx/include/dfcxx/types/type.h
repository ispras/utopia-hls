#ifndef DFCXX_TYPE_H
#define DFCXX_TYPE_H

#include <cstdint>

namespace dfcxx {

class DFType;
class TypeBuilder;
class DFCIRTypeConverter;
class DFCIRBuilder;

class DFTypeImpl {
  friend DFType;
  friend TypeBuilder;
  friend DFCIRTypeConverter;
  friend DFCIRBuilder;

public:
  virtual ~DFTypeImpl() = default;

  virtual uint16_t getTotalBits() const = 0;

  virtual bool operator==(const DFTypeImpl &rhs) const = 0;

  bool operator!=(const DFTypeImpl &rhs) const;

  virtual bool isFixed() const;

  virtual bool isFloat() const;
};

class DFType {
private:
  DFTypeImpl *impl;

public:
  DFType(DFTypeImpl *impl);

  DFTypeImpl *getImpl() const;

  uint16_t getTotalBits() const;

  bool operator==(const DFType &rhs) const;

  bool operator!=(const DFType &rhs) const;

  bool isFixed() const;

  bool isFloat() const;
};

} // namespace dfcxx

#endif // DFCXX_TYPE_H
