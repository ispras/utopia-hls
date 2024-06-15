#ifndef DFCXX_TYPEDEFS_H
#define DFCXX_TYPEDEFS_H

#include <unordered_map>

namespace dfcxx {

enum Ops {
  ADD_INT = 1,
  ADD_FLOAT,
  SUB_INT,
  SUB_FLOAT,
  MUL_INT,
  MUL_FLOAT,
  DIV_INT,
  DIV_FLOAT,
  AND_INT,
  AND_FLOAT,
  OR_INT,
  OR_FLOAT,
  XOR_INT,
  XOR_FLOAT,
  NOT_INT,
  NOT_FLOAT,
  NEG_INT,
  NEG_FLOAT,
  LESS_INT,
  LESS_FLOAT,
  LESS_EQ_INT,
  LESS_EQ_FLOAT,
  MORE_INT,
  MORE_FLOAT,
  MORE_EQ_INT,
  MORE_EQ_FLOAT,
  EQ_INT,
  EQ_FLOAT,
  NEQ_INT,
  NEQ_FLOAT
};

enum Scheduler {
  Linear = 0,
  Dijkstra
};

} // namespace dfcxx

typedef std::unordered_map<dfcxx::Ops, unsigned> DFLatencyConfig;

#endif // DFCXX_TYPEDEFS_H
