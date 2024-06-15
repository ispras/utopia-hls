#ifndef DFCXX_CHANNEL_H
#define DFCXX_CHANNEL_H

#include "dfcxx/node.h"

namespace dfcxx {

struct Channel {
  Node source;
  Node target;
  unsigned opInd;

  Channel(Node source, Node target, unsigned opInd);

  bool operator==(const Channel &channel) const;
};

} // namespace dfcxx

template <>
struct std::hash<dfcxx::Channel> {
  size_t operator()(const dfcxx::Channel &ch) const noexcept {
    return std::hash<dfcxx::DFVariableImpl *>()(ch.source.var) *
           std::hash<dfcxx::DFVariableImpl *>()(ch.target.var);
  }
};

#endif // DFCXX_CHANNEL_H
