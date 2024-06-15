#include "dfcxx/channel.h"

namespace dfcxx {

Channel::Channel(Node source, Node target, unsigned opInd) : source(source),
                                                             target(target),
                                                             opInd(opInd) {}

bool Channel::operator==(const Channel &channel) const {
  return source == channel.source && target == channel.target &&
         opInd == channel.opInd;
}

} // namespace dfcxx