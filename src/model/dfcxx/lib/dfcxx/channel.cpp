//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/channel.h"

namespace dfcxx {

Channel::Channel(Node *source, Node* target, unsigned opInd) : source(source),
                                                               target(target),
                                                               opInd(opInd) {}

bool Channel::operator==(const Channel &channel) const {
  return source == channel.source &&
         target == channel.target &&
         opInd == channel.opInd;
}

} // namespace dfcxx
