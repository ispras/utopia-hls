//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/ril/add.h" 
#include "hls/library/internal/ril/element_internal_ril.h"
#include "hls/library/internal/ril/mul.h"
#include "hls/library/internal/ril/sub.h"

namespace eda::hls::library::internal::ril {

SharedMetaElement ElementInternalRil::create(const NodeType &nodeType,
                                             const HWConfig &hwconfig) {
  std::string name = nodeType.name;
  SharedMetaElement metaElement;
  if (Add::isAdd(nodeType)) {
    metaElement = Add::create(nodeType, hwconfig);
  } else if (Mul::isMul(nodeType)) {
    metaElement = Mul::create(nodeType, hwconfig);
  } else if (Sub::isSub(nodeType)) {
    metaElement = Sub::create(nodeType, hwconfig);
  }
  return metaElement;
}

} // namespace eda::hls::library::internal::ril