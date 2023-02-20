//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/verilog/add.h"
#include "hls/library/internal/verilog/cast.h" 
#include "hls/library/internal/verilog/clip.h" 
#include "hls/library/internal/verilog/const.h"
#include "hls/library/internal/verilog/default.h"
#include "hls/library/internal/verilog/delay.h"
#include "hls/library/internal/verilog/dup.h"
#include "hls/library/internal/verilog/element_internal_verilog.h"
#include "hls/library/internal/verilog/eq.h"
#include "hls/library/internal/verilog/ge.h"
#include "hls/library/internal/verilog/gt.h"
#include "hls/library/internal/verilog/le.h"
#include "hls/library/internal/verilog/lt.h"
#include "hls/library/internal/verilog/merge.h"
#include "hls/library/internal/verilog/mul.h"
#include "hls/library/internal/verilog/mux.h" 
#include "hls/library/internal/verilog/ne.h"
#include "hls/library/internal/verilog/shl11.h"
#include "hls/library/internal/verilog/shl8.h"
#include "hls/library/internal/verilog/shr14.h" 
#include "hls/library/internal/verilog/shr3.h"  
#include "hls/library/internal/verilog/shr8.h"
#include "hls/library/internal/verilog/sink.h"
#include "hls/library/internal/verilog/source.h"
#include "hls/library/internal/verilog/split.h"  
#include "hls/library/internal/verilog/sub.h" 

namespace eda::hls::library::internal::verilog {

using SharedMetaElements = std::vector<std::shared_ptr<MetaElement>>;
using SharedMetaElement = std::shared_ptr<MetaElement>;

SharedMetaElement ElementInternalVerilog::create(const NodeType &nodeType,
                                                 const HWConfig &hwconfig) {
  std::string name = nodeType.name;
  SharedMetaElement metaElement;
  if (Delay::isDelay(nodeType)) {
    metaElement = Delay::create(nodeType, hwconfig);
  } else if (Dup::isDup(nodeType)) {
    metaElement = Dup::create(nodeType, hwconfig);
  } else if (Const::isConst(nodeType)) {
    metaElement = Const::create(nodeType, hwconfig);
  } else if (Merge::isMerge(nodeType)) {
    metaElement = Merge::create(nodeType, hwconfig);
  } else if (Split::isSplit(nodeType)) {
    metaElement = Split::create(nodeType, hwconfig);
  } else if (Mux::isMux(nodeType)) {
    metaElement = Mux::create(nodeType, hwconfig);
  } else if (Cast::isCast(nodeType)) {
    metaElement = Cast::create(nodeType, hwconfig);
  } else if (Add::isAdd(nodeType)) {
    metaElement = Add::create(nodeType, hwconfig);
  } else if (Mul::isMul(nodeType)) {
    metaElement = Mul::create(nodeType, hwconfig);
  } else if (Sub::isSub(nodeType)) {
    metaElement = Sub::create(nodeType, hwconfig);
  } else if (Eq::isEq(nodeType)) {
    metaElement = Eq::create(nodeType, hwconfig);
  } else if (Ne::isNe(nodeType)) {
    metaElement = Ne::create(nodeType, hwconfig);
  } else if (Gt::isGt(nodeType)) {
    metaElement = Gt::create(nodeType, hwconfig);
  } else if (Lt::isLt(nodeType)) {
    metaElement = Lt::create(nodeType, hwconfig);
  } else if (Le::isLe(nodeType)) {
    metaElement = Le::create(nodeType, hwconfig);
  } else if (Ge::isGe(nodeType)) {
    metaElement = Ge::create(nodeType, hwconfig);
  } else if (Shr3::isShr3(nodeType)) {
    metaElement = Shr3::create(nodeType, hwconfig);
  } else if (Shr14::isShr14(nodeType)) {
    metaElement = Shr14::create(nodeType, hwconfig);
  } else if (Shr8::isShr8(nodeType)) {
    metaElement = Shr8::create(nodeType, hwconfig);
  } else if (Shl11::isShl11(nodeType)) {
    metaElement = Shl11::create(nodeType, hwconfig);
  } else if (Shl8::isShl8(nodeType)) {
    metaElement = Shl8::create(nodeType, hwconfig);
  } else if (Source::isSource(nodeType)) {
    metaElement = Source::create(nodeType, hwconfig);
  } else if (Sink::isSink(nodeType)) {
    metaElement = Sink::create(nodeType, hwconfig);
  } else {
    std::cout << "Warning: Default MetaElement is created for: " << 
        nodeType.name << std::endl;
    metaElement = Default::create(nodeType, hwconfig);
  }
  //TODO: discuss whether default MetaElement is needed
  /*uassert(metaElement != nullptr, "Unsupported MetaElement for NodeType: " +
                                    nodeType.name);*/

  return metaElement;
}

SharedMetaElements ElementInternalVerilog::createDefaultElements() {
  SharedMetaElements defaultElements; 
  defaultElements.push_back(Clip::createDefaultElement());
  defaultElements.push_back(Merge::createDefaultElement());
  defaultElements.push_back(Split::createDefaultElement());
  defaultElements.push_back(Dup::createDefaultElement());
  defaultElements.push_back(Mux::createDefaultElement());
  return defaultElements;
}
} // namespace eda::hls::library::internal::verilog
