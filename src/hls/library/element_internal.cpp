//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/internal/cast.h" 
#include "hls/library/internal/clip.h" 
#include "hls/library/internal/const.h" 
#include "hls/library/internal/default.h"
#include "hls/library/internal/delay.h"
#include "hls/library/internal/dup.h"
#include "hls/library/internal/merge.h"
#include "hls/library/internal/mux.h" 
#include "hls/library/internal/split.h"  

namespace eda::hls::library {

std::vector<Port> ElementInternal::createPorts(const NodeType &nodetype) {
  std::vector<Port> ports;

  ports.push_back(Port("clock",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1)));

  ports.push_back(Port("reset",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1)));

  for (const auto *input: nodetype.inputs) {
    ports.push_back(Port(input->name,
                         Port::IN,
                         input->type.size,
                         model::Parameter(std::string("width"),
                                          input->type.size)));
  }

  for (const auto *output: nodetype.outputs) {
    ports.push_back(Port(output->name,
                         Port::OUT,
                         output->type.size,
                         model::Parameter(std::string("width"),
                                          output->type.size)));
  }
  return ports;
}

std::shared_ptr<MetaElement> ElementInternal::create(const NodeType &nodetype,
                                                     const HWConfig &hwconfig) {
  std::string name = nodetype.name;
  std::shared_ptr<MetaElement> metaElement;
  if (nodetype.isDelay()) {
    metaElement = Delay::create(nodetype, hwconfig);
  } else if (nodetype.isDup()) {
    metaElement = Dup::create(nodetype, hwconfig);
  } else if (nodetype.isConst()) {
    metaElement = Const::create(nodetype, hwconfig);
  } else if (nodetype.isMerge()) {
    metaElement = Merge::create(nodetype, hwconfig);
  } else if (nodetype.isSplit()) {
    metaElement = Split::create(nodetype, hwconfig);
  } else if (nodetype.isMux()) {
    metaElement = Mux::create(nodetype, hwconfig);
  } else if (nodetype.isClip()) {
    metaElement = Clip::create(nodetype, hwconfig);
  } else if (nodetype.isCast()) {
    metaElement = Cast::create(nodetype, hwconfig);
  } else {
    metaElement = Default::create(nodetype, hwconfig);
  }
  return metaElement;
}

std::vector<std::shared_ptr<MetaElement>> ElementInternal::createDefaultElements() {
  std::vector<std::shared_ptr<MetaElement>> defaultElements; 
  defaultElements.push_back(Clip::createDefaultElement());
  defaultElements.push_back(Merge::createDefaultElement());
  defaultElements.push_back(Split::createDefaultElement());
  defaultElements.push_back(Dup::createDefaultElement());
  defaultElements.push_back(Mux::createDefaultElement());
  return defaultElements;
}

} // namespace eda::hls::library
