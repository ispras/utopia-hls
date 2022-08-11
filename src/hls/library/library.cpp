//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/ipxact_parser.h"
#include "hls/library/library.h"
#include "hls/mapper/config/hwconfig.h"
#include "util/assert.h"

using namespace eda::hls::mapper;
using namespace eda::hls::library;

bool ElementKey::operator==(const ElementKey &elementKey) const {
  if (name != elementKey.name ||
      inputs.size() != elementKey.inputs.size() ||
      outputs.size() != elementKey.outputs.size()) {
    return false;
  }
  for (const auto &input : inputs) {
    if (elementKey.inputs.find(input) == elementKey.inputs.end()) {
      return false;
    }
  }
  for (const auto &output : outputs) {
    if (elementKey.outputs.find(output) == elementKey.outputs.end()) {
      return false;
    }
  }
  return true;
}

namespace eda::hls::library {

ElementKey::ElementKey(const NodeType &nodeType) {
  name = toLower(nodeType.name);
  for (const auto *input : nodeType.inputs) {
    inputs.insert(input->name);
  }
  for (const auto *output : nodeType.outputs) {
    outputs.insert(output->name);
  }
}

ElementKey::ElementKey(const std::shared_ptr<MetaElement> metaElement) {
  name = metaElement->name;
  for (const auto &port : metaElement->ports) {
    if (port.direction == Port::Direction::IN) {
      if (port.name != "clock" && port.name != "reset") {
        inputs.insert(port.name);
      }
    } else {
      outputs.insert(port.name);
    }
  }
}

//TODO
bool MetaElement::supports(const HWConfig &hwconfig) {
  return true;
}

void Library::initialize() {
  const auto defaultElements = ElementInternal::createDefaultElements();
  for (const auto &defaultElement : defaultElements) {
    ElementKey elementKey(defaultElement);
    storage.insert({elementKey, StorageEntry(StorageEntry(defaultElement,
                                                          true))});
  }
}

void Library::finalize() {
  storage.clear();
}

void Library::importLibrary(const std::string &libraryPath,
                            const std::string &catalogPath) {
  const auto &metaElements = IPXACTParser::get().getDelivery(libraryPath,
                                                             catalogPath);
  for (const auto &metaElement : metaElements) {
    ElementKey elementKey(metaElement);
    storage.insert({elementKey, StorageEntry(StorageEntry(metaElement, true))});
  }
}

void Library::excludeLibrary(const std::string &libraryName) {
  for (auto &storageEntry: storage) {
    if (storageEntry.second.metaElement->library == libraryName) {
      storageEntry.second.isEnabled = false;
    }
  }
}

void Library::includeLibrary(const std::string &libraryName) {
  for (auto &storageEntry: storage) {
    if (storageEntry.second.metaElement->library == libraryName) {
      storageEntry.second.isEnabled = true;
    }
  }
}

void Library::excludeElementFromLibrary(const std::string &elementName, 
                                        const std::string &libraryName) {
  for (auto &storageEntry: storage) {
    if (storageEntry.second.metaElement->library == libraryName &&
        storageEntry.second.metaElement->name    == elementName) {
      storageEntry.second.isEnabled = false;
    }
  }
}

void Library::includeElementFromLibrary(const std::string &elementName, 
                                        const std::string &libraryName) {
  for (auto &storageEntry: storage) {
    if (storageEntry.second.metaElement->library == libraryName &&
        storageEntry.second.metaElement->name    == elementName) {
      storageEntry.second.isEnabled = true;
    }
  }
}

std::shared_ptr<MetaElement> Library::find(const NodeType &nodeType,
                                           const HWConfig &hwconfig) {
  ElementKey elementKey(nodeType);
  //TODO: Proper diagnostics
  if (storage.count(elementKey) > 0 &&
      storage.find(elementKey)->second.isEnabled &&
      storage.find(elementKey)->second.metaElement->supports(hwconfig)) {
    return storage.find(elementKey)->second.metaElement;
  }
  auto metaElement = ElementInternal::create(nodeType, hwconfig);
  storage.insert({elementKey, StorageEntry(StorageEntry(metaElement, true))});
  return metaElement;
}
} // namespace eda::hls::library
