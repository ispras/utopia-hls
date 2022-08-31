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
  name = nodeType.name;
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
    LibraryToStorageEntry metaElementsEntries;
    metaElementsEntries.insert({defaultElement->library,
                                StorageEntry(defaultElement, true)});
    groupedMetaElements.insert({elementKey, metaElementsEntries});
  }
}

void Library::finalize() {
  groupedMetaElements.clear();
}

void Library::importLibrary(const std::string &libraryPath,
                            const std::string &catalogPath) {
  const auto &metaElements = IPXACTParser::get().getDelivery(libraryPath,
                                                             catalogPath);
  for (const auto &metaElement : metaElements) {
    ElementKey elementKey(metaElement);
    auto iterator = groupedMetaElements.find(elementKey);
    if (iterator == groupedMetaElements.end()) {
      LibraryToStorageEntry metaElementsEntries;
      metaElementsEntries.insert({metaElement->library,
                                  StorageEntry(metaElement, true)});
      groupedMetaElements.insert({elementKey, metaElementsEntries});
    } else {
      auto &metaElementsEntries = iterator->second;
      auto status = metaElementsEntries.insert({metaElement->library, 
                                                StorageEntry(metaElement,
                                                             true)});
      if (!status.second) {
        std::cout << "Element " << metaElement->name << " from "
                  << metaElement->library << " has been already imported!" 
                  << std::endl;
      }
    }
  }
}

void Library::excludeLibrary(const std::string &libraryName) {
  for (auto groupIterator = groupedMetaElements.begin();
       groupIterator != groupedMetaElements.end();
       groupIterator++) {
    auto entryIterator = groupIterator->second.find(libraryName);
    if (entryIterator != groupIterator->second.end()) {
      entryIterator->second.isEnabled = false;
    }
  }
}

void Library::includeLibrary(const std::string &libraryName) {
   for (auto groupIterator = groupedMetaElements.begin();
       groupIterator != groupedMetaElements.end();
       groupIterator++) {
    auto entryIterator = groupIterator->second.find(libraryName);
    if (entryIterator != groupIterator->second.end()) {
      entryIterator->second.isEnabled = true;
    }
  }
}

void Library::excludeElementFromLibrary(const std::string &elementName, 
                                        const std::string &libraryName) {
  for (auto groupIterator = groupedMetaElements.begin();
       groupIterator != groupedMetaElements.end();
       groupIterator++) {
    auto entryIterator = groupIterator->second.find(libraryName);
    if (entryIterator != groupIterator->second.end() &&
        entryIterator->second.metaElement->name == elementName) {
      entryIterator->second.isEnabled = false;
    }
  }
}

void Library::includeElementFromLibrary(const std::string &elementName, 
                                        const std::string &libraryName) {
  for (auto groupIterator = groupedMetaElements.begin();
       groupIterator != groupedMetaElements.end();
       groupIterator++) {
    auto entryIterator = groupIterator->second.find(libraryName);
    if (entryIterator != groupIterator->second.end() &&
        entryIterator->second.metaElement->name == elementName) {
      entryIterator->second.isEnabled = true;
    }
  }
}


std::vector<std::shared_ptr<MetaElement>> Library::find(
  const NodeType &nodeType,
  const HWConfig &hwconfig) {
  ElementKey elementKey(nodeType);
  std::vector<std::shared_ptr<MetaElement>> metaElements;
  auto groupIterator = groupedMetaElements.find(elementKey);
  if (groupIterator != groupedMetaElements.end()) {
    for (auto entryIterator = groupIterator->second.begin();
         entryIterator != groupIterator->second.end();
         entryIterator++) {
      if (entryIterator->second.isEnabled && 
          entryIterator->second.metaElement->supports(hwconfig)) {
        metaElements.push_back(entryIterator->second.metaElement);
      }
    }
  } else {
    LibraryToStorageEntry metaElementsEntries;
    groupedMetaElements.insert({elementKey, metaElementsEntries});
  }
  groupIterator = groupedMetaElements.find(elementKey);
  auto entryIterator = groupIterator->second.find("std");
  if (entryIterator == groupIterator->second.end()) {
    auto metaElement = ElementInternal::create(nodeType, hwconfig);
    groupIterator->second.insert({"std", StorageEntry(metaElement,
                                                      true)});
    metaElements.push_back(metaElement);
  }
  return metaElements;
}

} // namespace eda::hls::library
