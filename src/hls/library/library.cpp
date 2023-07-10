//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"

#include "hls/library/internal/verilog/element_internal_verilog.h"
#include "hls/library/ipxact_parser.h"
#include "hls/mapper/config/hwconfig.h"
#include "utils/assert.h"

using ElementInternalVerilog = 
    eda::hls::library::internal::verilog::ElementInternalVerilog;

namespace eda::hls::library {

Signature MetaElement::getSignature() {
  std::vector<std::string> inputTypeNames;
  std::vector<std::string> outputTypeNames;
  for (const auto &port : ports) {
    if (port.direction == Port::Direction::IN) {
      if (port.type == Port::Type::DATA) {   
        inputTypeNames.push_back("");
      }
    } else {
      outputTypeNames.push_back("");
    }
  }
  return Signature(name,
                   inputTypeNames,
                   outputTypeNames);
}

/// TODO: Discuss how we plan to describe supported config: include or exclude
bool MetaElement::supports(const HWConfig &hwconfig) {
  return true;
}

void Library::initialize() {
  const auto defaultElements = ElementInternalVerilog::createDefaultElements();
  for (const auto &defaultElement : defaultElements) {
    std::string name;
    std::vector<std::string> inputTypeNames;
    std::vector<std::string> outputTypeNames;
    Signature signature(name, inputTypeNames, outputTypeNames);
    LibraryToStorageEntry metaElementsEntries;
    metaElementsEntries.insert({defaultElement->libraryName,
                                StorageEntry(defaultElement)});
    groupedMetaElements.insert({signature, metaElementsEntries});
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
    std::string name;
    Signature signature = metaElement->getSignature();
    auto iterator = groupedMetaElements.find(signature);
    if (iterator == groupedMetaElements.end()) {
      LibraryToStorageEntry metaElementsEntries;
      metaElementsEntries.insert({metaElement->libraryName,
                                  StorageEntry(metaElement)});
      groupedMetaElements.insert({signature, metaElementsEntries});
    } else {
      auto &metaElementsEntries = iterator->second;
      auto status = metaElementsEntries.insert({metaElement->libraryName,
                                                StorageEntry(metaElement)});
      if (!status.second) {
        std::cout << "Element " << metaElement->name << " from "
                  << metaElement->libraryName << " has been already imported!"
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
  Signature signature(nodeType);
  std::vector<std::shared_ptr<MetaElement>> metaElements;
  auto groupIterator = groupedMetaElements.find(signature);
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
    groupedMetaElements.insert({signature, metaElementsEntries});
  }
  groupIterator = groupedMetaElements.find(signature);
  auto entryIterator = groupIterator->second.find("std");
  if (entryIterator == groupIterator->second.end()) {
    auto metaElement = ElementInternalVerilog::create(nodeType, hwconfig);
    groupIterator->second.insert({"std", StorageEntry(metaElement)});
    metaElements.push_back(metaElement);
  }
  return metaElements;
}

} // namespace eda::hls::library