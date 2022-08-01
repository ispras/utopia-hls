//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/mapper/config/hwconfig.h"
#include "hls/model/model.h"
#include "util/singleton.h"

#include <set>
#include <string>

using namespace eda::hls::mapper;
using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::library {
  /// Key for MetaElement / nodetype.
struct ElementKey {
  ElementKey(const NodeType &nodetype);
  ElementKey(const std::shared_ptr<MetaElement> metaElement);
  bool operator==(const ElementKey &elementKey) const;
  std::string name;
  std::set<std::string> inputs;
  std::set<std::string> outputs;
};
} // namespace eda::hls::library

namespace std {
template<>
struct hash<eda::hls::library::ElementKey> {
  size_t operator()(const eda::hls::library::ElementKey &element) const {
    size_t hash = std::hash<std::string>()(element.name);
    for (const auto &input : element.inputs) {
      hash = hash * 13 + std::hash<std::string>()(input);
    }
    for (const auto &output : element.outputs) {
      hash = hash * 13 + std::hash<std::string>()(output);
    }
    return hash;
  }
};
} // namespace std

namespace eda::hls::library {

/// RTL port with name, direction, and width.
struct Port {
  enum Direction { IN, OUT, INOUT };

  Port(const std::string &name,
       const Direction &direction,
       const unsigned width,
       const Parameter param):
    name(name),
    direction(direction),
    width(width),
    param(param) {}
  Port(const Port &port):
    name(port.name),
    direction(port.direction),
    width(port.width),
    param(port.param) {}

  const std::string name;
  const Direction direction;
  const unsigned width;
  const Parameter param;
};

/// Description of a constructed element (module).
struct Element final {
  // TODO: Code, Path, etc.
  explicit Element(const std::vector<Port> &ports): ports(ports) {}

  // TODO add mutual relation between spec ports and impl ports
  const std::vector<Port> ports;

  // TODO there should be different IRs: MLIR FIRRTL or Verilog|VHDL described in FIRRTL
  std::string ir;

  // TODO path
  std::string path;
};

/// Description of a parameterized constructor of elements.
struct MetaElement {
  MetaElement(const std::string &name,
              const std::string &library,
              const Parameters &params,
              const std::vector<Port> &ports):
      name(name),
      library(library),
      libName(toLibName(name)),
      params(params),
      ports(ports) {}

  // TODO: discuss naming conventions
  static const std::string toLibName(const std::string &name) {
    return name.substr(0, name.find("_"));
  }

  /// Estimates the indicators the given set of parameters.
  virtual void estimate(const Parameters &params,
                        Indicators &indicators) const = 0;

  virtual std::unique_ptr<Element> construct(const Parameters &params) const = 0;

  virtual ~MetaElement() = default;

  bool supports(const HWConfig &hwconfig);

  const std::string name;
  const std::string library;
  const std::string libName;
  const Parameters params;
  const std::vector<Port> ports;
};

/// Entry in cache of MetaElements
struct StorageEntry {
  StorageEntry(const std::shared_ptr<MetaElement> metaElement,
               const bool isEnabled,
               const unsigned int priority = 0):
      metaElement(metaElement), isEnabled(isEnabled), priority(priority) {}
  
  const std::shared_ptr<MetaElement> metaElement;
  bool isEnabled;
  unsigned int priority;
};

class Library final : public Singleton<Library> {
  friend class Singleton<Library>;

public:
/**
 * @brief Initializes the library by creating standard internal elements.
 *
 * @returns Nothing, but creates a set of standard internal elements.
 */
  void initialize();
/**
 * @brief Finalizes the library by clearing storage of meta-elements.
 *
 * @returns Nothing, but clears storage of meta-elements.
 */
  void finalize();

/**
 * @brief Search meta-element for given node type and hardware configuration.
 *
 * @param nodeType Input node type.
 * @param hwconfig Input hardware configuration.
 * @returns Found/constructed meta-element.
 */
  std::shared_ptr<MetaElement> find(const NodeType &nodeType,
                                    const HWConfig &hwconfig);

/**
 * @brief Imports an IP-XACT library using IP-XACT catalog.
 *
 * @param libraryPath Path to IP-XACT library.
 * @param catalogPath Path to IP-XACT catalog relative to the IP-XACT library.
 * @returns Nothing, but creates and stores the meta-elements in the storage.
 */
  void importLibrary(const std::string &libraryPath,
                     const std::string &catalogPath);

/**
 * @brief Excludes all elements from the given library from search.
 *
 * @param libraryPath Library name.
 * @returns Nothing, but excludes all elements from the library from search.
 */
  void excludeLibrary(const std::string &libraryName);

/**
 * @brief Includes all elements from the given library in search.
 *
 * @param libraryPath Library name.
 * @returns Nothing, but excludes all elements from the library in search.
 */
  void includeLibrary(const std::string &libraryName);

/**
 * @brief Excludes the given element from the given library from search.
 * 
 * @param elementName Name of the element to be excluded.
 * @param libraryPath Library name.
 * @returns Nothing, but excludes all elements from the library from search.
 */
  void excludeElementFromLibrary(const std::string &elementName, 
                                 const std::string &libraryName);

/**
 * @brief Includes the given element from the given library in search.
 * 
 * @param elementName Name of the element to be included.
 * @param libraryPath Library name.
 * @returns Nothing, but includes all elements from the library in search.
 */
  void includeElementFromLibrary(const std::string &elementName, 
                                 const std::string &libraryName);

  /*void add(const std::shared_ptr<MetaElement> &metaElement) {
    cache.push_back(metaElement);
  }*/

private:
  Library() {}

  /// Stored meta-elements.
  std::unordered_map<ElementKey, StorageEntry> storage;
};

} // namespace eda::hls::library

