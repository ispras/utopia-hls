//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/mapper/config/hwconfig.h"
#include "hls/model/model.h"
#include "util/singleton.h"

#include <set>
#include <string>
#include <unordered_map>

using ChanInd = eda::hls::model::ChanInd;
using Chan = eda::hls::model::Chan;
template<typename T>
using Constraint = eda::hls::model::Constraint<T>;
using HWConfig = eda::hls::mapper::HWConfig;
using Indicators = eda::hls::model::Indicators;
using Indicator = eda::hls::model::Indicator;
using NodeType = eda::hls::model::NodeType;
using Node = eda::hls::model::Node;
using Parameters = eda::hls::model::Parameters;
using Parameter = eda::hls::model::Parameter;
using Signature = eda::hls::model::Signature;
template<typename T>
using Singleton = eda::utils::Singleton<T>;
using Type = eda::hls::model::Type;

namespace eda::hls::library {

/**
 * @brief RTL port with name, direction, and width.
 * @author <a href="mailto:grigorovia@ispras.ru">Ivan Grigorov</a>
 */
struct Port {
  enum Direction { IN, OUT, INOUT };
  enum Type { DATA, CLOCK, RESET };

  Port(const std::string &name,
       const Direction direction,
       const unsigned width,
       const model::Parameter &param,
       const Type type = Type::DATA):
    name(name),
    direction(direction),
    width(width),
    param(param),
    type(type) {}

  Port(const Port &port):
    name(port.name),
    direction(port.direction),
    width(port.width),
    param(port.param),
    type(port.type) {}

  const std::string name;
  const Direction direction;
  const unsigned width;
  const Parameter param;
  const Type type;
};

/**
 * @brief Description of a constructed element (module).
 * @author <a href="mailto:grigorovia@ispras.ru">Ivan Grigorov</a>
 */
struct Element final {
  /// TODO: Code, Path, etc.
  explicit Element(const std::vector<Port> &ports): ports(ports) {}

  /// TODO: add mutual relation between spec ports and impl ports.
  const std::vector<Port> ports;

  /// TODO: there should be different IRs: MLIR FIRRTL or Verilog|VHDL.
  std::string ir;
  /// Path to the constructed file.
  std::string path;
};

/**
 * @brief Description of a parameterized constructor of elements.
 * @author <a href="mailto:grigorovia@ispras.ru">Ivan Grigorov</a>
 */
struct MetaElement {
  MetaElement(const std::string &name,
              const std::string &libraryName,
              const bool isCombinational,
              const Parameters &params,
              const std::vector<Port> &ports):
      name(name),
      libraryName(libraryName),
      isCombinational(isCombinational),
      params(params),
      ports(ports) {}
  
  virtual ~MetaElement() = default;

  /**
    * @brief Estimates the indicators with the given set of parameters.
    *
    * @param params Input set of parameters.
    * @param indicators Indicators to be estimated.
    * 
    * @returns Nothing, but estimates the indicators.
    */
  virtual void estimate(const Parameters &params,
                        Indicators &indicators) const = 0;
  
  /**
   * @brief Constructs an element for the given parameters.
   *
   * @returns The constructed element.
   */
  virtual std::unique_ptr<Element> construct() const = 0;

  /// TODO: Needs to be implemented
  bool supports(const HWConfig &hwconfig);

  /// TODO: Add types for MetaElement Ports.
  Signature getSignature();

  const std::string name;
  const std::string libraryName;
  const bool isCombinational;
  const Parameters params;
  const std::vector<Port> ports;
};

/**
 * @brief Entry in the cache of MetaElements.
 * @author <a href="mailto:grigorovia@ispras.ru">Ivan Grigorov</a>
 */
struct StorageEntry {
  StorageEntry(const std::shared_ptr<MetaElement> metaElement,
               const bool isEnabled = true,
               const unsigned int priority = 0):
      metaElement(metaElement), isEnabled(isEnabled), priority(priority) {}
  
  const std::shared_ptr<MetaElement> metaElement;
  bool isEnabled;
  unsigned int priority;
};

/**
 * @brief High-level synthesis library.
 * @author <a href="mailto:grigorovia@ispras.ru">Ivan Grigorov</a>.
 */
class Library final : public Singleton<Library> {
  friend class Singleton<Library>;
public:
  /**
   * @brief Initializes the library by creating standard internal elements.
   *
   * @returns Nothing, but creates standard internal elements and stores them.
   */
  void initialize();

  /**
   * @brief Finalizes the library by clearing the storage of meta-elements.
   *
   * @returns Nothing, but clears storage of meta-elements.
   */
  void finalize();

  /**
   * @brief Search for a meta-elements with a given node type and HW config.
   *
   * @param nodeType Input node type.
   * @param hwconfig Input hardware configuration.
   * 
   * @returns Found/constructed meta-elements.
   */
  std::vector<std::shared_ptr<MetaElement>> find(const NodeType &nodeType,
                                                 const HWConfig &hwconfig);

  /**
   * @brief Imports an IP-XACT library using an IP-XACT catalog.
   *
   * @param libraryPath Path to IP-XACT library.
   * @param catalogPath Path to IP-XACT catalog relative to IP-XACT library.
   * 
   * @returns Nothing, but constructs and stores meta-elements in storage.
   */
  void importLibrary(const std::string &libraryPath,
                     const std::string &catalogPath);

  /**
   * @brief Excludes all elements from the given library from search.
   *
   * @param libraryName Library name.
   * 
   * @returns Nothing, but excludes all elements from library from search.
   */
  void excludeLibrary(const std::string &libraryName);

  /**
   * @brief Includes all elements from the given library in search.
   *
   * @param libraryName Library name.
   * 
   * @returns Nothing, but excludes all elements from library in search.
   */
  void includeLibrary(const std::string &libraryName);

  /**
   * @brief Excludes the given element from the given library from search.
   * 
   * @param elementName Name of element to be excluded.
   * @param libraryName Library name.
   * 
   * @returns Nothing, but excludes all elements from library from search.
   */
  void excludeElementFromLibrary(const std::string &elementName, 
                                 const std::string &libraryName);

  /**
   * @brief Includes the given element from the given library in search.
   * 
   * @param elementName Name of element to be included.
   * @param libraryName Library name.
   * 
   * @returns Nothing, but includes all elements from library in search.
   */
  void includeElementFromLibrary(const std::string &elementName, 
                                 const std::string &libraryName);

private:
  Library() {}

  using LibraryToStorageEntry = std::unordered_map<std::string, StorageEntry>;

  /// Stored meta-elements (second level is for from different libraries).
  std::unordered_map<Signature, LibraryToStorageEntry> groupedMetaElements;
};

} // namespace eda::hls::library