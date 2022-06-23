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

#include <cassert>
#include <map>
#include <memory>
#include <string>

using namespace eda::hls::mapper::config::hwconfig;
using namespace eda::hls::model;
using namespace eda::util;

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
              const Parameters &params,
              const std::vector<Port> &ports):
      name(name), params(params), ports(ports) {}

  /// Estimates the indicators the given set of parameters.
  virtual void estimate(const Parameters &params,
                        Indicators &indicators) const = 0;

  virtual std::unique_ptr<Element> construct(const Parameters &params) const = 0;

  virtual ~MetaElement() = default;

  bool supports(const HWConfig &hwconfig);

  const std::string name;
  const Parameters params;
  const std::vector<Port> ports;
};

class Library final : public Singleton<Library> {
  friend class Singleton<Library>;

public:
  void initialize(const std::string &libraryPath,
                  const std::string &catalogPath);
  void finalize();

  /// Searches for a meta-element for the given node type and HWConfig.
  std::shared_ptr<MetaElement> find(const NodeType &nodetype,
                                    const HWConfig &hwconfig);

  /// Searches for a meta-element for the given name.
  //std::shared_ptr<MetaElement> find(const std::string &name);
  void importLibrary(const std::string &libraryPath,
                     const std::string &catalogPath);

  void add(const std::shared_ptr<MetaElement> &metaElement) {
    cache.push_back(metaElement);
  }

private:
  Library() {}

  /// Cached meta-elements.
  std::vector<std::shared_ptr<MetaElement>> cache;
};

} // namespace eda::hls::library
