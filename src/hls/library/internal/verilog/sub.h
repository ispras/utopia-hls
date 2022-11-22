//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/internal/verilog/element_internal_verilog.h"
#include "hls/library/library.h"

namespace eda::hls::library::internal::verilog {

struct Sub final : public ElementInternalVerilog {
  static constexpr const char *stages = "stages";

  Sub(const std::string &name,
      const std::string &libraryName, 
      const Parameters &params,
      const std::vector<Port> &ports) :
  ElementInternalVerilog(name, libraryName, params, ports) {}
  virtual ~Sub() = default;

  virtual void estimate(const Parameters &params, 
                        Indicators &indicators) const override;
  virtual std::unique_ptr<Element> construct() const override;
  static std::shared_ptr<MetaElement> create(const NodeType &nodetype,
                                             const HWConfig &hwconfig);
  static bool isSub(const NodeType &nodeType);
};
} // namespace eda::hls::library::internal::verilog