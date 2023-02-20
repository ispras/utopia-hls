//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/internal/ril/element_internal_ril.h"
#include "hls/library/library.h"

namespace eda::hls::library::internal::ril {

struct Add final : public ElementInternalRil {
  static constexpr const char *stages = "stages";

  Add(const std::string &name,
      const std::string &libraryName,
      const bool isCombinational,
      const Parameters &params,
      const std::vector<Port> &ports) :
  ElementInternalRil(name, libraryName, isCombinational, params, ports) {}
  virtual ~Add() = default;

  virtual void estimate(const Parameters &params,
                        Indicators &indicators) const override;
  virtual std::unique_ptr<Element> construct() const override;
  static SharedMetaElement create(const NodeType &nodetype,
                                  const HWConfig &hwconfig);
  static bool isAdd(const NodeType &nodeType);
};
} // namespace eda::hls::library::internal::ril
