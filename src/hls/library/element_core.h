//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"

#include <memory>
#include <string>
#include <vector>

namespace eda::hls::library {

struct ElementCore final : public MetaElement {
  ElementCore(const std::string &name,
              const Parameters &params,
              const std::vector<Port> &ports,
              const std::string &path) :
    MetaElement { name, params, ports }, path { path } {}
  virtual ~ElementCore() = default;
  virtual void estimate(const Parameters &params,
                        Indicators &indicators) const override;

  virtual std::unique_ptr<Element> construct(
    const Parameters &params) const override;

  /// Path to the element implementation.
  const std::string path;
};

} // namespace eda::hls::library
