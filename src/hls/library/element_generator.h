//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "util/assert.h"

#include <string>
#include <vector>

namespace eda::hls::library {

struct ElementGenerator final : public MetaElement {
  ElementGenerator(const std::string &name,
                   const std::string &library,
                   const Parameters &params,
                   const std::vector<Port> &ports,
                   const std::string &genPath) :
    MetaElement { name, library, params, ports }, genPath { genPath } {}
  virtual ~ElementGenerator() = default;
  virtual void estimate(
      const Parameters &params, Indicators &indicators) const override;
  virtual std::unique_ptr<Element> construct(
      const Parameters &params) const override;
  const std::string genPath;
};

} // namespace eda::hls::library
