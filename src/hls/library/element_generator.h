//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
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
                   const std::string &libraryName,
                   const bool isCombinational,
                   const Parameters &params,
                   const std::vector<Port> &ports,
                   const std::string &genPath) :
    MetaElement { name, libraryName, isCombinational, params, ports },
        genPath { genPath } {}

  virtual ~ElementGenerator() = default;

  virtual void estimate(const Parameters &params,
                        Indicators &indicators) const override;

  virtual std::unique_ptr<Element> construct() const override;

  /// Path to the generator.
  const std::string genPath;
};

} // namespace eda::hls::library