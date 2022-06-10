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

#include <memory>
#include <string>
#include <vector>

namespace eda::hls::library {

struct ElementInternal : public MetaElement {
  ElementInternal(const std::string &name,
                  const Parameters &params,
                  const std::vector<Port> &ports) :
  MetaElement(name, params, ports) {}
  virtual std::unique_ptr<Element> construct(
      const Parameters &params) const override;
  virtual void estimate(
      const Parameters &params, Indicators &indicators) const override;
};

} // namespace eda::hls::library
