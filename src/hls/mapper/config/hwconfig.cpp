//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/mapper/config/hwconfig.h"

#include <string>

namespace eda::hls::mapper {

HWConfig::HWConfig(const std::string &name, const std::string &family, const std::string &vendor) :
  name(name), family(family), vendor(vendor) {};

std::string HWConfig::getName() {
  return name;
}

std::string HWConfig::getFamily() {
  return family;
}

std::string HWConfig::getVendor() {
  return vendor;
}

} // namespace eda::hls::mapper
