//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

namespace eda::hls::mapper {

/**
 * \brief Describes configuration of hardware.
 * \author <a href="mailto:chupilko@ispras.ru>Mikhail Chupilko</a>
 */
class HWConfig {
public:
  // TODO: there will be more classes for different FPGA families.
  HWConfig(const std::string &name, const std::string &family, const std::string &vendor);

  std::string getName();
  std::string getFamily();
  std::string getVendor();

private:
  const std::string name;
  const std::string family;
  const std::string vendor;
};

/*
  params:
   -- Custom ASIC: technology, area
   -- Programmable ASIC
     - Parts: FPGA/BMC (LEs: the number, structure (LUTs, triggers, delays), placement;
                        BRAMs: size, ports, delay) | DSP (the number, impl ops, delay) | PU (Arch, Freq)
     - Communications: Interfaces | Buses
*/

} // namespace eda::hls::mapper
