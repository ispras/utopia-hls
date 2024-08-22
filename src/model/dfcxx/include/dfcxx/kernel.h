//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_KERNEL_H
#define DFCXX_KERNEL_H

#include "dfcxx/constant.h"
#include "dfcxx/control.h"
#include "dfcxx/io.h"
#include "dfcxx/kernmeta.h"
#include "dfcxx/offset.h"
#include "dfcxx/typedefs.h"
#include "dfcxx/types/types.h"
#include "dfcxx/vars/var.h"

#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace dfcxx {

class Kernel {
private:
  KernMeta meta;

protected:
  IO io;
  Offset offset;
  Constant constant;
  Control control;

  DFType dfUInt(uint8_t bits);

  DFType dfInt(uint8_t bits);

  DFType dfFloat(uint8_t expBits, uint8_t fracBits);

  DFType dfBool();

  Kernel();

public:
  virtual ~Kernel() = default;

  virtual std::string_view getName() = 0;

  bool compile(const DFLatencyConfig &config,
               const std::vector<std::string> &outputPaths,
               const Scheduler &sched);
  
  bool compile(const DFLatencyConfig &config,
               const DFOutputPaths &outputPaths,
               const Scheduler &sched);

  bool simulate(const std::string &inDataPath,
                const std::string &outFilePath,
                bool intermediateResults = false);

};

} // namespace dfcxx

#endif // DFCXX_KERNEL_H
