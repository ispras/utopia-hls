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
#include "dfcxx/graph.h"
#include "dfcxx/io.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/offset.h"
#include "dfcxx/typebuilders/builder.h"
#include "dfcxx/typedefs.h"
#include "dfcxx/types/types.h"
#include "dfcxx/varbuilders/builder.h"
#include "dfcxx/vars/var.h"

#include <string_view>

namespace dfcxx {

class DFCIRBuilder;

class Kernel {
  friend DFCIRBuilder;

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

private:
  KernStorage storage;
  TypeBuilder typeBuilder;
  VarBuilder varBuilder;
  Graph graph;

public:
  virtual ~Kernel() = default;

  virtual std::string_view getName() = 0;

  bool compile(const DFLatencyConfig &config, const Scheduler &sched);

  bool compile(const DFLatencyConfig &config, const std::string &filePath,
               const Scheduler &sched);
};

} // namespace dfcxx

#endif // DFCXX_KERNEL_H
