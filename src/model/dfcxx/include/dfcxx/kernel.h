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
#include <vector>

// This forward declaration is needed to avoid
// users having to include LLVM headers.
namespace llvm {
  class raw_fd_ostream;
}

namespace dfcxx {

class DFCIRBuilder;

class Kernel {
  friend DFCIRBuilder;

private:
  KernStorage storage;
  TypeBuilder typeBuilder;
  VarBuilder varBuilder;
  Graph graph;
  
  bool compileDot(llvm::raw_fd_ostream *stream);

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

};

} // namespace dfcxx

#endif // DFCXX_KERNEL_H
