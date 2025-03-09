//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_IO_H
#define DFCXX_IO_H

#include "dfcxx/kernel_meta.h"

namespace dfcxx {

class Kernel;

class IO {
  friend Kernel;

private:
  KernelMeta &meta;

  IO(KernelMeta &meta);

public:
  DFVariable input(const std::string &name, const DFType &type);

  DFVariable inputScalar(const std::string &name, const DFType &type);

  DFVariable newStream(const DFType &type);

  DFVariable newScalar(const DFType &type);

  DFVariable output(const std::string &name, const DFType &type);

  DFVariable outputScalar(const std::string &name, const DFType &type);
};

} // namespace dfcxx

#endif // DFCXX_IO_H
