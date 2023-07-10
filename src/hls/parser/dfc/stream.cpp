//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/parser/dfc/stream.h"
#include "hls/parser/dfc/internal/builder.h"

using Builder = eda::hls::parser::dfc::Builder;

namespace dfc {

void declare(const wire *var) {
  Builder::get().declareWire(var);
}

void connect(const wire *in, const wire *out) {
  Builder::get().connectWires(in, out);
}

void connect(const std::string &opcode,
             const std::vector<const wire*> &in,
             const std::vector<const wire*> &out) {
  Builder::get().connectWires(opcode, in, out);
}

void connectToInstanceInput(const std::string &instanceName,
                            const wire *in,
                            const std::string &inputName) {
  Builder::get().connectToInstanceInput(instanceName, in, inputName);
}

void connectToInstanceOutput(const std::string &instanceName,
                             const wire *out,
                             const std::string &outputName) {
  Builder::get().connectToInstanceOutput(instanceName, out, outputName);
}

void instance(const std::string &instanceName,
              const std::string &kernelName) {
  Builder::get().getKernel()->createInstanceUnit("INSTANCE",
                                                 instanceName,
                                                 kernelName);
}

} // namespace dfc