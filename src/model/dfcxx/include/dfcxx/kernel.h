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
#include "dfcxx/types/type.h"
#include "dfcxx/vars/var.h"

#include <initializer_list>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// This forward declaration is needed to avoid
// users having to include LLVM headers.
namespace llvm {
  class raw_fd_ostream;
}

namespace dfcxx {

class Kernel {
private:
  KernMeta meta;
  
  bool compileDot(llvm::raw_fd_ostream *stream);

  void rebindInput(DFVariable source, Node input, Kernel &kern);

  DFVariable rebindOutput(Node output, DFVariable target, Kernel &kern);

  void deleteNode(Node node);

protected:
  IO io;
  Offset offset;
  Constant constant;
  Control control;

  DFType dfUInt(uint8_t bits);

  DFType dfInt(uint8_t bits);

  DFType dfFloat(uint8_t expBits, uint8_t fracBits);

  DFType dfBool();

  using IOBinding = std::pair<DFVariable&, std::string>;

  template <typename Kern, typename... Args>
  void instance(std::initializer_list<IOBinding> bindings, Args && ...args) {
    Kern kern(std::forward<Args>(args)...);

    for (auto &binding: bindings) {
      Node node = kern.meta.graph.getNodeByName(binding.second);
      kern.meta.graph.resetNodeName(binding.second);
      if (node.type == OpType::IN) {
        rebindInput(binding.first, node, kern);
      } else {
        binding.first = rebindOutput(node, binding.first, kern);
      } 
    }

    meta.transferFrom(std::move(kern.meta));
  }

  Kernel();

public:
  virtual ~Kernel() = default;

  virtual std::string_view getName() = 0;
  
  const Graph &getGraph() const;

  bool compile(const DFLatencyConfig &config,
               const std::vector<std::string> &outputPaths,
               const Scheduler &sched);
  
  bool compile(const DFLatencyConfig &config,
               const DFOutputPaths &outputPaths,
               const Scheduler &sched);

  bool simulate(const std::string &inDataPath,
                const std::string &outFilePath);

};

} // namespace dfcxx

#endif // DFCXX_KERNEL_H
