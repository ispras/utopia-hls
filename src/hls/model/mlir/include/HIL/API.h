#pragma once

#include "Combine.h"
#include "Model.h"
#include "hls/model/model.h"

namespace mlir::transforms {
using mlir::model::MLIRModule;
class Transformer {
public:
  Transformer(MLIRModule &&module);
  Transformer(Transformer &&oth);
  void apply_transform(std::function<void(MLIRModule &)> transform);
  void undo_transforms();
  MLIRModule done();

private:
  MLIRModule module_;
  MLIRModule module_init_;
};

std::function<void(MLIRModule &)> ChanAddSourceTarget();
std::function<void(MLIRModule &)> InsertDelay(std::string chan_name,
                                              unsigned latency);
} // namespace mlir::transforms
