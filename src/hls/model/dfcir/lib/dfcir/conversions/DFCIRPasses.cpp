#include "dfcir/conversions/DFCIRPasses.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#define GEN_PASS_CLASSES
#include "dfcir/conversions/DFCIRPasses.h.inc"

namespace mlir::dfcir {

class DFCIRToFIRRTLPass : public DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass> {
public:
    void runOnOperation() override {

        }
    };

std::unique_ptr<mlir::Pass> createDFCIRToFIRRTLPass() {
    return std::make_unique<DFCIRToFIRRTLPass>();
}

} // namespace mlir::dfcir