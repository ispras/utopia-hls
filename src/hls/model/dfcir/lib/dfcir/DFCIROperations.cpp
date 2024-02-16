#include "dfcir/DFCIROperations.h"
#include "dfcir/DFCIRDialect.h"

::mlir::ParseResult mlir::dfcir::InputOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    ::mlir::Type resRawTypes[1];
    ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
    ::mlir::StringAttr nameAttr;
    ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctrlOperands;
    ::llvm::SMLoc ctrlOperandsLoc;
    (void)ctrlOperandsLoc;
    ::llvm::SmallVector<::mlir::Type, 1> ctrlTypes;

    if (parser.parseLess())
        return ::mlir::failure();
    {
        mlir::Type streamType;
        if (parser.parseCustomTypeWithFallback(streamType))
            return ::mlir::failure();
        //parseStreamable(parser, streamType);
        ::mlir::dfcir::DFCIRStreamType type = mlir::dfcir::DFCIRStreamType::get(result.getContext(), streamType);
        resRawTypes[0] = type;
    }
    if (parser.parseGreater())
        return ::mlir::failure();
    if (parser.parseLParen())
        return ::mlir::failure();

    if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
        return ::mlir::failure();
    }
    if (nameAttr) result.attributes.append("name", nameAttr);
    if (::mlir::succeeded(parser.parseOptionalComma())) {

        {
            ctrlOperandsLoc = parser.getCurrentLocation();
            ::mlir::OpAsmParser::UnresolvedOperand operand;
            ::mlir::OptionalParseResult parseResult =
                    parser.parseOptionalOperand(operand);
            if (parseResult.has_value()) {
                if (failed(*parseResult))
                    return ::mlir::failure();
                ctrlOperands.push_back(operand);
            }
        }
        if (parser.parseColon())
            return ::mlir::failure();

        {
            ::mlir::Type optionalType;
            ::mlir::OptionalParseResult parseResult =
                    parser.parseOptionalType(optionalType);
            if (parseResult.has_value()) {
                if (failed(*parseResult))
                    return ::mlir::failure();
                ctrlTypes.push_back(optionalType);
            }
        }
    }
    if (parser.parseRParen())
        return ::mlir::failure();
    {
        auto loc = parser.getCurrentLocation();(void)loc;
        if (parser.parseOptionalAttrDict(result.attributes))
            return ::mlir::failure();
    }
    result.addTypes(resTypes);
    if (parser.resolveOperands(ctrlOperands, ctrlTypes, ctrlOperandsLoc, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void mlir::dfcir::InputOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
    _odsPrinter << ' ';
    {
        auto type = getRes().getType();
        if (auto validType = ::llvm::dyn_cast<::mlir::dfcir::DFCIRStreamType>(type))
            _odsPrinter.printStrippedAttrOrType(validType);
        else
            _odsPrinter << type;
    }
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getNameAttr());
    if (getCtrl()) {
        _odsPrinter << ",";
        _odsPrinter << ' ';
        if (::mlir::Value value = getCtrl())
            _odsPrinter << value;
        _odsPrinter << ' ' << ":";
        _odsPrinter << ' ';
        _odsPrinter << (getCtrl() ? ::llvm::ArrayRef<::mlir::Type>(getCtrl().getType()) : ::llvm::ArrayRef<::mlir::Type>());
    }
    _odsPrinter << ")";
    ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back("name");
    _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

#define GET_OP_CLASSES
#include "dfcir/DFCIROperations.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerOperations() {
    addOperations<
#define GET_OP_LIST
#include "dfcir/DFCIROperations.cpp.inc"
    >();
}
