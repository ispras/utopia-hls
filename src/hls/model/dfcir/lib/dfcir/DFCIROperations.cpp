#include "dfcir/DFCIROperations.h"
#include "dfcir/DFCIRDialect.h"

::mlir::ParseResult mlir::dfcir::ScalarInputOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    ::mlir::Type resRawTypes[1];
    ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
    ::mlir::StringAttr nameAttr;
    if (parser.parseLess())
        return ::mlir::failure();

    {
        mlir::Type scalarType;
        if (parser.parseCustomTypeWithFallback(scalarType))
            return ::mlir::failure();
        ::mlir::dfcir::DFCIRScalarType type = mlir::dfcir::DFCIRScalarType::get(result.getContext(), scalarType);
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
    if (parser.parseRParen())
        return ::mlir::failure();
    {
        auto loc = parser.getCurrentLocation();(void)loc;
        if (parser.parseOptionalAttrDict(result.attributes))
            return ::mlir::failure();
    }
    result.addTypes(resTypes);
    return ::mlir::success();
}

void mlir::dfcir::ScalarInputOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
    _odsPrinter << "<";
    {
        auto type = getRes().getType().getScalarType();
        _odsPrinter << type;
        //if (auto validType = ::llvm::dyn_cast<::mlir::dfcir::DFCIRScalarType>(type))
        //    _odsPrinter.printStrippedAttrOrType(validType);
        //else
        //    _odsPrinter << type;
    }
    _odsPrinter << ">";
    _odsPrinter << ' ' << "(";
    _odsPrinter.printAttributeWithoutType(getNameAttr());
    _odsPrinter << ")";
    ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back("name");
    _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

llvm::StringRef mlir::dfcir::ScalarInputOp::getValueName() {
    return getName();
}

::mlir::ParseResult mlir::dfcir::ScalarOutputOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    ::mlir::Type resRawTypes[1];
    ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
    ::mlir::StringAttr nameAttr;
    ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> streamOperands;
    ::llvm::SMLoc streamOperandsLoc;
    (void)streamOperandsLoc;
    ::llvm::SmallVector<::mlir::Type, 1> streamTypes;
    if (parser.parseLess())
        return ::mlir::failure();

    {
        mlir::Type scalarType;
        if (parser.parseCustomTypeWithFallback(scalarType))
            return ::mlir::failure();
        ::mlir::dfcir::DFCIRScalarType type = mlir::dfcir::DFCIRScalarType::get(result.getContext(), scalarType);
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
    if (parser.parseRParen())
        return ::mlir::failure();
    if (::mlir::succeeded(parser.parseOptionalLess())) {
        if (parser.parseEqual())
            return ::mlir::failure();

        {
            streamOperandsLoc = parser.getCurrentLocation();
            ::mlir::OpAsmParser::UnresolvedOperand operand;
            ::mlir::OptionalParseResult parseResult =
                    parser.parseOptionalOperand(operand);
            if (parseResult.has_value()) {
                if (failed(*parseResult))
                    return ::mlir::failure();
                streamOperands.push_back(operand);
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
                streamTypes.push_back(optionalType);
            }
        }
    }
    {
        auto loc = parser.getCurrentLocation();(void)loc;
        if (parser.parseOptionalAttrDict(result.attributes))
            return ::mlir::failure();
    }
    result.addTypes(resTypes);
    if (parser.resolveOperands(streamOperands, streamTypes, streamOperandsLoc, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void mlir::dfcir::ScalarOutputOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
    _odsPrinter << "<";
    {
        auto type = getRes().getType().getScalarType();
        _odsPrinter << type;
        //if (auto validType = ::llvm::dyn_cast<::mlir::dfcir::DFCIRScalarType>(type))
        //    _odsPrinter.printStrippedAttrOrType(validType);
        //else
        //    _odsPrinter << type;
    }
    _odsPrinter << ">";
    _odsPrinter << ' ' << "(";
    _odsPrinter.printAttributeWithoutType(getNameAttr());
    _odsPrinter << ")";
    if (getStream()) {
        _odsPrinter << ' ' << "<";
        _odsPrinter << "=";
        _odsPrinter << ' ';
        if (::mlir::Value value = getStream())
            _odsPrinter << value;
        _odsPrinter << ' ' << ":";
        _odsPrinter << ' ';
        _odsPrinter << (getStream() ? ::llvm::ArrayRef<::mlir::Type>(getStream().getType()) : ::llvm::ArrayRef<::mlir::Type>());
    }
    ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back("name");
    _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

llvm::StringRef mlir::dfcir::ScalarOutputOp::getValueName() {
    return getName();
}

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
    _odsPrinter << "<";
    {
        auto type = getRes().getType().getStreamType();
        _odsPrinter << type;
        //if (auto validType = ::llvm::dyn_cast<::mlir::dfcir::DFCIRStreamType>(type))
        //    _odsPrinter.printStrippedAttrOrType(validType);
        //else
        //    _odsPrinter << type;
    }
    _odsPrinter << ">";
    _odsPrinter << ' ' << "(";
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

llvm::StringRef mlir::dfcir::InputOp::getValueName() {
    return getName();
}

::mlir::ParseResult mlir::dfcir::OutputOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
    ::mlir::Type resRawTypes[1];
    ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
    ::mlir::StringAttr nameAttr;
    ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctrlOperands;
    ::llvm::SMLoc ctrlOperandsLoc;
    (void)ctrlOperandsLoc;
    ::llvm::SmallVector<::mlir::Type, 1> ctrlTypes;
    ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> streamOperands;
    ::llvm::SMLoc streamOperandsLoc;
    (void)streamOperandsLoc;
    ::llvm::SmallVector<::mlir::Type, 1> streamTypes;
    if (parser.parseLess())
        return ::mlir::failure();

    {
        mlir::Type streamType;
        if (parser.parseCustomTypeWithFallback(streamType))
            return ::mlir::failure();
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
    if (::mlir::succeeded(parser.parseOptionalLess())) {
        if (parser.parseEqual())
            return ::mlir::failure();

        {
            streamOperandsLoc = parser.getCurrentLocation();
            ::mlir::OpAsmParser::UnresolvedOperand operand;
            ::mlir::OptionalParseResult parseResult =
                    parser.parseOptionalOperand(operand);
            if (parseResult.has_value()) {
                if (failed(*parseResult))
                    return ::mlir::failure();
                streamOperands.push_back(operand);
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
                streamTypes.push_back(optionalType);
            }
        }
    }
    {
        auto loc = parser.getCurrentLocation();(void)loc;
        if (parser.parseOptionalAttrDict(result.attributes))
            return ::mlir::failure();
    }
    result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ctrlOperands.size()), static_cast<int32_t>(streamOperands.size())}));
    result.addTypes(resTypes);
    if (parser.resolveOperands(ctrlOperands, ctrlTypes, ctrlOperandsLoc, result.operands))
        return ::mlir::failure();
    if (parser.resolveOperands(streamOperands, streamTypes, streamOperandsLoc, result.operands))
        return ::mlir::failure();
    return ::mlir::success();
}

void mlir::dfcir::OutputOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
    _odsPrinter << "<";
    {
        auto type = getRes().getType().getStreamType();
        _odsPrinter << type;
        //if (auto validType = ::llvm::dyn_cast<::mlir::dfcir::DFCIRStreamType>(type))
        //    _odsPrinter.printStrippedAttrOrType(validType);
        //else
        //    _odsPrinter << type;
    }
    _odsPrinter << ">";
    _odsPrinter << ' ' << "(";
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
    if (getStream()) {
        _odsPrinter << ' ' << "<";
        _odsPrinter << "=";
        _odsPrinter << ' ';
        if (::mlir::Value value = getStream())
            _odsPrinter << value;
        _odsPrinter << ' ' << ":";
        _odsPrinter << ' ';
        _odsPrinter << (getStream() ? ::llvm::ArrayRef<::mlir::Type>(getStream().getType()) : ::llvm::ArrayRef<::mlir::Type>());
    }
    ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back("operand_segment_sizes");
    elidedAttrs.push_back("name");
    _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

llvm::StringRef mlir::dfcir::OutputOp::getValueName() {
    return getName();
}

#define GET_OP_CLASSES
#include "dfcir/DFCIROperations.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerOperations() {
    addOperations<
#define GET_OP_LIST
#include "dfcir/DFCIROperations.cpp.inc"
    >();
}
