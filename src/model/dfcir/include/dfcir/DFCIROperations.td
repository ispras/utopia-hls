//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_Operations
#define DFCIR_Operations
  
include "DFCIRDialect.td"
include "DFCIROpInterfaces.td"
include "DFCIRTypes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

class DFCIR_Op<string name, list<Trait> traits = []> : Op<DFCIR_Dialect, name, traits>;

def KernelOp : DFCIR_Op<"kernel", 
    [NoRegionArguments, NoTerminator, SingleBlock]> {
  let summary = "Defines a dataflow kernel.";

  let arguments = (ins
    StrAttr:$name
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$name $body attr-dict";
}

def ScalarInputOp : DFCIR_Op<"scalarInput",
    [DeclareOpInterfaceMethods<NamedOpVal>, DeclareOpInterfaceMethods<InputOpInterface>]> {
  let arguments = (ins
    StrAttr:$name
  );

  let hasCustomAssemblyFormat = 1;

  let results = (outs Scalar:$res);

  let extraClassDeclaration = [{
    llvm::StringRef getValueName();
  }];
}

def ScalarOutputOp : DFCIR_Op<"scalarOutput", 
    [DeclareOpInterfaceMethods<NamedOpVal>, DeclareOpInterfaceMethods<OutputOpInterface>]> {
  let arguments = (ins
    StrAttr:$name,
    Optional<Stream>:$stream
  );

  let hasCustomAssemblyFormat = 1;

  let results = (outs Scalar:$res);

  let extraClassDeclaration = [{
    llvm::StringRef getValueName();
  }];
}

def InputOp : DFCIR_Op<"input",
    [DeclareOpInterfaceMethods<NamedOpVal>, DeclareOpInterfaceMethods<InputOpInterface>]> {
  let arguments = (ins
    StrAttr:$name,
    Optional<Stream>:$ctrl
  );

  let hasCustomAssemblyFormat = 1;

  let results = (outs Stream:$res);

  let extraClassDeclaration = [{
    llvm::StringRef getValueName();
  }];
}

def OutputOp : DFCIR_Op<"output", 
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<NamedOpVal>, DeclareOpInterfaceMethods<OutputOpInterface>]> {
  let arguments = (ins
    StrAttr:$name,
    Optional<Stream>:$ctrl,
    Optional<Stream>:$stream
  );

  let hasCustomAssemblyFormat = 1;

  let results = (outs Stream:$res);

  let extraClassDeclaration = [{
    llvm::StringRef getValueName();
  }];
}

def ConstantOp : DFCIR_Op<"constant", []> {
  let arguments = (ins AnyAttrOf<[Builtin_IntegerAttr,
                                  Builtin_FloatAttr]>:$value);

  let hasCustomAssemblyFormat = 1;

  let results = (outs Const:$res);
}

def ConnectOp : DFCIR_Op<"connect", []> {
  let arguments = (ins
    Variable:$dest,
    Variable:$src
  );

  let assemblyFormat = "`(` $dest `:` type($dest) `,` $src `:` type($src) `)` attr-dict";
}

def CastOp : DFCIR_Op<"cast", []> {
  let arguments = (ins CastVariable:$var);

  let assemblyFormat = "`<` type($res) `>` `(` $var `:` type($var) `)` attr-dict";

  let results = (outs CastVariable:$res);
}

def LatencyOp : DFCIR_Op<"latency", [DeclareOpInterfaceMethods<Scheduled>]> {
  let summary = "Sets a clock latency for the input.";

  let arguments = (ins
    Variable:$input,
    Builtin_IntegerAttr:$latency
  );

  let results = (outs Variable:$res);

  let builders = [
    OpBuilder<(ins
      "Type":$res,
      "Value":$input,
      "int32_t":$latency
    ), [{
      auto attr =
            mlir::IntegerAttr::get(IntegerType::get($_builder.getContext(), 32),
                                   latency);
      build($_builder, $_state, res, input, attr);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

class DFCIR_UnaryOp<string name, Constraint firstType,
                  Type resultType, list<Trait> traits = []> :
    DFCIR_Op<name, traits # [DeclareOpInterfaceMethods<Scheduled>, DeclareOpInterfaceMethods<UnaryOpInterface>]> {
  let arguments = (ins
    firstType:$first,
    Builtin_IntegerAttr:$latency
  );

  let results = (outs resultType:$res);

  let builders = [
    OpBuilder<(ins
      "Type":$res,
      "Value":$first,
      CArg<"int32_t", "-1">:$latency
    ), [{
      auto attr =
            mlir::IntegerAttr::get(IntegerType::get($_builder.getContext(), 32),
                                   latency);
      build($_builder, $_state, res, first, attr);
    }]>
  ];

  let extraClassDeclaration = [{
    int32_t getValueName();
  }];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      IntegerAttr latencyAttr;
      OpAsmParser::UnresolvedOperand firstRawOperands[1];
      llvm::ArrayRef<OpAsmParser::UnresolvedOperand> firstOperands(firstRawOperands);
      llvm::SMLoc firstOperandsLoc;
      Type firstRawTypes[1];
      llvm::ArrayRef<Type> firstTypes(firstRawTypes);
      Type resRawTypes[1];
      llvm::ArrayRef<Type> resTypes(resRawTypes);
      if (parser.parseLSquare()) {
        return failure();
      }

      llvm::SMLoc latencyLoc = parser.getCurrentLocation();
      if (auto noQuestion = parser.parseOptionalQuestion()) {
        auto noAttr = parser.parseCustomAttributeWithFallback(latencyAttr, Type{}) ||
	                    !latencyAttr;
        // !XOR(A, B) <=> A == B.
	      if (bool(noQuestion) == bool(noAttr)) {
	        return parser.emitError(latencyLoc) << "'" << result.name.getStringRef() <<
	               "' op requires either '?' or valid latency specified";	
	      }
      } else {
        latencyAttr =
            IntegerAttr::get(IntegerType::get(parser.getContext(), 32), -1);
      }

      result.getOrAddProperties<LessOp::Properties>().latency = latencyAttr;

      if (parser.parseRSquare() || 
          parser.parseLParen()) {
        return failure();
      }

      firstOperandsLoc = parser.getCurrentLocation();
      if (parser.parseOperand(firstRawOperands[0]) ||
          parser.parseColon() ||
          parser.parseCustomTypeWithFallback(firstRawTypes[0]) ||
	        parser.parseRParen() ||
          parser.parseColon() ||
	        parser.parseCustomTypeWithFallback(resRawTypes[0]) ||
	        parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
      }
      result.addTypes(resTypes);
      if (parser.resolveOperands(firstOperands, firstTypes, firstOperandsLoc, result.operands)) {
        return failure();
      }
      return success();
    }

    void $cppClass::print(OpAsmPrinter &printer) {
      printer << "[";
      auto latencyInt = static_cast<int32_t>(getLatency().getSInt());
      if (latencyInt == -1) {
        printer << "?";
      } else {
        printer << getLatency();
      }
      printer << "] (";
      printer << getFirst() << " : " << getFirst().getType() << ") : ";
      printer << getRes().getType();
      llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
      printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    }
  }];
}

def NotOp : DFCIR_UnaryOp<"not", Variable, Variable, [SameOperandsAndResultType]>;

def NegOp : DFCIR_UnaryOp<"neg", Variable, Variable, [SameOperandsAndResultType]>;

class DFCIR_BinaryOp<string name, Constraint firstType, Constraint secondType,
                  Type resultType, list<Trait> traits = []> :
    DFCIR_Op<name, traits # [DeclareOpInterfaceMethods<Scheduled>, DeclareOpInterfaceMethods<BinaryOpInterface>]> {
  let arguments = (ins
    firstType:$first,
    secondType:$second,
    Builtin_IntegerAttr:$latency
  );

  let results = (outs resultType:$res);

  let builders = [
    OpBuilder<(ins
      "Type":$res,
      "Value":$first,
      "Value":$second,
      CArg<"int32_t", "-1">:$latency
    ), [{
      auto attr = 
            mlir::IntegerAttr::get(IntegerType::get($_builder.getContext(), 32),
                                   latency);
      build($_builder, $_state, res, first, second, attr);
    }]>
  ];

  let extraClassDeclaration = [{
    int32_t getValueName();
  }];

  let hasCustomAssemblyFormat = 1;

  let extraClassDefinition = [{
    ParseResult $cppClass::parse(OpAsmParser &parser, OperationState &result) {
      IntegerAttr latencyAttr;
      OpAsmParser::UnresolvedOperand firstRawOperands[1];
      llvm::ArrayRef<OpAsmParser::UnresolvedOperand> firstOperands(firstRawOperands);
      llvm::SMLoc firstOperandsLoc;
      Type firstRawTypes[1];
      llvm::ArrayRef<Type> firstTypes(firstRawTypes);
      OpAsmParser::UnresolvedOperand secondRawOperands[1];
      llvm::ArrayRef<OpAsmParser::UnresolvedOperand> secondOperands(secondRawOperands);
      llvm::SMLoc secondOperandsLoc;
      Type secondRawTypes[1];
      llvm::ArrayRef<Type> secondTypes(secondRawTypes);
      Type resRawTypes[1];
      llvm::ArrayRef<Type> resTypes(resRawTypes);
      if (parser.parseLSquare()) {
        return failure();
      }
      
      llvm::SMLoc latencyLoc = parser.getCurrentLocation();
      if (auto noQuestion = parser.parseOptionalQuestion()) {
        auto noAttr = parser.parseCustomAttributeWithFallback(latencyAttr, Type{}) ||
	                    !latencyAttr;
        // !XOR(A, B) <=> A == B.
	      if (bool(noQuestion) == bool(noAttr)) {
	        return parser.emitError(latencyLoc) << "'" << result.name.getStringRef() <<
	               "' op requires either '?' or valid latency specified";	
	      }
      } else {
        latencyAttr =
            IntegerAttr::get(IntegerType::get(parser.getContext(), 32), -1);
      }

      result.getOrAddProperties<LessOp::Properties>().latency = latencyAttr;

      if (parser.parseRSquare() || 
          parser.parseLParen()) {
        return failure();
      }

      firstOperandsLoc = parser.getCurrentLocation();
      if (parser.parseOperand(firstRawOperands[0]) ||
          parser.parseColon() ||
          parser.parseCustomTypeWithFallback(firstRawTypes[0]) ||
          parser.parseComma()) {
        return failure();
      }
  
      secondOperandsLoc = parser.getCurrentLocation();
      if (parser.parseOperand(secondRawOperands[0]) ||
          parser.parseColon() || 
	        parser.parseCustomTypeWithFallback(secondRawTypes[0]) ||
	        parser.parseRParen() ||
          parser.parseColon() ||
	        parser.parseCustomTypeWithFallback(resRawTypes[0]) ||
	        parser.parseOptionalAttrDict(result.attributes)) {
        return failure();
      }
      result.addTypes(resTypes);
      if (parser.resolveOperands(firstOperands, firstTypes, firstOperandsLoc, result.operands) ||
          parser.resolveOperands(secondOperands, secondTypes, secondOperandsLoc, result.operands)) {
        return failure();
      }
      return success();
    }

    void $cppClass::print(OpAsmPrinter &printer) {
      printer << "[";
      auto latencyInt = static_cast<int32_t>(getLatency().getSInt());
      if (latencyInt == -1) {
        printer << "?";
      } else {
        printer << getLatency();
      }
      printer << "] (";
      printer << getFirst() << " : " << getFirst().getType() << ", ";
      printer << getSecond() << " : " << getSecond().getType() << ") : ";
      printer << getRes().getType();
      llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
      printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    }
  }];
}

def OffsetOp : DFCIR_Op<"offset", []> {
  let arguments = (ins
    Stream:$stream,
    Builtin_IntegerAttr:$offset
  );

  let hasCustomAssemblyFormat = 1;

  let results = (outs Stream:$res);
}

def ConvertOp : DFCIR_Op<"convert", []> {
  let arguments = (ins ConvertVariable:$var);

  let assemblyFormat = "`(` $var `:` type($var) `)` attr-dict `:` type($res)";

  let results = (outs Stream:$res); 
}

def MuxOp : DFCIR_Op<"mux", []> {
  let arguments = (ins
    Stream:$control,
    Variadic<Variable>:$vars
  );

  let hasCustomAssemblyFormat = 1;

  let results = (outs Variable:$res);
}

def LessOp : DFCIR_BinaryOp<"less", Variable, Variable, Variable, []>;

def LessEqOp : DFCIR_BinaryOp<"lessEq", Variable, Variable, Variable, []>;

def GreaterOp : DFCIR_BinaryOp<"greater", Variable, Variable, Variable, []>;

def GreaterEqOp : DFCIR_BinaryOp<"greaterEq", Variable, Variable, Variable, []>;

def EqOp : DFCIR_BinaryOp<"eq", Variable, Variable, Variable, []>;

def NotEqOp : DFCIR_BinaryOp<"notEq", Variable, Variable, Variable, []>;

def AndOp : DFCIR_BinaryOp<"and", Variable, Variable, Variable, []>;

def OrOp : DFCIR_BinaryOp<"or", Variable, Variable, Variable, []>;

def XorOp : DFCIR_BinaryOp<"xor", Variable, Variable, Variable, []>;

def AddOp : DFCIR_BinaryOp<"add", Variable, Variable, Variable, []>;

def SubOp : DFCIR_BinaryOp<"sub", Variable, Variable, Variable, []>;

def MulOp : DFCIR_BinaryOp<"mul", Variable, Variable, Variable, []>;

def DivOp : DFCIR_BinaryOp<"div", Variable, Variable, Variable, []>;

def ShiftLeftOp : DFCIR_Op<"shl", [DeclareOpInterfaceMethods<ShiftOpInterface>]> {
  let arguments = (ins 
    Variable:$first,
    Builtin_IntegerAttr:$bits
  );

  let assemblyFormat = "`(` $first `:` type($first) `,` $bits `)` `:` type($res) attr-dict";

  let results = (outs Variable:$res);
}

def ShiftRightOp : DFCIR_Op<"shr", [DeclareOpInterfaceMethods<ShiftOpInterface>]> {
  let arguments = (ins 
    Variable:$first,
    Builtin_IntegerAttr:$bits
  );

  let assemblyFormat = "`(` $first `:` type($first) `,` $bits `)` `:` type($res) attr-dict";

  let results = (outs Variable:$res);
}

// Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
// Type system rework required.

#endif // DFCIR_Operations
