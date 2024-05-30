#ifndef DFCIR_Types
#define DFCIR_Types

include "DFCIRDialect.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def DFType : TypeInterface<"DFType"> {
  let cppNamespace = "mlir::dfcir";

  let methods = [
    InterfaceMethod<"Get the DF-data type",
    "mlir::Type", "getDFType", (ins)>
  ];

}

def SVSynthesisable : TypeInterface<"SVSynthesisable"> {
  let cppNamespace = "mlir::dfcir";

  let methods = [
    InterfaceMethod<"Print type's SystemVerilog signature",
    "void", "printSVSignature", (ins "llvm::raw_string_ostream &":$out)>,
    InterfaceMethod<"Get type's  SystemVerilog bit width ",
    "uint64_t", "getBitWidth", (ins)>
  ];

}

class DFCIR_DialectType<string typeName, string typeMnemonic, list<Trait> traits = []> 
	: TypeDef<DFCIR_Dialect, typeName, traits> {
	let mnemonic = typeMnemonic;
}

def Fixed : DFCIR_DialectType<"DFCIRFixed", "fixed", [DeclareTypeInterfaceMethods<SVSynthesisable>]> {
	let parameters = (ins
		"bool":$sign,
		"unsigned":$integerBits,
		"unsigned":$fractionBits);
  let extraClassDeclaration = [{
    void printSVSignature(llvm::raw_string_ostream &out);
    uint64_t getBitWidth();
  }];
	let assemblyFormat = "`<` $sign `,` $integerBits `,` $fractionBits `>`";
}

def Float : DFCIR_DialectType<"DFCIRFloat", "float", [DeclareTypeInterfaceMethods<SVSynthesisable>]> {
	let parameters = (ins
		"unsigned":$exponentBits,
		"unsigned":$fractionBits);
  let extraClassDeclaration = [{
    void printSVSignature(llvm::raw_string_ostream &out);
    uint64_t getBitWidth();
  }];
	let assemblyFormat = "`<` $exponentBits `,` $fractionBits `>`";
}

def RawBits : DFCIR_DialectType<"DFCIRRawBits", "rawbits", [DeclareTypeInterfaceMethods<SVSynthesisable>]> {
	let parameters = (ins "unsigned":$bits);
	let assemblyFormat = "`<` $bits `>`";
  let extraClassDeclaration = [{
    void printSVSignature(llvm::raw_string_ostream &out);
    uint64_t getBitWidth();
  }];
}

def Primitive : TypeConstraint<Or<[Fixed.predicate, Float.predicate, RawBits.predicate]>, "primitive">;

def Complex : DFCIR_DialectType<"DFCIRComplex", "complex"> {
	let parameters = (ins "Type":$complexType);
	let assemblyFormat = "`<` qualified($complexType) `>`";
}

def Streamable : TypeConstraint<Or<[Primitive.predicate, Complex.predicate]>, "streamable">;

def Stream : DFCIR_DialectType<"DFCIRStream", "stream", [DeclareTypeInterfaceMethods<DFType>]> {
	let parameters = (ins "Type":$streamType);
  let assemblyFormat = "`<` qualified($streamType) `>`";
  let extraClassDeclaration = [{
    mlir::Type getDFType();
  }];
}

def Scalar : DFCIR_DialectType<"DFCIRScalar", "scalar", [DeclareTypeInterfaceMethods<DFType>]> {
	let parameters = (ins "Type":$scalarType);
  let assemblyFormat = "`<` qualified($scalarType) `>`";
  let extraClassDeclaration = [{
    mlir::Type getDFType();
  }];
}

def Const : DFCIR_DialectType<"DFCIRConstant", "const", [DeclareTypeInterfaceMethods<DFType>]> {
  let parameters = (ins "Type":$constType);
  let assemblyFormat = "`<` qualified($constType) `>`";
  let extraClassDeclaration = [{
    mlir::Type getDFType();
  }];
}

def ConvertVariable : TypeConstraint<Or<[Scalar.predicate, Const.predicate]>, "convertVariable">;

def CastVariable : TypeConstraint<Or<[Stream.predicate, Scalar.predicate]>, "castVariable">;

def Variable : AnyTypeOf<[Stream, Scalar, Const]>;

def CounterChain : DFCIR_DialectType<"DFCIRCounterChain", "cchain">;
	
def Entity : TypeConstraint<Or<[Variable.predicate, CounterChain.predicate]>, "entity">;

	
#endif // DFCIR_Types
