#ifndef DFCIR_Operations
#define DFCIR_Operations
	
include "DFCIRDialect.td"
include "DFCIRTypes.td"
include "DFCIROpInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class DFCIR_Op<string name, list<Trait> traits = []> : Op<DFCIR_Dialect, name, traits>;


def KernelOp : DFCIR_Op<"kernel", [NoRegionArguments, NoTerminator, SingleBlock]> {
	let summary = "Defines a dataflow kernel.";

	let arguments = (ins
		StrAttr:$name);

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = "$name $body attr-dict";
}

def ScalarInputOp : DFCIR_Op<"scalarInput", [DeclareOpInterfaceMethods<NamedOpVal>]> {
	let arguments = (ins
		StrAttr:$name);
		
	//let assemblyFormat = "`<` type($res) `>` `(` $name `)` attr-dict";
    let hasCustomAssemblyFormat = 1;

	let results = (outs Scalar:$res);

    let extraClassDeclaration = [{
        llvm::StringRef getValueName();
    }];
}

def ScalarOutputOp : DFCIR_Op<"scalarOutput", [DeclareOpInterfaceMethods<NamedOpVal>]> {
	let arguments = (ins
		StrAttr:$name,
		Optional<Stream>:$stream);
	
	//let assemblyFormat = "`<` type($res) `>` `(` $name `)` (`<` `=` $stream^ `:` type($stream))? attr-dict";
    let hasCustomAssemblyFormat = 1;

	let results = (outs Scalar:$res);

    let extraClassDeclaration = [{
        llvm::StringRef getValueName();
    }];
}

def InputOp : DFCIR_Op<"input", [DeclareOpInterfaceMethods<NamedOpVal>]> {
	let arguments = (ins
		StrAttr:$name,
		Optional<Stream>:$ctrl);
		
	//let assemblyFormat = "`<` type($res) `>` `(` $name (`,` $ctrl^ `:` type($ctrl))? `)` attr-dict";
    let hasCustomAssemblyFormat = 1;

    let results = (outs Stream:$res);

    let extraClassDeclaration = [{
        llvm::StringRef getValueName();
    }];
}

def OutputOp : DFCIR_Op<"output", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<NamedOpVal>]> {
	let arguments = (ins
		StrAttr:$name,
		Optional<Stream>:$ctrl,
		Optional<Stream>:$stream);
	
	//let assemblyFormat = "`<` type($res) `>` `(` $name (`,` $ctrl^ `:` type($ctrl))? `)` (`<` `=` $stream^ `:` type($stream))? attr-dict";
    let hasCustomAssemblyFormat = 1;

    let results = (outs Stream:$res);

    let extraClassDeclaration = [{
        llvm::StringRef getValueName();
    }];
}

def ConnectOp : DFCIR_Op<"connect", []> {
	let arguments = (ins
		Variable:$connecting,
		Variable:$connectee);
		
	let assemblyFormat = "`(` $connecting `:` type($connecting) `,` $connectee `:` type($connectee) `)` attr-dict";

}

def OffsetOp : DFCIR_Op<"offset", []> {
	let arguments = (ins
		Stream:$stream,
        Builtin_IntegerAttr:$offset);
		
	let assemblyFormat = "`(` $stream `:` type($stream) `,` $offset `)` attr-dict `:` type($res)";

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
		Variadic<Variable>:$vars);
		
	//let assemblyFormat = "`(` $control `:` type($control) `,` $vars `:` type($vars) `)` attr-dict `:` type($res)";
    let hasCustomAssemblyFormat = 1;

	let results = (outs Variable:$res);
}

def LessOp : DFCIR_Op<"less", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def LessEqOp : DFCIR_Op<"lessEq", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def MoreOp : DFCIR_Op<"more", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def MoreEqOp : DFCIR_Op<"moreEq", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def EqOp : DFCIR_Op<"eq", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def NotEqOp : DFCIR_Op<"notEq", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def AndOp : DFCIR_Op<"and", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def OrOp : DFCIR_Op<"or", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def XorOp : DFCIR_Op<"xor", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def NotOp : DFCIR_Op<"not", [SameOperandsAndResultType, ]> {
	let arguments = (ins Variable:$first);
		
	let assemblyFormat = "$first attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def NegOp : DFCIR_Op<"neg", [SameOperandsAndResultType, ]> {
	let arguments = (ins Variable:$first);
	
	let assemblyFormat = "$first attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def AddOp : DFCIR_Op<"add", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def SubOp : DFCIR_Op<"sub", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def MulOp : DFCIR_Op<"mul", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

def DivOp : DFCIR_Op<"div", []> {
	let arguments = (ins
		Variable:$first,
		Variable:$second);
		
	let assemblyFormat = "`(` $first `:` type($first) `,` $second `:` type($second)`)` attr-dict `:` type($res)";
	
	let results = (outs Variable:$res);
}

//def CreateCCOp : DFCIR_Op<"createCc", []> {
//	let results = (outs CounterChain);
//}

//def AddCounterOp : DFCIR_Op<"addCounter", []> {
//	let arguments = (ins
//		CounterChain:$chain,
//		Const:$step,
//		Variable:$max);

//	let assemblyFormat = "`(` $chain `,` $step `,` $max `:` type($max) `)` attr-dict `:` type($res)";

//	let results = (outs Stream:$res);
//}

//def SimpleCounterOp : DFCIR_Op<"simpleCounter", []> {
//	let arguments = (ins
//		Const:$width,
//		Variable:$max);
	
//	let assemblyFormat = "`<` type($res) `>` `(` $width `,` $max `:` type($max) `)` attr-dict";
	
//	let results = (outs Stream:$res);
//}

def ConstantOp : DFCIR_Op<"constant", []> {
	let arguments = (ins AnyAttrOf<[Builtin_IntegerAttr, Builtin_FloatAttr]>:$value);
	
	//let assemblyFormat = "`<` type($res) `>` $value attr-dict";
    let hasCustomAssemblyFormat = 1;

    let results = (outs Const:$res);
}

def CastOp : DFCIR_Op<"cast", []> {
	let arguments = (ins CastVariable:$var);
	
	let assemblyFormat = "`<` type($res) `>` `(` $var `:` type($var) `)` attr-dict";
	
	let results = (outs CastVariable:$res);
}

#endif // DFCIR_Operations
