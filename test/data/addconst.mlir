module {
  dfcir.kernel "AddConst" {
    %0 = dfcir.constant<!dfcir.fixed<false, 32, 0>> 5 : ui32
    %1 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x")
    %2 = dfcir.add(%1 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.const<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>>
    %3 = dfcir.add(%2 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %1 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>>
    %4 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%4 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %3 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
  }
}
