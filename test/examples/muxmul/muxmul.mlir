module {
  dfcir.kernel "MuxMul" {
    %0 = dfcir.constant<!dfcir.fixed<false, 32, 0>> 1 : ui32
    %1 = dfcir.constant<!dfcir.fixed<false, 32, 0>> 0 : ui32
    %2 = dfcir.input<!dfcir.fixed<false, 1, 0>> ("ctrl")
    %3 = dfcir.mux(%2: !dfcir.stream<!dfcir.fixed<false, 1, 0>>, %0, %1) : !dfcir.const<!dfcir.fixed<false, 32, 0>>
    %4 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x")
    %5 = dfcir.add[?] (%4 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %4 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %6 = dfcir.mul[?] (%5 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %3 : !dfcir.const<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %7 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%7 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %6 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
  }
}
