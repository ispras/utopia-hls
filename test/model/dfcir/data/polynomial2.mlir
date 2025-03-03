module {
  dfcir.kernel "Polynomial2" {
    %0 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x")
    %1 = dfcir.mul[?] (%0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %2 = dfcir.add[?] (%1 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %3 = dfcir.add[?] (%2 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %4 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%4 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %3 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
  }
}
