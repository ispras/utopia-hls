module {
  dfcir.kernel "MovingSum" {
    %0 = dfcir.input<!dfcir.fixed<false, 32, 0>> ("x")
    %1 = dfcir.offset(%0, 1 : i64) : !dfcir.stream<!dfcir.fixed<false, 32, 0>>
    %2 = dfcir.offset(%0, -1 : i64) : !dfcir.stream<!dfcir.fixed<false, 32, 0>>
    %3 = dfcir.add[?] (%2 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %4 = dfcir.add[?] (%3 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %1 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
    %5 = dfcir.output<!dfcir.fixed<false, 32, 0>> ("out") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%5 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %4 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>)
  }
}
