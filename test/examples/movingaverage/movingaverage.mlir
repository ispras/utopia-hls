dfcir.kernel "MovingAverage" {
  %0 = dfcir.constant<!dfcir.fixed<false, 32, 0>> 0 : i64
  %1 = dfcir.constant<!dfcir.float<8, 24>> 0.000000e+00 : f32
  %2 = dfcir.constant<!dfcir.fixed<false, 32, 0>> 1 : i64
  %3 = dfcir.constant<!dfcir.fixed<false, 32, 0>> 2 : i64
  %4 = dfcir.constant<!dfcir.fixed<false, 32, 0>> 3 : i64
  %5 = dfcir.input<!dfcir.float<8, 24>> ("x")
  %6 = dfcir.scalarInput<!dfcir.fixed<false, 32, 0>> ("size")
  %7 = dfcir.offset(%5, -1 : i64) : !dfcir.stream<!dfcir.float<8, 24>>
  %8 = dfcir.offset(%5, 1 : i64) : !dfcir.stream<!dfcir.float<8, 24>>
  %9 = dfcir.simpleCounter<!dfcir.fixed<false, 32, 0>> (%6: !dfcir.scalar<!dfcir.fixed<false, 32, 0>>)
  %10 = dfcir.greater[?] (%9 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %0 : !dfcir.const<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
  %11 = dfcir.sub[?] (%6 : !dfcir.scalar<!dfcir.fixed<false, 32, 0>>, %2 : !dfcir.const<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>> {latency = -1 : i32}
  %12 = dfcir.less[?] (%9 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>, %11 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
  %13 = dfcir.and[?] (%10 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>, %12 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
  %14 = dfcir.mux(%10: !dfcir.stream<!dfcir.fixed<false, 1, 0>>, %1: !dfcir.const<!dfcir.float<8, 24>>, %7) : !dfcir.stream<!dfcir.float<8, 24>>
  %15 = dfcir.mux(%12: !dfcir.stream<!dfcir.fixed<false, 1, 0>>, %1: !dfcir.const<!dfcir.float<8, 24>>, %8) : !dfcir.stream<!dfcir.float<8, 24>>
  %16 = dfcir.mux(%13: !dfcir.stream<!dfcir.fixed<false, 1, 0>>, %3: !dfcir.const<!dfcir.fixed<false, 32, 0>>, %4: !dfcir.const<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.fixed<false, 32, 0>>
  %17 = dfcir.add[?] (%14 : !dfcir.stream<!dfcir.float<8, 24>>, %5 : !dfcir.stream<!dfcir.float<8, 24>>) : !dfcir.stream<!dfcir.float<8, 24>> {latency = -1 : i32}
  %18 = dfcir.add[?] (%17 : !dfcir.stream<!dfcir.float<8, 24>>, %15 : !dfcir.stream<!dfcir.float<8, 24>>) : !dfcir.stream<!dfcir.float<8, 24>> {latency = -1 : i32}
  %19 = dfcir.div[?] (%18 : !dfcir.stream<!dfcir.float<8, 24>>, %16 : !dfcir.stream<!dfcir.fixed<false, 32, 0>>) : !dfcir.stream<!dfcir.float<8, 24>> {latency = -1 : i32}
  %20 = dfcir.output<!dfcir.float<8, 24>> ("y") <= %19 : !dfcir.stream<!dfcir.float<8, 24>> {operandSegmentSizes = array<i32: 0, 1>}
}
