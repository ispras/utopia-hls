// Объявление псевдонимов используемых типов.
!ui32t = !dfcir.fixed<false, 32, 0>
!ui32s = !dfcir.stream<!ui32t>
!ui32v = !dfcir.scalar<!ui32t>
!ui32c = !dfcir.const<!ui32t>
!boolt = !dfcir.fixed<false, 1, 0>
!bools = !dfcir.stream<!boolt>
!f32t  = !dfcir.float<8, 24>
!f32s  = !dfcir.stream<!f32t>
!f32c  = !dfcir.const<!f32t>

// Объявление ядра, вычисляющего скользящее среднее.
dfcir.kernel "MovingAverage" {
  // Объявление используемых констант и входных данных.
  %C0   = dfcir.constant<!ui32t> 0: i64
  %C0F  = dfcir.constant<!f32t> 0.0: f32
  %C1   = dfcir.constant<!ui32t> 1: i64
  %C2   = dfcir.constant<!ui32t> 2: i64
  %C3   = dfcir.constant<!ui32t> 3: i64
  %x    = dfcir.input<!f32t>("x")
  %size = dfcir.scalarInput<!ui32t>("size")

  %prevOrig = dfcir.offset(%x, -1: i64) : !f32s
  %nextOrig = dfcir.offset(%x,  1: i64) : !f32s

  // Объявление счетчика и вычисление предикатов.
  %cnt = dfcir.simpleCounter<!ui32t>(%size: !ui32v)
  %abvLowBnd = dfcir.greater(%cnt: !ui32s, %C0: !ui32c) : !bools
  %lessThanSize = dfcir.sub(%size: !ui32v, %C1: !ui32c) : !ui32s
  %blwUppBnd = dfcir.less(%cnt: !ui32s, %lessThanSize: !ui32s) : !bools
  %inBounds = dfcir.and(%abvLowBnd: !bools, %blwUppBnd: !bools) : !bools
  %prev = dfcir.mux(%abvLowBnd: !bools, %C0F: !f32c, %prevOrig) : !f32s
  %next = dfcir.mux(%blwUppBnd: !bools, %C0F: !f32c, %nextOrig) : !f32s

  // Вычисление суммы для подсчета скользящего среднего.
  %divisor = dfcir.mux(%inBounds: !bools, %C2: !ui32c, %C3: !ui32c) : !ui32s
  %sum1 = dfcir.add(%prev: !f32s, %x: !f32s) : !f32s
  %sum2 = dfcir.add(%sum1: !f32s, %next: !f32s) : !f32s
  %result = dfcir.div(%sum2: !f32s, %divisor: !ui32s) : !f32s

  // Объявление выходного потока.
  %y = dfcir.output<!f32t>("y") <= %result: !f32s
}
