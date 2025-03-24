module {
  dfcir.kernel "IDCT" {
    %0 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3784 : si32
    %1 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4017 : si32
    %2 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 799 : si32
    %3 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2276 : si32
    %4 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 181 : si32
    %5 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 8192 : si32
    %6 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1108 : si32
    %7 = dfcir.constant<!dfcir.fixed<true, 12, 0>> -256 : si13
    %8 = dfcir.constant<!dfcir.fixed<true, 12, 0>> 255 : si13
    %9 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2841 : si32
    %10 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2408 : si32
    %11 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1568 : si32
    %12 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 128 : si32
    %13 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 565 : si32
    %14 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 3406 : si32
    %15 = dfcir.input<!dfcir.fixed<true, 767, 0>> ("x")
    %16 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 767 : i32, 756 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %17 = dfcir.cast[?] (%16 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %18 = dfcir.mul[?] (%17 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %19 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 755 : i32, 744 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %20 = dfcir.cast[?] (%19 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %21 = dfcir.mul[?] (%20 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %22 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 743 : i32, 732 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %23 = dfcir.cast[?] (%22 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %24 = dfcir.mul[?] (%23 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %25 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 731 : i32, 720 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %26 = dfcir.cast[?] (%25 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %27 = dfcir.shl(%26 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %28 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 719 : i32, 708 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %29 = dfcir.cast[?] (%28 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %30 = dfcir.mul[?] (%29 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %31 = dfcir.add[?] (%23 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %29 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %32 = dfcir.mul[?] (%31 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %33 = dfcir.sub[?] (%32 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %30 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %34 = dfcir.sub[?] (%32 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %24 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %35 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 707 : i32, 696 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %36 = dfcir.cast[?] (%35 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %37 = dfcir.mul[?] (%36 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %38 = dfcir.add[?] (%36 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %20 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %39 = dfcir.mul[?] (%38 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %40 = dfcir.add[?] (%39 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %37 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %41 = dfcir.sub[?] (%39 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %21 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %42 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 695 : i32, 684 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %43 = dfcir.cast[?] (%42 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %44 = dfcir.mul[?] (%43 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %45 = dfcir.add[?] (%43 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %17 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %46 = dfcir.mul[?] (%45 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %47 = dfcir.sub[?] (%46 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %18 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %48 = dfcir.sub[?] (%47 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %33 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %49 = dfcir.add[?] (%47 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %33 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %50 = dfcir.add[?] (%46 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %44 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %51 = dfcir.sub[?] (%50 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %34 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %52 = dfcir.sub[?] (%51 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %48 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %53 = dfcir.mul[?] (%52 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %54 = dfcir.add[?] (%53 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %55 = dfcir.shr(%54 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %56 = dfcir.add[?] (%51 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %48 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %57 = dfcir.mul[?] (%56 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %58 = dfcir.add[?] (%57 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %59 = dfcir.shr(%58 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %60 = dfcir.add[?] (%50 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %34 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %61 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 683 : i32, 672 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %62 = dfcir.cast[?] (%61 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %63 = dfcir.shl(%62 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %64 = dfcir.add[?] (%63 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %65 = dfcir.sub[?] (%64 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %27 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %66 = dfcir.sub[?] (%65 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %41 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %67 = dfcir.sub[?] (%66 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %55 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %68 = dfcir.shr(%67 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %69 = dfcir.cast[?] (%68 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %70 = dfcir.cast[?] (%69 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %71 = dfcir.mul[?] (%70 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %72 = dfcir.add[?] (%66 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %55 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %73 = dfcir.shr(%72 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %74 = dfcir.cast[?] (%73 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %75 = dfcir.cast[?] (%74 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %76 = dfcir.mul[?] (%75 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %77 = dfcir.add[?] (%65 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %41 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %78 = dfcir.sub[?] (%77 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %59 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %79 = dfcir.shr(%78 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %80 = dfcir.cast[?] (%79 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %81 = dfcir.cast[?] (%80 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %82 = dfcir.mul[?] (%81 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %83 = dfcir.add[?] (%77 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %59 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %84 = dfcir.shr(%83 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %85 = dfcir.cast[?] (%84 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %86 = dfcir.cast[?] (%85 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %87 = dfcir.mul[?] (%86 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %88 = dfcir.add[?] (%64 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %27 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %89 = dfcir.sub[?] (%88 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %40 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %90 = dfcir.sub[?] (%89 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %49 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %91 = dfcir.shr(%90 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %92 = dfcir.cast[?] (%91 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %93 = dfcir.cast[?] (%92 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %94 = dfcir.mul[?] (%93 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %95 = dfcir.add[?] (%89 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %49 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %96 = dfcir.shr(%95 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %97 = dfcir.cast[?] (%96 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %98 = dfcir.cast[?] (%97 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %99 = dfcir.mul[?] (%98 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %100 = dfcir.add[?] (%88 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %40 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %101 = dfcir.sub[?] (%100 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %60 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %102 = dfcir.shr(%101 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %103 = dfcir.cast[?] (%102 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %104 = dfcir.cast[?] (%103 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %105 = dfcir.mul[?] (%104 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %106 = dfcir.add[?] (%100 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %60 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %107 = dfcir.shr(%106 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %108 = dfcir.cast[?] (%107 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %109 = dfcir.cast[?] (%108 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %110 = dfcir.mul[?] (%109 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %111 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 671 : i32, 660 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %112 = dfcir.cast[?] (%111 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %113 = dfcir.mul[?] (%112 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %114 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 659 : i32, 648 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %115 = dfcir.cast[?] (%114 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %116 = dfcir.mul[?] (%115 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %117 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 647 : i32, 636 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %118 = dfcir.cast[?] (%117 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %119 = dfcir.mul[?] (%118 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %120 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 635 : i32, 624 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %121 = dfcir.cast[?] (%120 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %122 = dfcir.shl(%121 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %123 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 623 : i32, 612 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %124 = dfcir.cast[?] (%123 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %125 = dfcir.mul[?] (%124 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %126 = dfcir.add[?] (%118 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %124 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %127 = dfcir.mul[?] (%126 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %128 = dfcir.sub[?] (%127 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %125 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %129 = dfcir.sub[?] (%127 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %119 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %130 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 611 : i32, 600 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %131 = dfcir.cast[?] (%130 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %132 = dfcir.mul[?] (%131 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %133 = dfcir.add[?] (%131 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %115 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %134 = dfcir.mul[?] (%133 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %135 = dfcir.add[?] (%134 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %132 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %136 = dfcir.sub[?] (%134 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %116 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %137 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 599 : i32, 588 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %138 = dfcir.cast[?] (%137 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %139 = dfcir.mul[?] (%138 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %140 = dfcir.add[?] (%138 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %112 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %141 = dfcir.mul[?] (%140 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %142 = dfcir.sub[?] (%141 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %113 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %143 = dfcir.sub[?] (%142 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %128 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %144 = dfcir.add[?] (%142 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %128 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %145 = dfcir.add[?] (%141 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %139 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %146 = dfcir.sub[?] (%145 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %129 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %147 = dfcir.sub[?] (%146 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %143 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %148 = dfcir.mul[?] (%147 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %149 = dfcir.add[?] (%148 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %150 = dfcir.shr(%149 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %151 = dfcir.add[?] (%146 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %143 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %152 = dfcir.mul[?] (%151 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %153 = dfcir.add[?] (%152 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %154 = dfcir.shr(%153 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %155 = dfcir.add[?] (%145 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %129 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %156 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 587 : i32, 576 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %157 = dfcir.cast[?] (%156 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %158 = dfcir.shl(%157 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %159 = dfcir.add[?] (%158 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %160 = dfcir.sub[?] (%159 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %122 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %161 = dfcir.sub[?] (%160 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %136 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %162 = dfcir.sub[?] (%161 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %150 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %163 = dfcir.shr(%162 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %164 = dfcir.cast[?] (%163 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %165 = dfcir.cast[?] (%164 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %166 = dfcir.mul[?] (%165 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %167 = dfcir.add[?] (%161 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %150 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %168 = dfcir.shr(%167 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %169 = dfcir.cast[?] (%168 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %170 = dfcir.cast[?] (%169 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %171 = dfcir.mul[?] (%170 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %172 = dfcir.add[?] (%160 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %136 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %173 = dfcir.sub[?] (%172 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %154 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %174 = dfcir.shr(%173 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %175 = dfcir.cast[?] (%174 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %176 = dfcir.cast[?] (%175 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %177 = dfcir.mul[?] (%176 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %178 = dfcir.add[?] (%172 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %154 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %179 = dfcir.shr(%178 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %180 = dfcir.cast[?] (%179 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %181 = dfcir.cast[?] (%180 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %182 = dfcir.mul[?] (%181 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %183 = dfcir.add[?] (%159 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %122 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %184 = dfcir.sub[?] (%183 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %135 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %185 = dfcir.sub[?] (%184 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %144 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %186 = dfcir.shr(%185 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %187 = dfcir.cast[?] (%186 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %188 = dfcir.cast[?] (%187 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %189 = dfcir.mul[?] (%188 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %190 = dfcir.add[?] (%184 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %144 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %191 = dfcir.shr(%190 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %192 = dfcir.cast[?] (%191 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %193 = dfcir.cast[?] (%192 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %194 = dfcir.mul[?] (%193 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %195 = dfcir.add[?] (%183 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %135 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %196 = dfcir.sub[?] (%195 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %197 = dfcir.shr(%196 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %198 = dfcir.cast[?] (%197 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %199 = dfcir.cast[?] (%198 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %200 = dfcir.mul[?] (%199 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %201 = dfcir.add[?] (%195 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %155 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %202 = dfcir.shr(%201 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %203 = dfcir.cast[?] (%202 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %204 = dfcir.cast[?] (%203 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %205 = dfcir.mul[?] (%204 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %206 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 575 : i32, 564 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %207 = dfcir.cast[?] (%206 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %208 = dfcir.mul[?] (%207 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %209 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 563 : i32, 552 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %210 = dfcir.cast[?] (%209 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %211 = dfcir.mul[?] (%210 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %212 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 551 : i32, 540 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %213 = dfcir.cast[?] (%212 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %214 = dfcir.mul[?] (%213 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %215 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 539 : i32, 528 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %216 = dfcir.cast[?] (%215 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %217 = dfcir.shl(%216 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %218 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 527 : i32, 516 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %219 = dfcir.cast[?] (%218 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %220 = dfcir.mul[?] (%219 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %221 = dfcir.add[?] (%213 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %219 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %222 = dfcir.mul[?] (%221 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %223 = dfcir.sub[?] (%222 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %220 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %224 = dfcir.sub[?] (%222 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %214 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %225 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 515 : i32, 504 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %226 = dfcir.cast[?] (%225 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %227 = dfcir.mul[?] (%226 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %228 = dfcir.add[?] (%226 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %210 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %229 = dfcir.mul[?] (%228 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %230 = dfcir.add[?] (%229 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %227 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %231 = dfcir.sub[?] (%229 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %211 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %232 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 503 : i32, 492 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %233 = dfcir.cast[?] (%232 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %234 = dfcir.mul[?] (%233 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %235 = dfcir.add[?] (%233 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %207 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %236 = dfcir.mul[?] (%235 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %237 = dfcir.sub[?] (%236 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %208 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %238 = dfcir.sub[?] (%237 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %223 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %239 = dfcir.add[?] (%237 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %223 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %240 = dfcir.add[?] (%236 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %234 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %241 = dfcir.sub[?] (%240 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %224 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %242 = dfcir.sub[?] (%241 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %238 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %243 = dfcir.mul[?] (%242 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %244 = dfcir.add[?] (%243 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %245 = dfcir.shr(%244 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %246 = dfcir.add[?] (%241 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %238 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %247 = dfcir.mul[?] (%246 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %248 = dfcir.add[?] (%247 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %249 = dfcir.shr(%248 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %250 = dfcir.add[?] (%240 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %224 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %251 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 491 : i32, 480 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %252 = dfcir.cast[?] (%251 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %253 = dfcir.shl(%252 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %254 = dfcir.add[?] (%253 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %255 = dfcir.sub[?] (%254 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %217 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %256 = dfcir.sub[?] (%255 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %231 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %257 = dfcir.sub[?] (%256 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %245 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %258 = dfcir.shr(%257 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %259 = dfcir.cast[?] (%258 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %260 = dfcir.cast[?] (%259 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %261 = dfcir.mul[?] (%260 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %262 = dfcir.add[?] (%256 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %245 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %263 = dfcir.shr(%262 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %264 = dfcir.cast[?] (%263 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %265 = dfcir.cast[?] (%264 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %266 = dfcir.mul[?] (%265 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %267 = dfcir.add[?] (%255 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %231 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %268 = dfcir.sub[?] (%267 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %249 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %269 = dfcir.shr(%268 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %270 = dfcir.cast[?] (%269 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %271 = dfcir.cast[?] (%270 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %272 = dfcir.mul[?] (%271 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %273 = dfcir.add[?] (%267 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %249 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %274 = dfcir.shr(%273 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %275 = dfcir.cast[?] (%274 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %276 = dfcir.cast[?] (%275 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %277 = dfcir.mul[?] (%276 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %278 = dfcir.add[?] (%254 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %217 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %279 = dfcir.sub[?] (%278 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %230 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %280 = dfcir.sub[?] (%279 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %239 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %281 = dfcir.shr(%280 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %282 = dfcir.cast[?] (%281 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %283 = dfcir.cast[?] (%282 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %284 = dfcir.mul[?] (%283 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %285 = dfcir.add[?] (%279 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %239 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %286 = dfcir.shr(%285 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %287 = dfcir.cast[?] (%286 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %288 = dfcir.cast[?] (%287 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %289 = dfcir.mul[?] (%288 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %290 = dfcir.add[?] (%278 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %230 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %291 = dfcir.sub[?] (%290 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %250 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %292 = dfcir.shr(%291 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %293 = dfcir.cast[?] (%292 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %294 = dfcir.cast[?] (%293 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %295 = dfcir.mul[?] (%294 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %296 = dfcir.add[?] (%290 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %250 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %297 = dfcir.shr(%296 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %298 = dfcir.cast[?] (%297 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %299 = dfcir.cast[?] (%298 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %300 = dfcir.mul[?] (%299 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %301 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 479 : i32, 468 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %302 = dfcir.cast[?] (%301 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %303 = dfcir.mul[?] (%302 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %304 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 467 : i32, 456 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %305 = dfcir.cast[?] (%304 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %306 = dfcir.mul[?] (%305 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %307 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 455 : i32, 444 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %308 = dfcir.cast[?] (%307 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %309 = dfcir.mul[?] (%308 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %310 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 443 : i32, 432 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %311 = dfcir.cast[?] (%310 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %312 = dfcir.shl(%311 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %313 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 431 : i32, 420 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %314 = dfcir.cast[?] (%313 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %315 = dfcir.mul[?] (%314 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %316 = dfcir.add[?] (%308 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %314 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %317 = dfcir.mul[?] (%316 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %318 = dfcir.sub[?] (%317 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %315 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %319 = dfcir.sub[?] (%317 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %309 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %320 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 419 : i32, 408 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %321 = dfcir.cast[?] (%320 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %322 = dfcir.mul[?] (%321 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %323 = dfcir.add[?] (%321 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %305 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %324 = dfcir.mul[?] (%323 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %325 = dfcir.add[?] (%324 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %322 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %326 = dfcir.sub[?] (%324 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %306 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %327 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 407 : i32, 396 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %328 = dfcir.cast[?] (%327 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %329 = dfcir.mul[?] (%328 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %330 = dfcir.add[?] (%328 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %302 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %331 = dfcir.mul[?] (%330 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %332 = dfcir.sub[?] (%331 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %303 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %333 = dfcir.sub[?] (%332 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %318 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %334 = dfcir.add[?] (%332 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %318 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %335 = dfcir.add[?] (%331 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %329 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %336 = dfcir.sub[?] (%335 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %319 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %337 = dfcir.sub[?] (%336 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %333 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %338 = dfcir.mul[?] (%337 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %339 = dfcir.add[?] (%338 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %340 = dfcir.shr(%339 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %341 = dfcir.add[?] (%336 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %333 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %342 = dfcir.mul[?] (%341 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %343 = dfcir.add[?] (%342 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %344 = dfcir.shr(%343 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %345 = dfcir.add[?] (%335 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %319 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %346 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 395 : i32, 384 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %347 = dfcir.cast[?] (%346 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %348 = dfcir.shl(%347 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %349 = dfcir.add[?] (%348 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %350 = dfcir.sub[?] (%349 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %312 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %351 = dfcir.sub[?] (%350 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %326 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %352 = dfcir.sub[?] (%351 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %340 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %353 = dfcir.shr(%352 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %354 = dfcir.cast[?] (%353 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %355 = dfcir.cast[?] (%354 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %356 = dfcir.shl(%355 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %357 = dfcir.add[?] (%351 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %340 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %358 = dfcir.shr(%357 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %359 = dfcir.cast[?] (%358 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %360 = dfcir.cast[?] (%359 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %361 = dfcir.shl(%360 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %362 = dfcir.add[?] (%350 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %326 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %363 = dfcir.sub[?] (%362 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %344 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %364 = dfcir.shr(%363 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %365 = dfcir.cast[?] (%364 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %366 = dfcir.cast[?] (%365 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %367 = dfcir.shl(%366 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %368 = dfcir.add[?] (%362 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %344 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %369 = dfcir.shr(%368 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %370 = dfcir.cast[?] (%369 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %371 = dfcir.cast[?] (%370 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %372 = dfcir.shl(%371 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %373 = dfcir.add[?] (%349 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %312 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %374 = dfcir.sub[?] (%373 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %325 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %375 = dfcir.sub[?] (%374 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %334 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %376 = dfcir.shr(%375 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %377 = dfcir.cast[?] (%376 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %378 = dfcir.cast[?] (%377 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %379 = dfcir.shl(%378 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %380 = dfcir.add[?] (%374 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %334 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %381 = dfcir.shr(%380 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %382 = dfcir.cast[?] (%381 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %383 = dfcir.cast[?] (%382 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %384 = dfcir.shl(%383 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %385 = dfcir.add[?] (%373 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %325 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %386 = dfcir.sub[?] (%385 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %345 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %387 = dfcir.shr(%386 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %388 = dfcir.cast[?] (%387 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %389 = dfcir.cast[?] (%388 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %390 = dfcir.shl(%389 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %391 = dfcir.add[?] (%385 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %345 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %392 = dfcir.shr(%391 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %393 = dfcir.cast[?] (%392 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %394 = dfcir.cast[?] (%393 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %395 = dfcir.shl(%394 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %396 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 383 : i32, 372 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %397 = dfcir.cast[?] (%396 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %398 = dfcir.mul[?] (%397 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %399 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 371 : i32, 360 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %400 = dfcir.cast[?] (%399 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %401 = dfcir.mul[?] (%400 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %402 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 359 : i32, 348 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %403 = dfcir.cast[?] (%402 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %404 = dfcir.mul[?] (%403 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %405 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 347 : i32, 336 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %406 = dfcir.cast[?] (%405 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %407 = dfcir.shl(%406 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %408 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 335 : i32, 324 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %409 = dfcir.cast[?] (%408 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %410 = dfcir.mul[?] (%409 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %411 = dfcir.add[?] (%403 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %409 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %412 = dfcir.mul[?] (%411 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %413 = dfcir.sub[?] (%412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %410 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %414 = dfcir.sub[?] (%412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %404 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %415 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 323 : i32, 312 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %416 = dfcir.cast[?] (%415 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %417 = dfcir.mul[?] (%416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %418 = dfcir.add[?] (%416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %400 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %419 = dfcir.mul[?] (%418 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %420 = dfcir.add[?] (%419 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %417 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %421 = dfcir.sub[?] (%419 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %401 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %422 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 311 : i32, 300 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %423 = dfcir.cast[?] (%422 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %424 = dfcir.mul[?] (%423 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %425 = dfcir.add[?] (%423 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %397 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %426 = dfcir.mul[?] (%425 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %427 = dfcir.sub[?] (%426 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %398 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %428 = dfcir.sub[?] (%427 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %413 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %429 = dfcir.add[?] (%427 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %413 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %430 = dfcir.add[?] (%426 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %424 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %431 = dfcir.sub[?] (%430 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %414 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %432 = dfcir.sub[?] (%431 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %428 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %433 = dfcir.mul[?] (%432 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %434 = dfcir.add[?] (%433 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %435 = dfcir.shr(%434 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %436 = dfcir.add[?] (%431 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %428 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %437 = dfcir.mul[?] (%436 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %438 = dfcir.add[?] (%437 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %439 = dfcir.shr(%438 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %440 = dfcir.add[?] (%430 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %414 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %441 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 299 : i32, 288 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %442 = dfcir.cast[?] (%441 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %443 = dfcir.shl(%442 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %444 = dfcir.add[?] (%443 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %445 = dfcir.sub[?] (%444 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %407 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %446 = dfcir.sub[?] (%445 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %421 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %447 = dfcir.sub[?] (%446 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %435 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %448 = dfcir.shr(%447 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %449 = dfcir.cast[?] (%448 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %450 = dfcir.cast[?] (%449 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %451 = dfcir.mul[?] (%450 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %452 = dfcir.add[?] (%260 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %450 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %453 = dfcir.mul[?] (%452 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %454 = dfcir.add[?] (%446 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %435 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %455 = dfcir.shr(%454 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %456 = dfcir.cast[?] (%455 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %457 = dfcir.cast[?] (%456 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %458 = dfcir.mul[?] (%457 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %459 = dfcir.add[?] (%265 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %457 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %460 = dfcir.mul[?] (%459 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %461 = dfcir.add[?] (%445 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %421 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %462 = dfcir.sub[?] (%461 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %439 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %463 = dfcir.shr(%462 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %464 = dfcir.cast[?] (%463 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %465 = dfcir.cast[?] (%464 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %466 = dfcir.mul[?] (%465 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %467 = dfcir.add[?] (%271 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %465 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %468 = dfcir.mul[?] (%467 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %469 = dfcir.add[?] (%461 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %439 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %470 = dfcir.shr(%469 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %471 = dfcir.cast[?] (%470 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %472 = dfcir.cast[?] (%471 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %473 = dfcir.mul[?] (%472 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %474 = dfcir.add[?] (%276 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %472 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %475 = dfcir.mul[?] (%474 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %476 = dfcir.add[?] (%444 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %407 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %477 = dfcir.sub[?] (%476 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %420 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %478 = dfcir.sub[?] (%477 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %429 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %479 = dfcir.shr(%478 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %480 = dfcir.cast[?] (%479 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %481 = dfcir.cast[?] (%480 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %482 = dfcir.mul[?] (%481 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %483 = dfcir.add[?] (%283 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %481 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %484 = dfcir.mul[?] (%483 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %485 = dfcir.add[?] (%477 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %429 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %486 = dfcir.shr(%485 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %487 = dfcir.cast[?] (%486 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %488 = dfcir.cast[?] (%487 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %489 = dfcir.mul[?] (%488 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %490 = dfcir.add[?] (%288 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %488 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %491 = dfcir.mul[?] (%490 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %492 = dfcir.add[?] (%476 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %420 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %493 = dfcir.sub[?] (%492 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %440 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %494 = dfcir.shr(%493 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %495 = dfcir.cast[?] (%494 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %496 = dfcir.cast[?] (%495 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %497 = dfcir.mul[?] (%496 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %498 = dfcir.add[?] (%294 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %496 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %499 = dfcir.mul[?] (%498 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %500 = dfcir.add[?] (%492 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %440 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %501 = dfcir.shr(%500 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %502 = dfcir.cast[?] (%501 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %503 = dfcir.cast[?] (%502 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %504 = dfcir.mul[?] (%503 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %505 = dfcir.add[?] (%299 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %503 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %506 = dfcir.mul[?] (%505 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %507 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 287 : i32, 276 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %508 = dfcir.cast[?] (%507 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %509 = dfcir.mul[?] (%508 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %510 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 275 : i32, 264 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %511 = dfcir.cast[?] (%510 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %512 = dfcir.mul[?] (%511 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %513 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 263 : i32, 252 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %514 = dfcir.cast[?] (%513 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %515 = dfcir.mul[?] (%514 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %516 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 251 : i32, 240 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %517 = dfcir.cast[?] (%516 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %518 = dfcir.shl(%517 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %519 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 239 : i32, 228 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %520 = dfcir.cast[?] (%519 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %521 = dfcir.mul[?] (%520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %522 = dfcir.add[?] (%514 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %523 = dfcir.mul[?] (%522 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %524 = dfcir.sub[?] (%523 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %521 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %525 = dfcir.sub[?] (%523 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %515 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %526 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 227 : i32, 216 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %527 = dfcir.cast[?] (%526 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %528 = dfcir.mul[?] (%527 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %529 = dfcir.add[?] (%527 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %511 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %530 = dfcir.mul[?] (%529 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %531 = dfcir.add[?] (%530 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %528 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %532 = dfcir.sub[?] (%530 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %512 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %533 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 215 : i32, 204 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %534 = dfcir.cast[?] (%533 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %535 = dfcir.mul[?] (%534 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %536 = dfcir.add[?] (%534 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %508 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %537 = dfcir.mul[?] (%536 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %538 = dfcir.sub[?] (%537 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %509 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %539 = dfcir.sub[?] (%538 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %524 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %540 = dfcir.add[?] (%538 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %524 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %541 = dfcir.add[?] (%537 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %535 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %542 = dfcir.sub[?] (%541 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %525 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %543 = dfcir.sub[?] (%542 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %539 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %544 = dfcir.mul[?] (%543 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %545 = dfcir.add[?] (%544 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %546 = dfcir.shr(%545 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %547 = dfcir.add[?] (%542 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %539 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %548 = dfcir.mul[?] (%547 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %549 = dfcir.add[?] (%548 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %550 = dfcir.shr(%549 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %551 = dfcir.add[?] (%541 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %525 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %552 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 203 : i32, 192 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %553 = dfcir.cast[?] (%552 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %554 = dfcir.shl(%553 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %555 = dfcir.add[?] (%554 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %556 = dfcir.sub[?] (%555 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %557 = dfcir.sub[?] (%556 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %532 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %558 = dfcir.sub[?] (%557 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %546 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %559 = dfcir.shr(%558 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %560 = dfcir.cast[?] (%559 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %561 = dfcir.cast[?] (%560 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %562 = dfcir.mul[?] (%561 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %563 = dfcir.add[?] (%561 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %165 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %564 = dfcir.mul[?] (%563 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %565 = dfcir.add[?] (%557 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %546 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %566 = dfcir.shr(%565 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %567 = dfcir.cast[?] (%566 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %568 = dfcir.cast[?] (%567 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %569 = dfcir.mul[?] (%568 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %570 = dfcir.add[?] (%568 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %170 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %571 = dfcir.mul[?] (%570 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %572 = dfcir.add[?] (%556 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %532 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %573 = dfcir.sub[?] (%572 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %550 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %574 = dfcir.shr(%573 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %575 = dfcir.cast[?] (%574 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %576 = dfcir.cast[?] (%575 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %577 = dfcir.mul[?] (%576 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %578 = dfcir.add[?] (%576 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %176 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %579 = dfcir.mul[?] (%578 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %580 = dfcir.add[?] (%572 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %550 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %581 = dfcir.shr(%580 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %582 = dfcir.cast[?] (%581 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %583 = dfcir.cast[?] (%582 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %584 = dfcir.mul[?] (%583 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %585 = dfcir.add[?] (%583 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %181 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %586 = dfcir.mul[?] (%585 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %587 = dfcir.add[?] (%555 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %588 = dfcir.sub[?] (%587 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %531 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %589 = dfcir.sub[?] (%588 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %540 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %590 = dfcir.shr(%589 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %591 = dfcir.cast[?] (%590 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %592 = dfcir.cast[?] (%591 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %593 = dfcir.mul[?] (%592 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %594 = dfcir.add[?] (%592 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %188 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %595 = dfcir.mul[?] (%594 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %596 = dfcir.add[?] (%588 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %540 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %597 = dfcir.shr(%596 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %598 = dfcir.cast[?] (%597 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %599 = dfcir.cast[?] (%598 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %600 = dfcir.mul[?] (%599 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %601 = dfcir.add[?] (%599 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %193 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %602 = dfcir.mul[?] (%601 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %603 = dfcir.add[?] (%587 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %531 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %604 = dfcir.sub[?] (%603 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %551 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %605 = dfcir.shr(%604 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %606 = dfcir.cast[?] (%605 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %607 = dfcir.cast[?] (%606 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %608 = dfcir.mul[?] (%607 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %609 = dfcir.add[?] (%607 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %199 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %610 = dfcir.mul[?] (%609 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %611 = dfcir.add[?] (%603 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %551 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %612 = dfcir.shr(%611 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %613 = dfcir.cast[?] (%612 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %614 = dfcir.cast[?] (%613 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %615 = dfcir.mul[?] (%614 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %616 = dfcir.add[?] (%614 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %204 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %617 = dfcir.mul[?] (%616 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %618 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 191 : i32, 180 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %619 = dfcir.cast[?] (%618 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %620 = dfcir.mul[?] (%619 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %621 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 179 : i32, 168 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %622 = dfcir.cast[?] (%621 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %623 = dfcir.mul[?] (%622 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %624 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 167 : i32, 156 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %625 = dfcir.cast[?] (%624 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %626 = dfcir.mul[?] (%625 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %627 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 155 : i32, 144 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %628 = dfcir.cast[?] (%627 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %629 = dfcir.shl(%628 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %630 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 143 : i32, 132 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %631 = dfcir.cast[?] (%630 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %632 = dfcir.mul[?] (%631 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %633 = dfcir.add[?] (%625 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %631 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %634 = dfcir.mul[?] (%633 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %635 = dfcir.sub[?] (%634 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %632 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %636 = dfcir.sub[?] (%634 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %626 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %637 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 131 : i32, 120 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %638 = dfcir.cast[?] (%637 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %639 = dfcir.mul[?] (%638 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %640 = dfcir.add[?] (%638 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %622 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %641 = dfcir.mul[?] (%640 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %642 = dfcir.add[?] (%641 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %639 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %643 = dfcir.sub[?] (%641 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %623 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %644 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 119 : i32, 108 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %645 = dfcir.cast[?] (%644 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %646 = dfcir.mul[?] (%645 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %647 = dfcir.add[?] (%645 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %619 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %648 = dfcir.mul[?] (%647 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %649 = dfcir.sub[?] (%648 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %620 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %650 = dfcir.sub[?] (%649 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %635 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %651 = dfcir.add[?] (%649 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %635 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %652 = dfcir.add[?] (%648 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %646 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %653 = dfcir.sub[?] (%652 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %636 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %654 = dfcir.sub[?] (%653 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %650 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %655 = dfcir.mul[?] (%654 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %656 = dfcir.add[?] (%655 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %657 = dfcir.shr(%656 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %658 = dfcir.add[?] (%653 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %650 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %659 = dfcir.mul[?] (%658 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %660 = dfcir.add[?] (%659 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %661 = dfcir.shr(%660 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %662 = dfcir.add[?] (%652 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %636 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %663 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 107 : i32, 96 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %664 = dfcir.cast[?] (%663 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %665 = dfcir.shl(%664 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %666 = dfcir.add[?] (%665 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %667 = dfcir.sub[?] (%666 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %629 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %668 = dfcir.sub[?] (%667 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %643 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %669 = dfcir.sub[?] (%668 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %657 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %670 = dfcir.shr(%669 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %671 = dfcir.cast[?] (%670 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %672 = dfcir.cast[?] (%671 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %673 = dfcir.mul[?] (%672 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %674 = dfcir.add[?] (%672 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %70 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %675 = dfcir.mul[?] (%674 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %676 = dfcir.add[?] (%668 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %657 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %677 = dfcir.shr(%676 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %678 = dfcir.cast[?] (%677 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %679 = dfcir.cast[?] (%678 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %680 = dfcir.mul[?] (%679 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %681 = dfcir.add[?] (%679 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %75 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %682 = dfcir.mul[?] (%681 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %683 = dfcir.add[?] (%667 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %643 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %684 = dfcir.sub[?] (%683 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %661 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %685 = dfcir.shr(%684 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %686 = dfcir.cast[?] (%685 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %687 = dfcir.cast[?] (%686 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %688 = dfcir.mul[?] (%687 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %689 = dfcir.add[?] (%687 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %81 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %690 = dfcir.mul[?] (%689 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %691 = dfcir.add[?] (%683 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %661 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %692 = dfcir.shr(%691 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %693 = dfcir.cast[?] (%692 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %694 = dfcir.cast[?] (%693 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %695 = dfcir.mul[?] (%694 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %696 = dfcir.add[?] (%694 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %86 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %697 = dfcir.mul[?] (%696 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %698 = dfcir.add[?] (%666 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %629 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %699 = dfcir.sub[?] (%698 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %642 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %700 = dfcir.sub[?] (%699 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %651 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %701 = dfcir.shr(%700 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %702 = dfcir.cast[?] (%701 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %703 = dfcir.cast[?] (%702 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %704 = dfcir.mul[?] (%703 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %705 = dfcir.add[?] (%703 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %93 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %706 = dfcir.mul[?] (%705 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %707 = dfcir.add[?] (%699 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %651 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %708 = dfcir.shr(%707 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %709 = dfcir.cast[?] (%708 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %710 = dfcir.cast[?] (%709 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %711 = dfcir.mul[?] (%710 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %712 = dfcir.add[?] (%710 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %98 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %713 = dfcir.mul[?] (%712 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %714 = dfcir.add[?] (%698 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %642 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %715 = dfcir.sub[?] (%714 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %662 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %716 = dfcir.shr(%715 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %717 = dfcir.cast[?] (%716 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %718 = dfcir.cast[?] (%717 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %719 = dfcir.mul[?] (%718 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %720 = dfcir.add[?] (%718 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %104 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %721 = dfcir.mul[?] (%720 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %722 = dfcir.add[?] (%714 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %662 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %723 = dfcir.shr(%722 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %724 = dfcir.cast[?] (%723 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %725 = dfcir.cast[?] (%724 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %726 = dfcir.mul[?] (%725 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %727 = dfcir.add[?] (%725 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %109 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %728 = dfcir.mul[?] (%727 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %729 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 95 : i32, 84 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %730 = dfcir.cast[?] (%729 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %731 = dfcir.mul[?] (%730 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %14 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %732 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 83 : i32, 72 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %733 = dfcir.cast[?] (%732 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %734 = dfcir.mul[?] (%733 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %0 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %735 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 71 : i32, 60 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %736 = dfcir.cast[?] (%735 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %737 = dfcir.mul[?] (%736 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %2 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %738 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 59 : i32, 48 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %739 = dfcir.cast[?] (%738 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %740 = dfcir.shl(%739 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %741 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 47 : i32, 36 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %742 = dfcir.cast[?] (%741 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %743 = dfcir.mul[?] (%742 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %744 = dfcir.add[?] (%736 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %742 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %745 = dfcir.mul[?] (%744 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %10 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %746 = dfcir.sub[?] (%745 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %743 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %747 = dfcir.sub[?] (%745 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %737 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %748 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 35 : i32, 24 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %749 = dfcir.cast[?] (%748 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %750 = dfcir.mul[?] (%749 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %11 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %751 = dfcir.add[?] (%749 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %733 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %752 = dfcir.mul[?] (%751 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %6 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %753 = dfcir.add[?] (%752 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %750 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %754 = dfcir.sub[?] (%752 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %734 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %755 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 23 : i32, 12 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %756 = dfcir.cast[?] (%755 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %757 = dfcir.mul[?] (%756 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %3 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %758 = dfcir.add[?] (%756 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %730 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %759 = dfcir.mul[?] (%758 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %13 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %760 = dfcir.sub[?] (%759 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %731 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %761 = dfcir.sub[?] (%760 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %746 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %762 = dfcir.add[?] (%760 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %746 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %763 = dfcir.add[?] (%759 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %757 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %764 = dfcir.sub[?] (%763 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %747 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %765 = dfcir.sub[?] (%764 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %761 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %766 = dfcir.mul[?] (%765 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %767 = dfcir.add[?] (%766 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %768 = dfcir.shr(%767 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %769 = dfcir.add[?] (%764 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %761 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %770 = dfcir.mul[?] (%769 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %771 = dfcir.add[?] (%770 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %772 = dfcir.shr(%771 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %773 = dfcir.add[?] (%763 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %747 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %774 = dfcir.bits(%15 : !dfcir.stream<!dfcir.fixed<true, 767, 0>>, 11 : i32, 0 : i32) : !dfcir.stream<!dfcir.rawbits<12>>
    %775 = dfcir.cast[?] (%774 : !dfcir.stream<!dfcir.rawbits<12>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %776 = dfcir.shl(%775 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 11 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %777 = dfcir.add[?] (%776 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %778 = dfcir.sub[?] (%777 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %740 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %779 = dfcir.sub[?] (%778 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %754 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %780 = dfcir.sub[?] (%779 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %768 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %781 = dfcir.shr(%780 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %782 = dfcir.cast[?] (%781 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %783 = dfcir.cast[?] (%782 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %784 = dfcir.shl(%783 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %785 = dfcir.add[?] (%784 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %786 = dfcir.sub[?] (%785 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %356 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %787 = dfcir.add[?] (%785 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %356 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %788 = dfcir.add[?] (%779 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %768 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %789 = dfcir.shr(%788 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %790 = dfcir.cast[?] (%789 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %791 = dfcir.cast[?] (%790 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %792 = dfcir.shl(%791 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %793 = dfcir.add[?] (%792 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %794 = dfcir.sub[?] (%793 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %361 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %795 = dfcir.add[?] (%793 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %361 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %796 = dfcir.add[?] (%778 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %754 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %797 = dfcir.sub[?] (%796 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %772 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %798 = dfcir.shr(%797 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %799 = dfcir.cast[?] (%798 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %800 = dfcir.cast[?] (%799 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %801 = dfcir.shl(%800 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %802 = dfcir.add[?] (%801 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %803 = dfcir.sub[?] (%802 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %367 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %804 = dfcir.add[?] (%802 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %367 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %805 = dfcir.add[?] (%796 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %772 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %806 = dfcir.shr(%805 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %807 = dfcir.cast[?] (%806 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %808 = dfcir.cast[?] (%807 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %809 = dfcir.shl(%808 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %810 = dfcir.add[?] (%809 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %811 = dfcir.sub[?] (%810 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %372 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %812 = dfcir.add[?] (%810 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %372 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %813 = dfcir.add[?] (%777 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %740 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %814 = dfcir.sub[?] (%813 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %753 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %815 = dfcir.sub[?] (%814 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %762 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %816 = dfcir.shr(%815 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %817 = dfcir.cast[?] (%816 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %818 = dfcir.cast[?] (%817 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %819 = dfcir.shl(%818 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %820 = dfcir.add[?] (%819 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %821 = dfcir.sub[?] (%820 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %379 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %822 = dfcir.add[?] (%820 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %379 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %823 = dfcir.add[?] (%814 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %762 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %824 = dfcir.shr(%823 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %825 = dfcir.cast[?] (%824 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %826 = dfcir.cast[?] (%825 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %827 = dfcir.shl(%826 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %828 = dfcir.add[?] (%827 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %829 = dfcir.sub[?] (%828 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %384 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %830 = dfcir.add[?] (%828 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %384 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %831 = dfcir.add[?] (%813 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %753 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %832 = dfcir.sub[?] (%831 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %773 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %833 = dfcir.shr(%832 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %834 = dfcir.cast[?] (%833 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %835 = dfcir.cast[?] (%834 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %836 = dfcir.shl(%835 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %837 = dfcir.add[?] (%836 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %838 = dfcir.sub[?] (%837 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %390 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %839 = dfcir.add[?] (%837 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %390 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %840 = dfcir.add[?] (%831 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %773 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %841 = dfcir.shr(%840 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %842 = dfcir.cast[?] (%841 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %843 = dfcir.cast[?] (%842 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %844 = dfcir.shl(%843 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %845 = dfcir.add[?] (%844 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %5 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %846 = dfcir.sub[?] (%845 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %395 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %847 = dfcir.add[?] (%845 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %395 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %848 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 2676 : si32
    %849 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 4 : si32
    %850 = dfcir.add[?] (%610 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %851 = dfcir.add[?] (%850 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %608 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %852 = dfcir.shr(%851 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %853 = dfcir.sub[?] (%839 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %852 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %854 = dfcir.add[?] (%839 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %852 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %855 = dfcir.sub[?] (%850 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %200 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %856 = dfcir.shr(%855 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %857 = dfcir.sub[?] (%838 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %856 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %858 = dfcir.add[?] (%838 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %856 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %859 = dfcir.add[?] (%499 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %860 = dfcir.sub[?] (%859 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %497 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %861 = dfcir.shr(%860 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %862 = dfcir.sub[?] (%859 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %295 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %863 = dfcir.shr(%862 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %864 = dfcir.add[?] (%721 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %865 = dfcir.sub[?] (%864 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %105 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %866 = dfcir.shr(%865 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %867 = dfcir.sub[?] (%866 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %861 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %868 = dfcir.add[?] (%866 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %861 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %869 = dfcir.sub[?] (%853 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %868 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %870 = dfcir.shr(%869 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %871 = dfcir.cast[?] (%870 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %872 = dfcir.greater[?] (%871 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %873 = dfcir.cast[?] (%872 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %874 = dfcir.greaterEq[?] (%871 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %875 = dfcir.cast[?] (%874 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %876 = dfcir.add[?] (%875 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %873 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %877 = dfcir.mux(%876: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %871, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %878 = dfcir.cast[?] (%877 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %879 = dfcir.add[?] (%853 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %868 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %880 = dfcir.shr(%879 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %881 = dfcir.cast[?] (%880 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %882 = dfcir.greater[?] (%881 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %883 = dfcir.cast[?] (%882 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %884 = dfcir.greaterEq[?] (%881 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %885 = dfcir.cast[?] (%884 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %886 = dfcir.add[?] (%885 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %883 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %887 = dfcir.mux(%886: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %881, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %888 = dfcir.cast[?] (%887 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %889 = dfcir.add[?] (%864 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %719 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %890 = dfcir.shr(%889 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %891 = dfcir.sub[?] (%890 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %863 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %892 = dfcir.sub[?] (%891 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %867 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %893 = dfcir.mul[?] (%892 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %894 = dfcir.add[?] (%893 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %895 = dfcir.shr(%894 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %896 = dfcir.sub[?] (%857 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %895 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %897 = dfcir.shr(%896 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %898 = dfcir.cast[?] (%897 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %899 = dfcir.greater[?] (%898 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %900 = dfcir.cast[?] (%899 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %901 = dfcir.greaterEq[?] (%898 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %902 = dfcir.cast[?] (%901 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %903 = dfcir.add[?] (%902 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %900 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %904 = dfcir.mux(%903: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %898, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %905 = dfcir.cast[?] (%904 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %906 = dfcir.add[?] (%857 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %895 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %907 = dfcir.shr(%906 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %908 = dfcir.cast[?] (%907 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %909 = dfcir.greater[?] (%908 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %910 = dfcir.cast[?] (%909 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %911 = dfcir.greaterEq[?] (%908 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %912 = dfcir.cast[?] (%911 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %913 = dfcir.add[?] (%912 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %910 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %914 = dfcir.mux(%913: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %908, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %915 = dfcir.cast[?] (%914 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %916 = dfcir.add[?] (%891 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %867 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %917 = dfcir.mul[?] (%916 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %918 = dfcir.add[?] (%917 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %919 = dfcir.shr(%918 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %920 = dfcir.sub[?] (%858 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %919 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %921 = dfcir.shr(%920 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %922 = dfcir.cast[?] (%921 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %923 = dfcir.greater[?] (%922 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %924 = dfcir.cast[?] (%923 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %925 = dfcir.greaterEq[?] (%922 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %926 = dfcir.cast[?] (%925 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %927 = dfcir.add[?] (%926 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %924 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %928 = dfcir.mux(%927: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %922, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %929 = dfcir.cast[?] (%928 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %930 = dfcir.add[?] (%858 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %919 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %931 = dfcir.shr(%930 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %932 = dfcir.cast[?] (%931 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %933 = dfcir.greater[?] (%932 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %934 = dfcir.cast[?] (%933 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %935 = dfcir.greaterEq[?] (%932 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %936 = dfcir.cast[?] (%935 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %937 = dfcir.add[?] (%936 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %934 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %938 = dfcir.mux(%937: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %932, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %939 = dfcir.cast[?] (%938 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %940 = dfcir.add[?] (%890 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %863 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %941 = dfcir.sub[?] (%854 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %940 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %942 = dfcir.shr(%941 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %943 = dfcir.cast[?] (%942 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %944 = dfcir.greater[?] (%943 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %945 = dfcir.cast[?] (%944 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %946 = dfcir.greaterEq[?] (%943 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %947 = dfcir.cast[?] (%946 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %948 = dfcir.add[?] (%947 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %945 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %949 = dfcir.mux(%948: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %943, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %950 = dfcir.cast[?] (%949 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %951 = dfcir.add[?] (%854 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %940 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %952 = dfcir.shr(%951 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %953 = dfcir.cast[?] (%952 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %954 = dfcir.greater[?] (%953 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %955 = dfcir.cast[?] (%954 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %956 = dfcir.greaterEq[?] (%953 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %957 = dfcir.cast[?] (%956 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %958 = dfcir.add[?] (%957 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %955 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %959 = dfcir.mux(%958: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %953, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %960 = dfcir.cast[?] (%959 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %961 = dfcir.add[?] (%579 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %962 = dfcir.add[?] (%961 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %577 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %963 = dfcir.shr(%962 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %964 = dfcir.sub[?] (%804 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %963 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %965 = dfcir.add[?] (%804 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %963 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %966 = dfcir.sub[?] (%961 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %177 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %967 = dfcir.shr(%966 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %968 = dfcir.sub[?] (%803 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %967 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %969 = dfcir.add[?] (%803 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %967 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %970 = dfcir.add[?] (%468 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %971 = dfcir.sub[?] (%970 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %466 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %972 = dfcir.shr(%971 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %973 = dfcir.sub[?] (%970 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %272 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %974 = dfcir.shr(%973 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %975 = dfcir.add[?] (%690 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %976 = dfcir.sub[?] (%975 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %82 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %977 = dfcir.shr(%976 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %978 = dfcir.sub[?] (%977 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %972 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %979 = dfcir.add[?] (%977 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %972 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %980 = dfcir.sub[?] (%964 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %979 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %981 = dfcir.shr(%980 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %982 = dfcir.cast[?] (%981 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %983 = dfcir.greater[?] (%982 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %984 = dfcir.cast[?] (%983 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %985 = dfcir.greaterEq[?] (%982 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %986 = dfcir.cast[?] (%985 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %987 = dfcir.add[?] (%986 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %984 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %988 = dfcir.mux(%987: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %982, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %989 = dfcir.cast[?] (%988 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %990 = dfcir.add[?] (%964 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %979 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %991 = dfcir.shr(%990 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %992 = dfcir.cast[?] (%991 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %993 = dfcir.greater[?] (%992 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %994 = dfcir.cast[?] (%993 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %995 = dfcir.greaterEq[?] (%992 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %996 = dfcir.cast[?] (%995 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %997 = dfcir.add[?] (%996 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %994 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %998 = dfcir.mux(%997: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %992, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %999 = dfcir.cast[?] (%998 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1000 = dfcir.add[?] (%975 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %688 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1001 = dfcir.shr(%1000 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1002 = dfcir.sub[?] (%1001 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %974 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1003 = dfcir.sub[?] (%1002 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %978 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1004 = dfcir.mul[?] (%1003 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1005 = dfcir.add[?] (%1004 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1006 = dfcir.shr(%1005 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1007 = dfcir.sub[?] (%968 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1006 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1008 = dfcir.shr(%1007 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1009 = dfcir.cast[?] (%1008 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1010 = dfcir.greater[?] (%1009 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1011 = dfcir.cast[?] (%1010 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1012 = dfcir.greaterEq[?] (%1009 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1013 = dfcir.cast[?] (%1012 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1014 = dfcir.add[?] (%1013 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1011 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1015 = dfcir.mux(%1014: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1009, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1016 = dfcir.cast[?] (%1015 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1017 = dfcir.add[?] (%968 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1006 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1018 = dfcir.shr(%1017 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1019 = dfcir.cast[?] (%1018 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1020 = dfcir.greater[?] (%1019 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1021 = dfcir.cast[?] (%1020 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1022 = dfcir.greaterEq[?] (%1019 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1023 = dfcir.cast[?] (%1022 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1024 = dfcir.add[?] (%1023 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1021 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1025 = dfcir.mux(%1024: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1019, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1026 = dfcir.cast[?] (%1025 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1027 = dfcir.add[?] (%1002 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %978 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1028 = dfcir.mul[?] (%1027 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1029 = dfcir.add[?] (%1028 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1030 = dfcir.shr(%1029 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1031 = dfcir.sub[?] (%969 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1030 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1032 = dfcir.shr(%1031 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1033 = dfcir.cast[?] (%1032 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1034 = dfcir.greater[?] (%1033 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1035 = dfcir.cast[?] (%1034 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1036 = dfcir.greaterEq[?] (%1033 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1037 = dfcir.cast[?] (%1036 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1038 = dfcir.add[?] (%1037 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1035 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1039 = dfcir.mux(%1038: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1033, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1040 = dfcir.cast[?] (%1039 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1041 = dfcir.add[?] (%969 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1030 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1042 = dfcir.shr(%1041 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1043 = dfcir.cast[?] (%1042 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1044 = dfcir.greater[?] (%1043 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1045 = dfcir.cast[?] (%1044 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1046 = dfcir.greaterEq[?] (%1043 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1047 = dfcir.cast[?] (%1046 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1048 = dfcir.add[?] (%1047 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1045 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1049 = dfcir.mux(%1048: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1043, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1050 = dfcir.cast[?] (%1049 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1051 = dfcir.add[?] (%1001 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %974 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1052 = dfcir.sub[?] (%965 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1051 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1053 = dfcir.shr(%1052 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1054 = dfcir.cast[?] (%1053 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1055 = dfcir.greater[?] (%1054 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1056 = dfcir.cast[?] (%1055 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1057 = dfcir.greaterEq[?] (%1054 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1058 = dfcir.cast[?] (%1057 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1059 = dfcir.add[?] (%1058 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1056 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1060 = dfcir.mux(%1059: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1054, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1061 = dfcir.cast[?] (%1060 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1062 = dfcir.add[?] (%965 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1051 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1063 = dfcir.shr(%1062 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1064 = dfcir.cast[?] (%1063 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1065 = dfcir.greater[?] (%1064 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1066 = dfcir.cast[?] (%1065 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1067 = dfcir.greaterEq[?] (%1064 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1068 = dfcir.cast[?] (%1067 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1069 = dfcir.add[?] (%1068 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1066 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1070 = dfcir.mux(%1069: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1064, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1071 = dfcir.cast[?] (%1070 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1072 = dfcir.add[?] (%564 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1073 = dfcir.add[?] (%1072 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %562 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1074 = dfcir.shr(%1073 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1075 = dfcir.sub[?] (%787 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1074 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1076 = dfcir.add[?] (%787 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1074 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1077 = dfcir.sub[?] (%1072 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %166 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1078 = dfcir.shr(%1077 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1079 = dfcir.sub[?] (%786 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1078 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1080 = dfcir.add[?] (%786 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1078 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1081 = dfcir.add[?] (%453 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1082 = dfcir.sub[?] (%1081 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %451 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1083 = dfcir.shr(%1082 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1084 = dfcir.sub[?] (%1081 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %261 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1085 = dfcir.shr(%1084 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1086 = dfcir.add[?] (%675 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1087 = dfcir.sub[?] (%1086 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %71 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1088 = dfcir.shr(%1087 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1089 = dfcir.sub[?] (%1088 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1083 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1090 = dfcir.add[?] (%1088 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1083 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1091 = dfcir.sub[?] (%1075 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1090 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1092 = dfcir.shr(%1091 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1093 = dfcir.cast[?] (%1092 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1094 = dfcir.greater[?] (%1093 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1095 = dfcir.cast[?] (%1094 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1096 = dfcir.greaterEq[?] (%1093 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1097 = dfcir.cast[?] (%1096 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1098 = dfcir.add[?] (%1097 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1095 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1099 = dfcir.mux(%1098: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1093, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1100 = dfcir.cast[?] (%1099 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1101 = dfcir.add[?] (%1075 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1090 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1102 = dfcir.shr(%1101 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1103 = dfcir.cast[?] (%1102 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1104 = dfcir.greater[?] (%1103 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1105 = dfcir.cast[?] (%1104 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1106 = dfcir.greaterEq[?] (%1103 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1107 = dfcir.cast[?] (%1106 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1108 = dfcir.add[?] (%1107 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1105 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1109 = dfcir.mux(%1108: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1103, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1110 = dfcir.cast[?] (%1109 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1111 = dfcir.add[?] (%1086 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %673 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1112 = dfcir.shr(%1111 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1113 = dfcir.sub[?] (%1112 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1085 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1114 = dfcir.sub[?] (%1113 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1089 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1115 = dfcir.mul[?] (%1114 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1116 = dfcir.add[?] (%1115 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1117 = dfcir.shr(%1116 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1118 = dfcir.sub[?] (%1079 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1117 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1119 = dfcir.shr(%1118 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1120 = dfcir.cast[?] (%1119 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1121 = dfcir.greater[?] (%1120 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1122 = dfcir.cast[?] (%1121 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1123 = dfcir.greaterEq[?] (%1120 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1124 = dfcir.cast[?] (%1123 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1125 = dfcir.add[?] (%1124 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1122 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1126 = dfcir.mux(%1125: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1120, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1127 = dfcir.cast[?] (%1126 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1128 = dfcir.add[?] (%1079 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1117 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1129 = dfcir.shr(%1128 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1130 = dfcir.cast[?] (%1129 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1131 = dfcir.greater[?] (%1130 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1132 = dfcir.cast[?] (%1131 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1133 = dfcir.greaterEq[?] (%1130 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1134 = dfcir.cast[?] (%1133 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1135 = dfcir.add[?] (%1134 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1132 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1136 = dfcir.mux(%1135: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1130, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1137 = dfcir.cast[?] (%1136 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1138 = dfcir.add[?] (%1113 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1089 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1139 = dfcir.mul[?] (%1138 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1140 = dfcir.add[?] (%1139 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1141 = dfcir.shr(%1140 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1142 = dfcir.sub[?] (%1080 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1141 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1143 = dfcir.shr(%1142 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1144 = dfcir.cast[?] (%1143 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1145 = dfcir.greater[?] (%1144 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1146 = dfcir.cast[?] (%1145 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1147 = dfcir.greaterEq[?] (%1144 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1148 = dfcir.cast[?] (%1147 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1149 = dfcir.add[?] (%1148 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1146 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1150 = dfcir.mux(%1149: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1144, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1151 = dfcir.cast[?] (%1150 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1152 = dfcir.add[?] (%1080 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1141 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1153 = dfcir.shr(%1152 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1154 = dfcir.cast[?] (%1153 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1155 = dfcir.greater[?] (%1154 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1156 = dfcir.cast[?] (%1155 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1157 = dfcir.greaterEq[?] (%1154 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1158 = dfcir.cast[?] (%1157 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1159 = dfcir.add[?] (%1158 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1156 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1160 = dfcir.mux(%1159: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1154, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1161 = dfcir.cast[?] (%1160 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1162 = dfcir.add[?] (%1112 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1085 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1163 = dfcir.sub[?] (%1076 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1162 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1164 = dfcir.shr(%1163 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1165 = dfcir.cast[?] (%1164 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1166 = dfcir.greater[?] (%1165 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1167 = dfcir.cast[?] (%1166 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1168 = dfcir.greaterEq[?] (%1165 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1169 = dfcir.cast[?] (%1168 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1170 = dfcir.add[?] (%1169 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1167 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1171 = dfcir.mux(%1170: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1165, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1172 = dfcir.cast[?] (%1171 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1173 = dfcir.add[?] (%1076 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1162 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1174 = dfcir.shr(%1173 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1175 = dfcir.cast[?] (%1174 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1176 = dfcir.greater[?] (%1175 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1177 = dfcir.cast[?] (%1176 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1178 = dfcir.greaterEq[?] (%1175 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1179 = dfcir.cast[?] (%1178 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1180 = dfcir.add[?] (%1179 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1177 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1181 = dfcir.mux(%1180: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1175, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1182 = dfcir.cast[?] (%1181 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1183 = dfcir.add[?] (%595 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1184 = dfcir.add[?] (%1183 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %593 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1185 = dfcir.shr(%1184 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1186 = dfcir.sub[?] (%822 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1185 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1187 = dfcir.add[?] (%822 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1185 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1188 = dfcir.sub[?] (%1183 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %189 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1189 = dfcir.shr(%1188 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1190 = dfcir.sub[?] (%821 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1189 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1191 = dfcir.add[?] (%821 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1189 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1192 = dfcir.add[?] (%484 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1193 = dfcir.sub[?] (%1192 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %482 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1194 = dfcir.shr(%1193 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1195 = dfcir.sub[?] (%1192 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %284 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1196 = dfcir.shr(%1195 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1197 = dfcir.add[?] (%706 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1198 = dfcir.sub[?] (%1197 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %94 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1199 = dfcir.shr(%1198 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1200 = dfcir.sub[?] (%1199 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1194 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1201 = dfcir.add[?] (%1199 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1194 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1202 = dfcir.sub[?] (%1186 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1201 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1203 = dfcir.shr(%1202 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1204 = dfcir.cast[?] (%1203 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1205 = dfcir.greater[?] (%1204 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1206 = dfcir.cast[?] (%1205 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1207 = dfcir.greaterEq[?] (%1204 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1208 = dfcir.cast[?] (%1207 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1209 = dfcir.add[?] (%1208 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1206 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1210 = dfcir.mux(%1209: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1204, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1211 = dfcir.cast[?] (%1210 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1212 = dfcir.add[?] (%1186 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1201 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1213 = dfcir.shr(%1212 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1214 = dfcir.cast[?] (%1213 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1215 = dfcir.greater[?] (%1214 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1216 = dfcir.cast[?] (%1215 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1217 = dfcir.greaterEq[?] (%1214 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1218 = dfcir.cast[?] (%1217 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1219 = dfcir.add[?] (%1218 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1216 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1220 = dfcir.mux(%1219: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1214, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1221 = dfcir.cast[?] (%1220 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1222 = dfcir.add[?] (%1197 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %704 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1223 = dfcir.shr(%1222 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1224 = dfcir.sub[?] (%1223 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1196 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1225 = dfcir.sub[?] (%1224 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1200 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1226 = dfcir.mul[?] (%1225 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1227 = dfcir.add[?] (%1226 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1228 = dfcir.shr(%1227 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1229 = dfcir.sub[?] (%1190 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1228 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1230 = dfcir.shr(%1229 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1231 = dfcir.cast[?] (%1230 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1232 = dfcir.greater[?] (%1231 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1233 = dfcir.cast[?] (%1232 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1234 = dfcir.greaterEq[?] (%1231 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1235 = dfcir.cast[?] (%1234 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1236 = dfcir.add[?] (%1235 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1233 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1237 = dfcir.mux(%1236: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1231, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1238 = dfcir.cast[?] (%1237 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1239 = dfcir.add[?] (%1190 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1228 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1240 = dfcir.shr(%1239 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1241 = dfcir.cast[?] (%1240 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1242 = dfcir.greater[?] (%1241 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1243 = dfcir.cast[?] (%1242 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1244 = dfcir.greaterEq[?] (%1241 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1245 = dfcir.cast[?] (%1244 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1246 = dfcir.add[?] (%1245 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1243 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1247 = dfcir.mux(%1246: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1241, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1248 = dfcir.cast[?] (%1247 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1249 = dfcir.add[?] (%1224 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1200 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1250 = dfcir.mul[?] (%1249 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1251 = dfcir.add[?] (%1250 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1252 = dfcir.shr(%1251 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1253 = dfcir.sub[?] (%1191 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1252 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1254 = dfcir.shr(%1253 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1255 = dfcir.cast[?] (%1254 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1256 = dfcir.greater[?] (%1255 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1257 = dfcir.cast[?] (%1256 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1258 = dfcir.greaterEq[?] (%1255 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1259 = dfcir.cast[?] (%1258 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1260 = dfcir.add[?] (%1259 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1257 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1261 = dfcir.mux(%1260: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1255, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1262 = dfcir.cast[?] (%1261 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1263 = dfcir.add[?] (%1191 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1252 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1264 = dfcir.shr(%1263 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1265 = dfcir.cast[?] (%1264 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1266 = dfcir.greater[?] (%1265 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1267 = dfcir.cast[?] (%1266 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1268 = dfcir.greaterEq[?] (%1265 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1269 = dfcir.cast[?] (%1268 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1270 = dfcir.add[?] (%1269 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1267 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1271 = dfcir.mux(%1270: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1265, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1272 = dfcir.cast[?] (%1271 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1273 = dfcir.add[?] (%1223 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1196 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1274 = dfcir.sub[?] (%1187 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1273 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1275 = dfcir.shr(%1274 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1276 = dfcir.cast[?] (%1275 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1277 = dfcir.greater[?] (%1276 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1278 = dfcir.cast[?] (%1277 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1279 = dfcir.greaterEq[?] (%1276 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1280 = dfcir.cast[?] (%1279 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1281 = dfcir.add[?] (%1280 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1278 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1282 = dfcir.mux(%1281: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1276, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1283 = dfcir.cast[?] (%1282 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1284 = dfcir.add[?] (%1187 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1273 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1285 = dfcir.shr(%1284 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1286 = dfcir.cast[?] (%1285 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1287 = dfcir.greater[?] (%1286 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1288 = dfcir.cast[?] (%1287 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1289 = dfcir.greaterEq[?] (%1286 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1290 = dfcir.cast[?] (%1289 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1291 = dfcir.add[?] (%1290 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1288 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1292 = dfcir.mux(%1291: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1286, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1293 = dfcir.cast[?] (%1292 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1294 = dfcir.add[?] (%602 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1295 = dfcir.add[?] (%1294 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %600 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1296 = dfcir.shr(%1295 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1297 = dfcir.sub[?] (%830 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1296 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1298 = dfcir.add[?] (%830 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1296 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1299 = dfcir.sub[?] (%1294 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %194 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1300 = dfcir.shr(%1299 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1301 = dfcir.sub[?] (%829 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1300 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1302 = dfcir.add[?] (%829 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1300 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1303 = dfcir.add[?] (%491 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1304 = dfcir.sub[?] (%1303 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %489 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1305 = dfcir.shr(%1304 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1306 = dfcir.sub[?] (%1303 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %289 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1307 = dfcir.shr(%1306 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1308 = dfcir.add[?] (%713 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1309 = dfcir.sub[?] (%1308 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %99 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1310 = dfcir.shr(%1309 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1311 = dfcir.sub[?] (%1310 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1305 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1312 = dfcir.add[?] (%1310 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1305 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1313 = dfcir.sub[?] (%1297 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1312 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1314 = dfcir.shr(%1313 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1315 = dfcir.cast[?] (%1314 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1316 = dfcir.greater[?] (%1315 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1317 = dfcir.cast[?] (%1316 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1318 = dfcir.greaterEq[?] (%1315 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1319 = dfcir.cast[?] (%1318 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1320 = dfcir.add[?] (%1319 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1317 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1321 = dfcir.mux(%1320: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1315, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1322 = dfcir.cast[?] (%1321 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1323 = dfcir.add[?] (%1297 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1312 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1324 = dfcir.shr(%1323 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1325 = dfcir.cast[?] (%1324 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1326 = dfcir.greater[?] (%1325 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1327 = dfcir.cast[?] (%1326 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1328 = dfcir.greaterEq[?] (%1325 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1329 = dfcir.cast[?] (%1328 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1330 = dfcir.add[?] (%1329 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1327 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1331 = dfcir.mux(%1330: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1325, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1332 = dfcir.cast[?] (%1331 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1333 = dfcir.add[?] (%1308 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %711 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1334 = dfcir.shr(%1333 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1335 = dfcir.sub[?] (%1334 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1307 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1336 = dfcir.sub[?] (%1335 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1311 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1337 = dfcir.mul[?] (%1336 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1338 = dfcir.add[?] (%1337 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1339 = dfcir.shr(%1338 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1340 = dfcir.sub[?] (%1301 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1339 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1341 = dfcir.shr(%1340 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1342 = dfcir.cast[?] (%1341 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1343 = dfcir.greater[?] (%1342 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1344 = dfcir.cast[?] (%1343 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1345 = dfcir.greaterEq[?] (%1342 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1346 = dfcir.cast[?] (%1345 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1347 = dfcir.add[?] (%1346 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1344 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1348 = dfcir.mux(%1347: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1342, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1349 = dfcir.cast[?] (%1348 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1350 = dfcir.add[?] (%1301 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1339 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1351 = dfcir.shr(%1350 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1352 = dfcir.cast[?] (%1351 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1353 = dfcir.greater[?] (%1352 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1354 = dfcir.cast[?] (%1353 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1355 = dfcir.greaterEq[?] (%1352 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1356 = dfcir.cast[?] (%1355 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1357 = dfcir.add[?] (%1356 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1354 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1358 = dfcir.mux(%1357: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1352, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1359 = dfcir.cast[?] (%1358 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1360 = dfcir.add[?] (%1335 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1311 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1361 = dfcir.mul[?] (%1360 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1362 = dfcir.add[?] (%1361 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1363 = dfcir.shr(%1362 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1364 = dfcir.sub[?] (%1302 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1363 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1365 = dfcir.shr(%1364 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1366 = dfcir.cast[?] (%1365 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1367 = dfcir.greater[?] (%1366 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1368 = dfcir.cast[?] (%1367 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1369 = dfcir.greaterEq[?] (%1366 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1370 = dfcir.cast[?] (%1369 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1371 = dfcir.add[?] (%1370 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1368 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1372 = dfcir.mux(%1371: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1366, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1373 = dfcir.cast[?] (%1372 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1374 = dfcir.add[?] (%1302 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1363 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1375 = dfcir.shr(%1374 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1376 = dfcir.cast[?] (%1375 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1377 = dfcir.greater[?] (%1376 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1378 = dfcir.cast[?] (%1377 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1379 = dfcir.greaterEq[?] (%1376 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1380 = dfcir.cast[?] (%1379 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1381 = dfcir.add[?] (%1380 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1378 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1382 = dfcir.mux(%1381: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1376, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1383 = dfcir.cast[?] (%1382 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1384 = dfcir.add[?] (%1334 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1307 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1385 = dfcir.sub[?] (%1298 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1384 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1386 = dfcir.shr(%1385 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1387 = dfcir.cast[?] (%1386 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1388 = dfcir.greater[?] (%1387 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1389 = dfcir.cast[?] (%1388 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1390 = dfcir.greaterEq[?] (%1387 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1391 = dfcir.cast[?] (%1390 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1392 = dfcir.add[?] (%1391 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1389 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1393 = dfcir.mux(%1392: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1387, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1394 = dfcir.cast[?] (%1393 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1395 = dfcir.add[?] (%1298 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1384 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1396 = dfcir.shr(%1395 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1397 = dfcir.cast[?] (%1396 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1398 = dfcir.greater[?] (%1397 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1399 = dfcir.cast[?] (%1398 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1400 = dfcir.greaterEq[?] (%1397 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1401 = dfcir.cast[?] (%1400 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1402 = dfcir.add[?] (%1401 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1399 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1403 = dfcir.mux(%1402: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1397, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1404 = dfcir.cast[?] (%1403 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1405 = dfcir.add[?] (%571 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1406 = dfcir.add[?] (%1405 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %569 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1407 = dfcir.shr(%1406 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1408 = dfcir.sub[?] (%795 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1407 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1409 = dfcir.add[?] (%795 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1407 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1410 = dfcir.sub[?] (%1405 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %171 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1411 = dfcir.shr(%1410 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1412 = dfcir.sub[?] (%794 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1411 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1413 = dfcir.add[?] (%794 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1411 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1414 = dfcir.add[?] (%460 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1415 = dfcir.sub[?] (%1414 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %458 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1416 = dfcir.shr(%1415 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1417 = dfcir.sub[?] (%1414 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %266 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1418 = dfcir.shr(%1417 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1419 = dfcir.add[?] (%682 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1420 = dfcir.sub[?] (%1419 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %76 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1421 = dfcir.shr(%1420 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1422 = dfcir.sub[?] (%1421 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1423 = dfcir.add[?] (%1421 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1416 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1424 = dfcir.sub[?] (%1408 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1423 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1425 = dfcir.shr(%1424 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1426 = dfcir.cast[?] (%1425 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1427 = dfcir.greater[?] (%1426 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1428 = dfcir.cast[?] (%1427 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1429 = dfcir.greaterEq[?] (%1426 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1430 = dfcir.cast[?] (%1429 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1431 = dfcir.add[?] (%1430 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1428 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1432 = dfcir.mux(%1431: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1426, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1433 = dfcir.cast[?] (%1432 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1434 = dfcir.add[?] (%1408 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1423 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1435 = dfcir.shr(%1434 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1436 = dfcir.cast[?] (%1435 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1437 = dfcir.greater[?] (%1436 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1438 = dfcir.cast[?] (%1437 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1439 = dfcir.greaterEq[?] (%1436 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1440 = dfcir.cast[?] (%1439 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1441 = dfcir.add[?] (%1440 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1438 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1442 = dfcir.mux(%1441: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1436, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1443 = dfcir.cast[?] (%1442 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1444 = dfcir.add[?] (%1419 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %680 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1445 = dfcir.shr(%1444 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1446 = dfcir.sub[?] (%1445 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1418 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1447 = dfcir.sub[?] (%1446 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1422 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1448 = dfcir.mul[?] (%1447 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1449 = dfcir.add[?] (%1448 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1450 = dfcir.shr(%1449 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1451 = dfcir.sub[?] (%1412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1450 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1452 = dfcir.shr(%1451 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1453 = dfcir.cast[?] (%1452 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1454 = dfcir.greater[?] (%1453 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1455 = dfcir.cast[?] (%1454 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1456 = dfcir.greaterEq[?] (%1453 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1457 = dfcir.cast[?] (%1456 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1458 = dfcir.add[?] (%1457 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1455 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1459 = dfcir.mux(%1458: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1453, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1460 = dfcir.cast[?] (%1459 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1461 = dfcir.add[?] (%1412 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1450 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1462 = dfcir.shr(%1461 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1463 = dfcir.cast[?] (%1462 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1464 = dfcir.greater[?] (%1463 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1465 = dfcir.cast[?] (%1464 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1466 = dfcir.greaterEq[?] (%1463 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1467 = dfcir.cast[?] (%1466 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1468 = dfcir.add[?] (%1467 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1465 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1469 = dfcir.mux(%1468: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1463, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1470 = dfcir.cast[?] (%1469 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1471 = dfcir.add[?] (%1446 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1422 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1472 = dfcir.mul[?] (%1471 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1473 = dfcir.add[?] (%1472 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1474 = dfcir.shr(%1473 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1475 = dfcir.sub[?] (%1413 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1474 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1476 = dfcir.shr(%1475 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1477 = dfcir.cast[?] (%1476 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1478 = dfcir.greater[?] (%1477 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1479 = dfcir.cast[?] (%1478 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1480 = dfcir.greaterEq[?] (%1477 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1481 = dfcir.cast[?] (%1480 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1482 = dfcir.add[?] (%1481 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1479 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1483 = dfcir.mux(%1482: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1477, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1484 = dfcir.cast[?] (%1483 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1485 = dfcir.add[?] (%1413 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1474 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1486 = dfcir.shr(%1485 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1487 = dfcir.cast[?] (%1486 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1488 = dfcir.greater[?] (%1487 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1489 = dfcir.cast[?] (%1488 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1490 = dfcir.greaterEq[?] (%1487 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1491 = dfcir.cast[?] (%1490 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1492 = dfcir.add[?] (%1491 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1489 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1493 = dfcir.mux(%1492: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1487, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1494 = dfcir.cast[?] (%1493 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1495 = dfcir.add[?] (%1445 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1418 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1496 = dfcir.sub[?] (%1409 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1495 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1497 = dfcir.shr(%1496 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1498 = dfcir.cast[?] (%1497 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1499 = dfcir.greater[?] (%1498 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1500 = dfcir.cast[?] (%1499 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1501 = dfcir.greaterEq[?] (%1498 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1502 = dfcir.cast[?] (%1501 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1503 = dfcir.add[?] (%1502 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1500 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1504 = dfcir.mux(%1503: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1498, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1505 = dfcir.cast[?] (%1504 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1506 = dfcir.add[?] (%1409 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1495 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1507 = dfcir.shr(%1506 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1508 = dfcir.cast[?] (%1507 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1509 = dfcir.greater[?] (%1508 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1510 = dfcir.cast[?] (%1509 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1511 = dfcir.greaterEq[?] (%1508 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1512 = dfcir.cast[?] (%1511 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1513 = dfcir.add[?] (%1512 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1510 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1514 = dfcir.mux(%1513: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1508, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1515 = dfcir.cast[?] (%1514 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1516 = dfcir.add[?] (%586 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1517 = dfcir.add[?] (%1516 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %584 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1518 = dfcir.shr(%1517 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1519 = dfcir.sub[?] (%812 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1520 = dfcir.add[?] (%812 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1518 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1521 = dfcir.sub[?] (%1516 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %182 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1522 = dfcir.shr(%1521 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1523 = dfcir.sub[?] (%811 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1522 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1524 = dfcir.add[?] (%811 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1522 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1525 = dfcir.add[?] (%475 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1526 = dfcir.sub[?] (%1525 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %473 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1527 = dfcir.shr(%1526 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1528 = dfcir.sub[?] (%1525 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %277 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1529 = dfcir.shr(%1528 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1530 = dfcir.add[?] (%697 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1531 = dfcir.sub[?] (%1530 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %87 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1532 = dfcir.shr(%1531 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1533 = dfcir.sub[?] (%1532 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1527 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1534 = dfcir.add[?] (%1532 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1527 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1535 = dfcir.sub[?] (%1519 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1534 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1536 = dfcir.shr(%1535 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1537 = dfcir.cast[?] (%1536 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1538 = dfcir.greater[?] (%1537 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1539 = dfcir.cast[?] (%1538 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1540 = dfcir.greaterEq[?] (%1537 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1541 = dfcir.cast[?] (%1540 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1542 = dfcir.add[?] (%1541 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1539 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1543 = dfcir.mux(%1542: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1537, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1544 = dfcir.cast[?] (%1543 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1545 = dfcir.add[?] (%1519 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1534 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1546 = dfcir.shr(%1545 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1547 = dfcir.cast[?] (%1546 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1548 = dfcir.greater[?] (%1547 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1549 = dfcir.cast[?] (%1548 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1550 = dfcir.greaterEq[?] (%1547 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1551 = dfcir.cast[?] (%1550 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1552 = dfcir.add[?] (%1551 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1549 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1553 = dfcir.mux(%1552: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1547, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1554 = dfcir.cast[?] (%1553 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1555 = dfcir.add[?] (%1530 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %695 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1556 = dfcir.shr(%1555 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1557 = dfcir.sub[?] (%1556 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1529 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1558 = dfcir.sub[?] (%1557 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1533 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1559 = dfcir.mul[?] (%1558 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1560 = dfcir.add[?] (%1559 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1561 = dfcir.shr(%1560 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1562 = dfcir.sub[?] (%1523 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1561 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1563 = dfcir.shr(%1562 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1564 = dfcir.cast[?] (%1563 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1565 = dfcir.greater[?] (%1564 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1566 = dfcir.cast[?] (%1565 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1567 = dfcir.greaterEq[?] (%1564 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1568 = dfcir.cast[?] (%1567 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1569 = dfcir.add[?] (%1568 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1566 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1570 = dfcir.mux(%1569: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1564, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1571 = dfcir.cast[?] (%1570 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1572 = dfcir.add[?] (%1523 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1561 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1573 = dfcir.shr(%1572 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1574 = dfcir.cast[?] (%1573 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1575 = dfcir.greater[?] (%1574 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1576 = dfcir.cast[?] (%1575 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1577 = dfcir.greaterEq[?] (%1574 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1578 = dfcir.cast[?] (%1577 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1579 = dfcir.add[?] (%1578 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1576 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1580 = dfcir.mux(%1579: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1574, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1581 = dfcir.cast[?] (%1580 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1582 = dfcir.add[?] (%1557 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1533 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1583 = dfcir.mul[?] (%1582 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1584 = dfcir.add[?] (%1583 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1585 = dfcir.shr(%1584 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1586 = dfcir.sub[?] (%1524 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1585 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1587 = dfcir.shr(%1586 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1588 = dfcir.cast[?] (%1587 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1589 = dfcir.greater[?] (%1588 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1590 = dfcir.cast[?] (%1589 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1591 = dfcir.greaterEq[?] (%1588 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1592 = dfcir.cast[?] (%1591 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1593 = dfcir.add[?] (%1592 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1590 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1594 = dfcir.mux(%1593: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1588, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1595 = dfcir.cast[?] (%1594 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1596 = dfcir.add[?] (%1524 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1585 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1597 = dfcir.shr(%1596 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1598 = dfcir.cast[?] (%1597 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1599 = dfcir.greater[?] (%1598 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1600 = dfcir.cast[?] (%1599 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1601 = dfcir.greaterEq[?] (%1598 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1602 = dfcir.cast[?] (%1601 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1603 = dfcir.add[?] (%1602 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1600 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1604 = dfcir.mux(%1603: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1598, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1605 = dfcir.cast[?] (%1604 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1606 = dfcir.add[?] (%1556 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1529 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1607 = dfcir.sub[?] (%1520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1606 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1608 = dfcir.shr(%1607 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1609 = dfcir.cast[?] (%1608 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1610 = dfcir.greater[?] (%1609 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1611 = dfcir.cast[?] (%1610 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1612 = dfcir.greaterEq[?] (%1609 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1613 = dfcir.cast[?] (%1612 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1614 = dfcir.add[?] (%1613 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1611 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1615 = dfcir.mux(%1614: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1609, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1616 = dfcir.cast[?] (%1615 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1617 = dfcir.add[?] (%1520 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1606 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1618 = dfcir.shr(%1617 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1619 = dfcir.cast[?] (%1618 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1620 = dfcir.greater[?] (%1619 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1621 = dfcir.cast[?] (%1620 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1622 = dfcir.greaterEq[?] (%1619 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1623 = dfcir.cast[?] (%1622 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1624 = dfcir.add[?] (%1623 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1621 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1625 = dfcir.mux(%1624: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1619, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1626 = dfcir.cast[?] (%1625 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1627 = dfcir.add[?] (%617 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1628 = dfcir.add[?] (%1627 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %615 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1629 = dfcir.shr(%1628 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1630 = dfcir.sub[?] (%847 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1629 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1631 = dfcir.add[?] (%847 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1629 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1632 = dfcir.sub[?] (%1627 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %205 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1633 = dfcir.shr(%1632 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1634 = dfcir.sub[?] (%846 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1633 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1635 = dfcir.add[?] (%846 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1633 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1636 = dfcir.add[?] (%506 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1637 = dfcir.sub[?] (%1636 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %504 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1638 = dfcir.shr(%1637 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1639 = dfcir.sub[?] (%1636 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %300 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1640 = dfcir.shr(%1639 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1641 = dfcir.add[?] (%728 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %849 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1642 = dfcir.sub[?] (%1641 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %110 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1643 = dfcir.shr(%1642 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1644 = dfcir.sub[?] (%1643 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1638 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1645 = dfcir.add[?] (%1643 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1638 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1646 = dfcir.sub[?] (%1630 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1645 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1647 = dfcir.shr(%1646 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1648 = dfcir.cast[?] (%1647 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1649 = dfcir.greater[?] (%1648 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1650 = dfcir.cast[?] (%1649 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1651 = dfcir.greaterEq[?] (%1648 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1652 = dfcir.cast[?] (%1651 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1653 = dfcir.add[?] (%1652 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1650 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1654 = dfcir.mux(%1653: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1648, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1655 = dfcir.cast[?] (%1654 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1656 = dfcir.add[?] (%1630 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1645 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1657 = dfcir.shr(%1656 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1658 = dfcir.cast[?] (%1657 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1659 = dfcir.greater[?] (%1658 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1660 = dfcir.cast[?] (%1659 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1661 = dfcir.greaterEq[?] (%1658 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1662 = dfcir.cast[?] (%1661 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1663 = dfcir.add[?] (%1662 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1660 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1664 = dfcir.mux(%1663: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1658, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1665 = dfcir.cast[?] (%1664 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1666 = dfcir.add[?] (%1641 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %726 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1667 = dfcir.shr(%1666 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 3 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1668 = dfcir.sub[?] (%1667 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1640 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1669 = dfcir.sub[?] (%1668 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1644 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1670 = dfcir.mul[?] (%1669 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1671 = dfcir.add[?] (%1670 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1672 = dfcir.shr(%1671 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1673 = dfcir.sub[?] (%1634 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1672 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1674 = dfcir.shr(%1673 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1675 = dfcir.cast[?] (%1674 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1676 = dfcir.greater[?] (%1675 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1677 = dfcir.cast[?] (%1676 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1678 = dfcir.greaterEq[?] (%1675 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1679 = dfcir.cast[?] (%1678 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1680 = dfcir.add[?] (%1679 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1677 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1681 = dfcir.mux(%1680: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1675, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1682 = dfcir.cast[?] (%1681 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1683 = dfcir.add[?] (%1634 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1672 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1684 = dfcir.shr(%1683 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1685 = dfcir.cast[?] (%1684 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1686 = dfcir.greater[?] (%1685 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1687 = dfcir.cast[?] (%1686 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1688 = dfcir.greaterEq[?] (%1685 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1689 = dfcir.cast[?] (%1688 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1690 = dfcir.add[?] (%1689 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1687 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1691 = dfcir.mux(%1690: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1685, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1692 = dfcir.cast[?] (%1691 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1693 = dfcir.add[?] (%1668 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1644 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1694 = dfcir.mul[?] (%1693 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %4 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1695 = dfcir.add[?] (%1694 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %12 : !dfcir.const<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1696 = dfcir.shr(%1695 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 8 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1697 = dfcir.sub[?] (%1635 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1696 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1698 = dfcir.shr(%1697 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1699 = dfcir.cast[?] (%1698 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1700 = dfcir.greater[?] (%1699 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1701 = dfcir.cast[?] (%1700 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1702 = dfcir.greaterEq[?] (%1699 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1703 = dfcir.cast[?] (%1702 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1704 = dfcir.add[?] (%1703 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1701 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1705 = dfcir.mux(%1704: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1699, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1706 = dfcir.cast[?] (%1705 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1707 = dfcir.add[?] (%1635 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1696 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1708 = dfcir.shr(%1707 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1709 = dfcir.cast[?] (%1708 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1710 = dfcir.greater[?] (%1709 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1711 = dfcir.cast[?] (%1710 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1712 = dfcir.greaterEq[?] (%1709 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1713 = dfcir.cast[?] (%1712 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1714 = dfcir.add[?] (%1713 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1711 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1715 = dfcir.mux(%1714: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1709, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1716 = dfcir.cast[?] (%1715 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1717 = dfcir.add[?] (%1667 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1640 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1718 = dfcir.sub[?] (%1631 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1717 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1719 = dfcir.shr(%1718 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1720 = dfcir.cast[?] (%1719 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1721 = dfcir.greater[?] (%1720 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1722 = dfcir.cast[?] (%1721 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1723 = dfcir.greaterEq[?] (%1720 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1724 = dfcir.cast[?] (%1723 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1725 = dfcir.add[?] (%1724 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1722 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1726 = dfcir.mux(%1725: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1720, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1727 = dfcir.cast[?] (%1726 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1728 = dfcir.add[?] (%1631 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, %1717 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 31, 0>> {latency = -1 : i32}
    %1729 = dfcir.shr(%1728 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>, 14 : i32) : !dfcir.stream<!dfcir.fixed<true, 31, 0>>
    %1730 = dfcir.cast[?] (%1729 : !dfcir.stream<!dfcir.fixed<true, 31, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>> {latency = -1 : i32}
    %1731 = dfcir.greater[?] (%1730 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %8 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1732 = dfcir.cast[?] (%1731 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1733 = dfcir.greaterEq[?] (%1730 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>, %7 : !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<false, 1, 0>> {latency = -1 : i32}
    %1734 = dfcir.cast[?] (%1733 : !dfcir.stream<!dfcir.fixed<false, 1, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1735 = dfcir.add[?] (%1734 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %1732 : !dfcir.stream<!dfcir.fixed<false, 2, 0>>) : !dfcir.stream<!dfcir.fixed<false, 2, 0>> {latency = -1 : i32}
    %1736 = dfcir.mux(%1735: !dfcir.stream<!dfcir.fixed<false, 2, 0>>, %8: !dfcir.const<!dfcir.fixed<true, 12, 0>>, %1730, %7: !dfcir.const<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.fixed<true, 12, 0>>
    %1737 = dfcir.cast[?] (%1736 : !dfcir.stream<!dfcir.fixed<true, 12, 0>>) : !dfcir.stream<!dfcir.rawbits<9>> {latency = -1 : i32}
    %1738 = dfcir.cat(%1626 : !dfcir.stream<!dfcir.rawbits<9>>, %1737 : !dfcir.stream<!dfcir.rawbits<9>>) : !dfcir.stream<!dfcir.rawbits<18>>
    %1739 = dfcir.cat(%1515 : !dfcir.stream<!dfcir.rawbits<9>>, %1738 : !dfcir.stream<!dfcir.rawbits<18>>) : !dfcir.stream<!dfcir.rawbits<27>>
    %1740 = dfcir.cat(%1404 : !dfcir.stream<!dfcir.rawbits<9>>, %1739 : !dfcir.stream<!dfcir.rawbits<27>>) : !dfcir.stream<!dfcir.rawbits<36>>
    %1741 = dfcir.cat(%1293 : !dfcir.stream<!dfcir.rawbits<9>>, %1740 : !dfcir.stream<!dfcir.rawbits<36>>) : !dfcir.stream<!dfcir.rawbits<45>>
    %1742 = dfcir.cat(%1182 : !dfcir.stream<!dfcir.rawbits<9>>, %1741 : !dfcir.stream<!dfcir.rawbits<45>>) : !dfcir.stream<!dfcir.rawbits<54>>
    %1743 = dfcir.cat(%1071 : !dfcir.stream<!dfcir.rawbits<9>>, %1742 : !dfcir.stream<!dfcir.rawbits<54>>) : !dfcir.stream<!dfcir.rawbits<63>>
    %1744 = dfcir.cat(%960 : !dfcir.stream<!dfcir.rawbits<9>>, %1743 : !dfcir.stream<!dfcir.rawbits<63>>) : !dfcir.stream<!dfcir.rawbits<72>>
    %1745 = dfcir.cat(%1716 : !dfcir.stream<!dfcir.rawbits<9>>, %1744 : !dfcir.stream<!dfcir.rawbits<72>>) : !dfcir.stream<!dfcir.rawbits<81>>
    %1746 = dfcir.cat(%1605 : !dfcir.stream<!dfcir.rawbits<9>>, %1745 : !dfcir.stream<!dfcir.rawbits<81>>) : !dfcir.stream<!dfcir.rawbits<90>>
    %1747 = dfcir.cat(%1494 : !dfcir.stream<!dfcir.rawbits<9>>, %1746 : !dfcir.stream<!dfcir.rawbits<90>>) : !dfcir.stream<!dfcir.rawbits<99>>
    %1748 = dfcir.cat(%1383 : !dfcir.stream<!dfcir.rawbits<9>>, %1747 : !dfcir.stream<!dfcir.rawbits<99>>) : !dfcir.stream<!dfcir.rawbits<108>>
    %1749 = dfcir.cat(%1272 : !dfcir.stream<!dfcir.rawbits<9>>, %1748 : !dfcir.stream<!dfcir.rawbits<108>>) : !dfcir.stream<!dfcir.rawbits<117>>
    %1750 = dfcir.cat(%1161 : !dfcir.stream<!dfcir.rawbits<9>>, %1749 : !dfcir.stream<!dfcir.rawbits<117>>) : !dfcir.stream<!dfcir.rawbits<126>>
    %1751 = dfcir.cat(%1050 : !dfcir.stream<!dfcir.rawbits<9>>, %1750 : !dfcir.stream<!dfcir.rawbits<126>>) : !dfcir.stream<!dfcir.rawbits<135>>
    %1752 = dfcir.cat(%939 : !dfcir.stream<!dfcir.rawbits<9>>, %1751 : !dfcir.stream<!dfcir.rawbits<135>>) : !dfcir.stream<!dfcir.rawbits<144>>
    %1753 = dfcir.cat(%1692 : !dfcir.stream<!dfcir.rawbits<9>>, %1752 : !dfcir.stream<!dfcir.rawbits<144>>) : !dfcir.stream<!dfcir.rawbits<153>>
    %1754 = dfcir.cat(%1581 : !dfcir.stream<!dfcir.rawbits<9>>, %1753 : !dfcir.stream<!dfcir.rawbits<153>>) : !dfcir.stream<!dfcir.rawbits<162>>
    %1755 = dfcir.cat(%1470 : !dfcir.stream<!dfcir.rawbits<9>>, %1754 : !dfcir.stream<!dfcir.rawbits<162>>) : !dfcir.stream<!dfcir.rawbits<171>>
    %1756 = dfcir.cat(%1359 : !dfcir.stream<!dfcir.rawbits<9>>, %1755 : !dfcir.stream<!dfcir.rawbits<171>>) : !dfcir.stream<!dfcir.rawbits<180>>
    %1757 = dfcir.cat(%1248 : !dfcir.stream<!dfcir.rawbits<9>>, %1756 : !dfcir.stream<!dfcir.rawbits<180>>) : !dfcir.stream<!dfcir.rawbits<189>>
    %1758 = dfcir.cat(%1137 : !dfcir.stream<!dfcir.rawbits<9>>, %1757 : !dfcir.stream<!dfcir.rawbits<189>>) : !dfcir.stream<!dfcir.rawbits<198>>
    %1759 = dfcir.cat(%1026 : !dfcir.stream<!dfcir.rawbits<9>>, %1758 : !dfcir.stream<!dfcir.rawbits<198>>) : !dfcir.stream<!dfcir.rawbits<207>>
    %1760 = dfcir.cat(%915 : !dfcir.stream<!dfcir.rawbits<9>>, %1759 : !dfcir.stream<!dfcir.rawbits<207>>) : !dfcir.stream<!dfcir.rawbits<216>>
    %1761 = dfcir.cat(%1665 : !dfcir.stream<!dfcir.rawbits<9>>, %1760 : !dfcir.stream<!dfcir.rawbits<216>>) : !dfcir.stream<!dfcir.rawbits<225>>
    %1762 = dfcir.cat(%1554 : !dfcir.stream<!dfcir.rawbits<9>>, %1761 : !dfcir.stream<!dfcir.rawbits<225>>) : !dfcir.stream<!dfcir.rawbits<234>>
    %1763 = dfcir.cat(%1443 : !dfcir.stream<!dfcir.rawbits<9>>, %1762 : !dfcir.stream<!dfcir.rawbits<234>>) : !dfcir.stream<!dfcir.rawbits<243>>
    %1764 = dfcir.cat(%1332 : !dfcir.stream<!dfcir.rawbits<9>>, %1763 : !dfcir.stream<!dfcir.rawbits<243>>) : !dfcir.stream<!dfcir.rawbits<252>>
    %1765 = dfcir.cat(%1221 : !dfcir.stream<!dfcir.rawbits<9>>, %1764 : !dfcir.stream<!dfcir.rawbits<252>>) : !dfcir.stream<!dfcir.rawbits<261>>
    %1766 = dfcir.cat(%1110 : !dfcir.stream<!dfcir.rawbits<9>>, %1765 : !dfcir.stream<!dfcir.rawbits<261>>) : !dfcir.stream<!dfcir.rawbits<270>>
    %1767 = dfcir.cat(%999 : !dfcir.stream<!dfcir.rawbits<9>>, %1766 : !dfcir.stream<!dfcir.rawbits<270>>) : !dfcir.stream<!dfcir.rawbits<279>>
    %1768 = dfcir.cat(%888 : !dfcir.stream<!dfcir.rawbits<9>>, %1767 : !dfcir.stream<!dfcir.rawbits<279>>) : !dfcir.stream<!dfcir.rawbits<288>>
    %1769 = dfcir.cat(%1655 : !dfcir.stream<!dfcir.rawbits<9>>, %1768 : !dfcir.stream<!dfcir.rawbits<288>>) : !dfcir.stream<!dfcir.rawbits<297>>
    %1770 = dfcir.cat(%1544 : !dfcir.stream<!dfcir.rawbits<9>>, %1769 : !dfcir.stream<!dfcir.rawbits<297>>) : !dfcir.stream<!dfcir.rawbits<306>>
    %1771 = dfcir.cat(%1433 : !dfcir.stream<!dfcir.rawbits<9>>, %1770 : !dfcir.stream<!dfcir.rawbits<306>>) : !dfcir.stream<!dfcir.rawbits<315>>
    %1772 = dfcir.cat(%1322 : !dfcir.stream<!dfcir.rawbits<9>>, %1771 : !dfcir.stream<!dfcir.rawbits<315>>) : !dfcir.stream<!dfcir.rawbits<324>>
    %1773 = dfcir.cat(%1211 : !dfcir.stream<!dfcir.rawbits<9>>, %1772 : !dfcir.stream<!dfcir.rawbits<324>>) : !dfcir.stream<!dfcir.rawbits<333>>
    %1774 = dfcir.cat(%1100 : !dfcir.stream<!dfcir.rawbits<9>>, %1773 : !dfcir.stream<!dfcir.rawbits<333>>) : !dfcir.stream<!dfcir.rawbits<342>>
    %1775 = dfcir.cat(%989 : !dfcir.stream<!dfcir.rawbits<9>>, %1774 : !dfcir.stream<!dfcir.rawbits<342>>) : !dfcir.stream<!dfcir.rawbits<351>>
    %1776 = dfcir.cat(%878 : !dfcir.stream<!dfcir.rawbits<9>>, %1775 : !dfcir.stream<!dfcir.rawbits<351>>) : !dfcir.stream<!dfcir.rawbits<360>>
    %1777 = dfcir.cat(%1682 : !dfcir.stream<!dfcir.rawbits<9>>, %1776 : !dfcir.stream<!dfcir.rawbits<360>>) : !dfcir.stream<!dfcir.rawbits<369>>
    %1778 = dfcir.cat(%1571 : !dfcir.stream<!dfcir.rawbits<9>>, %1777 : !dfcir.stream<!dfcir.rawbits<369>>) : !dfcir.stream<!dfcir.rawbits<378>>
    %1779 = dfcir.cat(%1460 : !dfcir.stream<!dfcir.rawbits<9>>, %1778 : !dfcir.stream<!dfcir.rawbits<378>>) : !dfcir.stream<!dfcir.rawbits<387>>
    %1780 = dfcir.cat(%1349 : !dfcir.stream<!dfcir.rawbits<9>>, %1779 : !dfcir.stream<!dfcir.rawbits<387>>) : !dfcir.stream<!dfcir.rawbits<396>>
    %1781 = dfcir.cat(%1238 : !dfcir.stream<!dfcir.rawbits<9>>, %1780 : !dfcir.stream<!dfcir.rawbits<396>>) : !dfcir.stream<!dfcir.rawbits<405>>
    %1782 = dfcir.cat(%1127 : !dfcir.stream<!dfcir.rawbits<9>>, %1781 : !dfcir.stream<!dfcir.rawbits<405>>) : !dfcir.stream<!dfcir.rawbits<414>>
    %1783 = dfcir.cat(%1016 : !dfcir.stream<!dfcir.rawbits<9>>, %1782 : !dfcir.stream<!dfcir.rawbits<414>>) : !dfcir.stream<!dfcir.rawbits<423>>
    %1784 = dfcir.cat(%905 : !dfcir.stream<!dfcir.rawbits<9>>, %1783 : !dfcir.stream<!dfcir.rawbits<423>>) : !dfcir.stream<!dfcir.rawbits<432>>
    %1785 = dfcir.cat(%1706 : !dfcir.stream<!dfcir.rawbits<9>>, %1784 : !dfcir.stream<!dfcir.rawbits<432>>) : !dfcir.stream<!dfcir.rawbits<441>>
    %1786 = dfcir.cat(%1595 : !dfcir.stream<!dfcir.rawbits<9>>, %1785 : !dfcir.stream<!dfcir.rawbits<441>>) : !dfcir.stream<!dfcir.rawbits<450>>
    %1787 = dfcir.cat(%1484 : !dfcir.stream<!dfcir.rawbits<9>>, %1786 : !dfcir.stream<!dfcir.rawbits<450>>) : !dfcir.stream<!dfcir.rawbits<459>>
    %1788 = dfcir.cat(%1373 : !dfcir.stream<!dfcir.rawbits<9>>, %1787 : !dfcir.stream<!dfcir.rawbits<459>>) : !dfcir.stream<!dfcir.rawbits<468>>
    %1789 = dfcir.cat(%1262 : !dfcir.stream<!dfcir.rawbits<9>>, %1788 : !dfcir.stream<!dfcir.rawbits<468>>) : !dfcir.stream<!dfcir.rawbits<477>>
    %1790 = dfcir.cat(%1151 : !dfcir.stream<!dfcir.rawbits<9>>, %1789 : !dfcir.stream<!dfcir.rawbits<477>>) : !dfcir.stream<!dfcir.rawbits<486>>
    %1791 = dfcir.cat(%1040 : !dfcir.stream<!dfcir.rawbits<9>>, %1790 : !dfcir.stream<!dfcir.rawbits<486>>) : !dfcir.stream<!dfcir.rawbits<495>>
    %1792 = dfcir.cat(%929 : !dfcir.stream<!dfcir.rawbits<9>>, %1791 : !dfcir.stream<!dfcir.rawbits<495>>) : !dfcir.stream<!dfcir.rawbits<504>>
    %1793 = dfcir.cat(%1727 : !dfcir.stream<!dfcir.rawbits<9>>, %1792 : !dfcir.stream<!dfcir.rawbits<504>>) : !dfcir.stream<!dfcir.rawbits<513>>
    %1794 = dfcir.cat(%1616 : !dfcir.stream<!dfcir.rawbits<9>>, %1793 : !dfcir.stream<!dfcir.rawbits<513>>) : !dfcir.stream<!dfcir.rawbits<522>>
    %1795 = dfcir.cat(%1505 : !dfcir.stream<!dfcir.rawbits<9>>, %1794 : !dfcir.stream<!dfcir.rawbits<522>>) : !dfcir.stream<!dfcir.rawbits<531>>
    %1796 = dfcir.cat(%1394 : !dfcir.stream<!dfcir.rawbits<9>>, %1795 : !dfcir.stream<!dfcir.rawbits<531>>) : !dfcir.stream<!dfcir.rawbits<540>>
    %1797 = dfcir.cat(%1283 : !dfcir.stream<!dfcir.rawbits<9>>, %1796 : !dfcir.stream<!dfcir.rawbits<540>>) : !dfcir.stream<!dfcir.rawbits<549>>
    %1798 = dfcir.cat(%1172 : !dfcir.stream<!dfcir.rawbits<9>>, %1797 : !dfcir.stream<!dfcir.rawbits<549>>) : !dfcir.stream<!dfcir.rawbits<558>>
    %1799 = dfcir.cat(%1061 : !dfcir.stream<!dfcir.rawbits<9>>, %1798 : !dfcir.stream<!dfcir.rawbits<558>>) : !dfcir.stream<!dfcir.rawbits<567>>
    %1800 = dfcir.cat(%950 : !dfcir.stream<!dfcir.rawbits<9>>, %1799 : !dfcir.stream<!dfcir.rawbits<567>>) : !dfcir.stream<!dfcir.rawbits<576>>
    %1801 = dfcir.cast[?] (%1800 : !dfcir.stream<!dfcir.rawbits<576>>) : !dfcir.stream<!dfcir.fixed<true, 575, 0>> {latency = -1 : i32}
    %1802 = dfcir.output<!dfcir.fixed<true, 575, 0>> ("out") {operandSegmentSizes = array<i32: 0, 0>}
    dfcir.connect(%1802 : !dfcir.stream<!dfcir.fixed<true, 575, 0>>, %1801 : !dfcir.stream<!dfcir.fixed<true, 575, 0>>)
    %1803 = dfcir.constant<!dfcir.fixed<true, 31, 0>> 1609 : si32
  }
}
