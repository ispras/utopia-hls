firrtl.circuit "main" {
    firrtl.module @main (
        in %clock : !firrtl.clock,
        in %reset : !firrtl.reset,
        in %n1_x : !firrtl.sint<16>,
        in %n1_y : !firrtl.sint<16>,

        out %n6_z: !firrtl.sint<16>)
        {


        %n2_clock, %n2_reset, %n2_x, %n2_x1, %n2_x2 = firrtl.instance n2 @split(in clock : !firrtl.clock, in reset : !firrtl.reset, in x: !firrtl.sint<16>, out x1: !firrtl.sint<16>, out x2: !firrtl.sint<16>)
        %n3_clock, %n3_reset, %n3_x, %n3_y, %n3_z, %n3_w = firrtl.instance n3 @kernel1(in clock : !firrtl.clock, in reset : !firrtl.reset, in x: !firrtl.sint<16>, in y: !firrtl.sint<16>, out z: !firrtl.sint<16>, out w: !firrtl.sint<16>)
        %n4_clock, %n4_reset, %n4_x, %n4_w, %n4_z = firrtl.instance n4 @kernel2(in clock : !firrtl.clock, in reset : !firrtl.reset, in x: !firrtl.sint<16>, in w: !firrtl.sint<16>, out z: !firrtl.sint<16>)
        %n5_clock, %n5_reset, %n5_z1, %n5_z2, %n5_z = firrtl.instance n5 @merge(in clock : !firrtl.clock, in reset : !firrtl.reset, in z1: !firrtl.sint<16>, in z2: !firrtl.sint<16>, out z: !firrtl.sint<16>)
        %delay_Z_2_6_clock, %delay_Z_2_6_reset, %delay_Z_2_6_in, %delay_Z_2_6_out = firrtl.instance delay_Z_2_6 @delay_Z_2(in clock : !firrtl.clock, in reset : !firrtl.reset, in in: !firrtl.sint<16>, out out: !firrtl.sint<16>)
        %delay_Y_1_7_clock, %delay_Y_1_7_reset, %delay_Y_1_7_in, %delay_Y_1_7_out = firrtl.instance delay_Y_1_7 @delay_Y_1(in clock : !firrtl.clock, in reset : !firrtl.reset, in in: !firrtl.sint<16>, out out: !firrtl.sint<16>)
        %delay_X_2_8_clock, %delay_X_2_8_reset, %delay_X_2_8_in, %delay_X_2_8_out = firrtl.instance delay_X_2_8 @delay_X_2(in clock : !firrtl.clock, in reset : !firrtl.reset, in in: !firrtl.sint<16>, out out: !firrtl.sint<16>)
        firrtl.connect %n2_clock, %clock : !firrtl.clock, !firrtl.clock
        firrtl.connect %n2_reset, %reset : !firrtl.reset, !firrtl.reset
        firrtl.connect %n2_x, %n1_x : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %n3_x, %n2_x1 : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %delay_X_2_8_in, %n2_x2 : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %n3_clock, %clock : !firrtl.clock, !firrtl.clock
        firrtl.connect %n3_reset, %reset : !firrtl.reset, !firrtl.reset
        firrtl.connect %delay_Z_2_6_in, %n3_z : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %n4_w, %n3_w : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %n4_clock, %clock : !firrtl.clock, !firrtl.clock
        firrtl.connect %n4_reset, %reset : !firrtl.reset, !firrtl.reset
        firrtl.connect %n5_z2, %n4_z : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %n5_clock, %clock : !firrtl.clock, !firrtl.clock
        firrtl.connect %n5_reset, %reset : !firrtl.reset, !firrtl.reset
        firrtl.connect %delay_Z_2_6_clock, %clock : !firrtl.clock, !firrtl.clock
        firrtl.connect %delay_Z_2_6_reset, %reset : !firrtl.reset, !firrtl.reset
        firrtl.connect %n5_z1, %delay_Z_2_6_out : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %delay_Y_1_7_clock, %clock : !firrtl.clock, !firrtl.clock
        firrtl.connect %delay_Y_1_7_reset, %reset : !firrtl.reset, !firrtl.reset
        firrtl.connect %delay_Y_1_7_in, %n1_y : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %n3_y, %delay_Y_1_7_out : !firrtl.sint<16>, !firrtl.sint<16>
        firrtl.connect %delay_X_2_8_clock, %clock : !firrtl.clock, !firrtl.clock
        firrtl.connect %delay_X_2_8_reset, %reset : !firrtl.reset, !firrtl.reset
        firrtl.connect %n4_x, %delay_X_2_8_out : !firrtl.sint<16>, !firrtl.sint<16>
        } 

    firrtl.extmodule @delay_X_2(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in in : !firrtl.sint<16>,
        out out: !firrtl.sint<16>)
    firrtl.extmodule @delay_Y_1(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in in : !firrtl.sint<16>,
        out out: !firrtl.sint<16>)
    firrtl.extmodule @delay_Z_2(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in in : !firrtl.sint<16>,
        out out: !firrtl.sint<16>)
    firrtl.extmodule @kernel1(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in x : !firrtl.sint<16>,
        in y : !firrtl.sint<16>,
        out z: !firrtl.sint<16>,
        out w: !firrtl.sint<16>)
    firrtl.extmodule @kernel2(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in x : !firrtl.sint<16>,
        in w : !firrtl.sint<16>,
        out z: !firrtl.sint<16>)
    firrtl.extmodule @merge(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in z1 : !firrtl.sint<16>,
        in z2 : !firrtl.sint<16>,
        out z: !firrtl.sint<16>)
    firrtl.extmodule @sink(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in z : !firrtl.sint<16>)
    firrtl.extmodule @source(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        out x: !firrtl.sint<16>,
        out y: !firrtl.sint<16>)
    firrtl.extmodule @split(
        in clock : !firrtl.clock,
        in reset : !firrtl.reset,
        in x : !firrtl.sint<16>,
        out x1: !firrtl.sint<16>,
        out x2: !firrtl.sint<16>)
}