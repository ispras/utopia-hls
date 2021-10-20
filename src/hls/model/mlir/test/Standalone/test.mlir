// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        standalone.model {
            standalone.nodetype "source" [] => [
              !standalone.output<"X"<1.0> 0 "x">,
              !standalone.output<"Y"<1.0> 0 "y">
            ]
            standalone.nodetype "split" [
              !standalone.input<"X"<1.0> "x">
            ] => [
              !standalone.output<"X"<0.5> 1 "x1">,
              !standalone.output<"X"<0.5> 1 "x2">
            ]
            standalone.nodetype "kernel1" [
              !standalone.input<"X"<1.0> "x">,
              !standalone.input<"Y"<0.5> "y">
            ] => [
              !standalone.output<"Z"<0.25> 1 "z">,
              !standalone.output<"W"<1.0> 2 "w">
            ]
            standalone.nodetype "kernel2" [
              !standalone.input<"X"<0.5> "x">,
              !standalone.input<"W"<0.5> "w">
            ] => [
              !standalone.output<"Z"<0.25> 1 "z">
            ]
            standalone.nodetype "merge" [
              !standalone.input<"Z"<0.5> "z1">,
              !standalone.input<"Z"<0.5> "z2">
            ] => [
              !standalone.output<"Z"<1.0> 1 "z">
            ]
            standalone.nodetype "sink" [
              !standalone.input<"Z"<1.0> "z">
            ] => []
        }
        standalone.chan "X" "x1"
        standalone.chan "X" "x2"
        standalone.chan "X" "x"
        standalone.chan "Y" "y"
        standalone.chan "Z" "z1"
        standalone.chan "Z" "z2"
        standalone.chan "Z" "z"
        standalone.chan "W" "w"

        standalone.node "source"  "n1" []           => ["x", "y"]
        standalone.node "split"   "n2" ["x"]        => ["x1", "x2"]
        standalone.node "kernel1" "n3" ["x1", "y"]  => ["z1", "w"]
        standalone.node "kernel2" "n4" ["x2", "w"]  => ["z2"]
        standalone.node "merge"   "n5" ["z1", "z2"] => ["z"]
        standalone.node "sink"    "n6" ["x"]        => []
        return
    }
}
