// RUN: hil-opt %s | hil-opt | FileCheck %s
// CHECK: hil.model
// CHECK: hil.nodetype "{{.*}}" [{{.*}}] => [{{.*}}]
// CHECK: hil.graph "main"
// CHECK: hil.chans
// CHECK: hil.chan "{{.*}}" "{{.*}}"
// CHECK: hil.nodes
// CHECK: hil.node "{{.*}}" "{{.*}}" [{{.*}}] => [{{.*}}]
hil.model {
  hil.nodetype "source" [] => [
    !hil.output<"X"<1.0> 0 "x">,
    !hil.output<"Y"<1.0> 0 "y">
  ]
  hil.nodetype "split" [
    !hil.input<"X"<1.0> "x">
  ] => [
    !hil.output<"X"<0.5> 1 "x1">,
    !hil.output<"X"<0.5> 1 "x2">
  ]
  hil.nodetype "kernel1" [
    !hil.input<"X"<1.0> "x">,
    !hil.input<"Y"<0.5> "y">
  ] => [
    !hil.output<"Z"<0.25> 1 "z">,
    !hil.output<"W"<1.0> 2 "w">
  ]
  hil.nodetype "kernel2" [
    !hil.input<"X"<0.5> "x">,
    !hil.input<"W"<0.5> "w">
  ] => [
    !hil.output<"Z"<0.25> 1 "z">
  ]
  hil.nodetype "merge" [
    !hil.input<"Z"<0.5> "z1">,
    !hil.input<"Z"<0.5> "z2">
  ] => [
    !hil.output<"Z"<1.0> 1 "z">
  ]
  hil.nodetype "sink" [
    !hil.input<"Z"<1.0> "z">
  ] => []
}
 
hil.graph "main" {
  hil.chans {
    hil.chan "X" "x1"
    hil.chan "X" "x2"
    hil.chan "X" "x"
    hil.chan "Y" "y"
    hil.chan "Z" "z1"
    hil.chan "Z" "z2"
    hil.chan "Z" "z"
    hil.chan "W" "w"
  }
  hil.nodes {
    hil.node "source"  "n1" []           => ["x", "y"]
    hil.node "split"   "n2" ["x"]        => ["x1", "x2"]
    hil.node "kernel1" "n3" ["x1", "y"]  => ["z1", "w"]
    hil.node "kernel2" "n4" ["x2", "w"]  => ["z2"]
    hil.node "merge"   "n5" ["z1", "z2"] => ["z"]
    hil.node "sink"    "n6" ["z"]        => []
  }
}
