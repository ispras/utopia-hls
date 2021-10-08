<!-- SPDX-License-Identifier: Apache-2.0 ->

# High-Level Representation

High-level modeling and synthesis facilities:

* `parser` parses a high-level description of the accelerator and builds the IR;
* `model` implements the high-level IR w/ basic transformations;
* `scheduler` performs spatial/temporal optimization of the computation graph (IR);
* `library` contains the library infrastructure and basic libraries;
* `mapper` binds the operations w/ library elements and maps the computation graph to HW;
* `compiler` constructs the RTL IR and generates the HDL description;
* `runtime` provides the drivers and the API for interacting w/ the accelerator;
* `debugger` includes simulation, verification, and debugging tools.
