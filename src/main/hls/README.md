# High-Level Representation

High-level modeling and synthesis facilities:

* `parser` parses a high-level description of the accelerator and builds the IR;
* `model` implements the high-level IR, including the basic transformations;
* `scheduler` performs spatial/temporal optimization of the computation graph;
* `library` implements the library infrastructure and some functional/communication units;
* `mapper` binds the operations w/ library elements and maps the computation graph to the FPGA;
* `builder` constructs the RTL IR and generates the HDL description;
* `runtime` provides drivers and an API for interacting w/ the constructed accelerator.
