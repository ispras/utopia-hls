[//]: <> (SPDX-License-Identifier: Apache-2.0)

# Utopia HLS

Utopia is an open-source high-level synthesis tool for streaming dataflow computers.

With the use of its domain-specific language, DFCxx, a streaming computation dataflow can be specified. This specification is used to compile a generator for corresponding SystemVerilog modules.

DFCxx provides a library of C++ classes (like `Stream`, `Scalar`, `Kernel`), allowing relations between objects to represent the computational logic of a streaming computer.

By inheriting from abstract class `dfcxx::Kernel` and implementing at least a single constructor, this new class represents the main unit of computation, <i>kernel</i>.

The following code describes a kernel for the pipelined computation of a second-order polynomial ***x<sup>2</sup> + 2 * x*** for unsigned 32-bit integer values.

```cpp
#include "dfcxx/DFCXX.h"

class Polynomial2 : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "Polynomial2"; // Explicitly set kernel's name.
  }

  ~Polynomial2() override = default;

  Polynomial2() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType &type = dfUInt(32); // Unsigned 32-bit integer type declaration.
    DFVariable x = io.input("x", type); // Input stream named "x" is declared.
    DFVariable squared = x * x; // "x * x" is computed.
    DFVariable squaredPlusX = squared + x; // "x" is added to "x * x".
    DFVariable result = squaredPlusX + x; // "x" is added to "x * x + x".
    DFVariable out = io.output("out", type); // Output stream named "out" is declared.
    out.connect(result); // "out" stream is connected to the result of computation.
  }
};
```

User-defined source files and include directories for a kernel are used in the compilation of Utopia HLS, so that the resulting binary executable becomes a generator for application-specific SystemVerilog modules.

The computational logic of a streaming dataflow computer has to be statically scheduled. Currently there are two implementations of scheduling algorithms: linear programming and a greedy as-soon-as-possible algorithm.

When the Utopia HLS binary executable is compiled, its integrated command line interface allows a filesystem path to a JSON latency configuration file to be specified, as well as the chosen scheduling strategy.

## Licensing and Distribution

Utopia HLS is distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Coding Style

See `CODE_STYLE.md` for more details.

## Prerequisites

### Packages

It is recommended to use Utopia HLS on Debian-based operating systems (e.g. Ubuntu 20.04). The required dependencies' package names are specific to `apt` package manager, present in Debian-based operating systems, and are specified below:

* `build-essential`
* `clang` + `lld` or `g++` + `gcc` as a C/C++ compiler
* `cmake` ver. 3.20.0 or higher (*)
* `liblpsolve55-dev`
* `ninja-build` (preferred) or `make`
* `libctemplate-dev` as a template generator for SystemVerilog stubs

The following command can be used to install all of these dependencies regardless of what exactly will be used to compile Utopia HLS:

```bash
sudo apt install build-essential clang cmake g++ gcc liblpsolve55-dev lld make ninja-build libctemplate-dev
```

(*)**Note**: in case `cmake` which was installed from `apt install` has a version lower than 3.20.0, follow this [guide](https://apt.kitware.com/) and use `sudo apt install cmake` again.

### Precompiled CIRCT & MLIR Installation

Utopia HLS utilizes SystemVerilog-generation capabilities from CIRCT project, which itself is based on LLVM Multi-Level Intermediate Representation (MLIR) project.

**Currently supported CIRCT release: 1.110.0, Mar 22 2025 ([link](https://github.com/llvm/circt/releases/tag/firtool-1.110.0))**

For both CIRCT and MLIR projects it is possible to download precompiled libraries (this approach is more convenient and saves time), but both projects can also be compiled from sources.

Precompiled releases of CIRCT include the precompiled LLVM MLIR releases as well: visit the corresponding [page](https://github.com/llvm/circt/releases) to see all available releases.

Note that <ins>it's the libraries that are required</ins>, not the binary executables.<br>
Look for the archives which have the following names:
`circt-full-static-<ARCH>` or `circt-full-shared-<ARCH>` for static and dynamic libraries respectively.

### Compiling CIRCT & LLVM from Scratch

MLIR is included in CIRCT in the form of a Git submodule, so the compilation process starts with cloning the CIRCT repository into some directory `CIRCT_DIR`.

```bash
cd <WORKDIR>
git clone --depth 1 --branch firtool-1.110.0 https://github.com/llvm/circt/ <CIRCT_DIR>
cd <CIRCT_DIR>
git submodule init
git submodule update
```

Now that `llvm`-subdirectory has been initialized, we may build MLIR in some directory `<LLVM_BUILD_DIR>`.

The following commands are used to build the simplest MLIR revision, which allows debugging - `RelWithDebInfo` may be changed for `Release` or `Debug` if the user deems it so.

```bash
cd llvm
cmake -S llvm -B <LLVM_BUILD_DIR> -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build <LLVM_BUILD_DIR>
```

This process is going to take a while.
After building MLIR, it can be used to build CIRCT in some directory `<CIRCT_BUILD_DIR>`.

```bash
cd <CIRCT_DIR>
cmake -S . -B <CIRCT_BUILD_DIR> -DMLIR_DIR=<LLVM_BUILD_DIR>/lib/cmake/mlir -DLLVM_DIR=<LLVM_BUILD_DIR>/lib/cmake/llvm -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build <CIRCT_BUILD_DIR>
```

## Project Compilation

### Building from Source

Utopia HLS is a CMake-based project, so the compilation can be set via as much as two commands.

CMake buildsystem-generation script accepts a number of required and optional arguments (prefixed with `-D`), which are presented below:

* `SRC_FILES`: (quoted) *required* semicolon-separated list of filesystem paths; is used to specify source files for user-defined kernels to be added to the Utopia HLS compilation.
* `CMAKE_PREFIX_PATH`: standard CMake variable; *optional* (quoted) semicolon-separated list of <ins>full</ins> filesystem paths; is used to locate installed CIRCT and MLIR packages. In case a precompiled CIRCT release is used, specifying a path to its unarchived top-level directory is enough (e.g. `-DCMAKE_PREFIX_PATH=~/firtool-1.110.0`). In case CIRCT and MLIR are compiled manually, **paths to each independent build (not source) directory have to be specified** (e.g. `-DCMAKE_PREFIX_PATH-"~/circt/build;~/circt/llvm/build"`). This variable may be omitted if CIRCT and MLIR are installed in default system paths.
* `INCLUDE_DIRS`: (quoted) *optional* semicolon-separated list of filesystem paths; is used to specify include directories for user-defined kernels to be added to the Utopia HLS compilation.
* `OUT`: *optional* string; used to specify a custom name for the final Utopia HLS executable. Unless it is explicitly provided, the final executable will be named `umain`.
* `BUILD_TESTS`: *optional* binary (`ON`/`OFF`) variable; switched to `OFF` by default; is used to enable tests compilation (target `test/utest`). Values other than `ON` are interpreted as `OFF`.

Other standard CMake variables/options may also be specified to affect the final compilation process.

Here is an example of a fully-formed CMake buildsystem-generation script (directory `build` may be changed for any other directory):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=~/firtool-1.110.0 -DINCLUDE_DIRS=~ -DSRC_FILES="~/utopia-user/simple.cpp;~/utopia-user/buf.cpp" -DOUT=executable -DBUILD_TESTS=ON
```

To start the compilation process, a simple `cmake --build`-command is used.

```bash
cmake --build build
```

### User-Defined Kernel Structure

User-defined kernels are comprised of valid C++ headers and source files. One of the provided source files has to define a function with the following signature:

```cpp
std::unique_ptr<dfcxx::Kernel> start();
```

This function will be called from internal Utopia HLS to get the pointer to the top-level DFCxx kernel.

Kernel-object be a valid DFCxx kernel, instantiating a class derived from `dfcxx::Kernel` class.

Here is an example of a valid `start`-function definition (some kernel named `Polynomial2` is used):

```cpp
std::unique_ptr<dfcxx::Kernel> start() {
  Polynomial2 *kernel = new Polynomial2();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
```

## Usage

### CLI

The compiled Utopia HLS executable has a CLI to accept a number of parameters affecting the final output. Utopia HLS CLI is based on a hierarchical structure of subcommands, namely *modes*.

For the executable there are three general arguments: `-h,--help`, `-H,--help-all` and `-v,--version` for printing simple and extended help-messages, and printing current executable's version respectively.

Unless neither of the three arguments is used, first argument is the mode which the executable has to function in. Currently there are only two available modes: `hls` and `sim`.

`hls` mode is used to perform the high-level synthesis of digital hardware description from the input DFCxx kernel. The list of arguments for `hls`-mode is presented below:

* `-h,--help`: *optional* flag; used to print the help-message about other arguments.
* `--config <PATH>`: *optional* filesystem-path option; used to specify the file for a JSON latency configuration file (default if option specified with no `<PATH>`: `latency.json`). Its format is presented in `docs/latency_config.md`.
* `--out-sv <PATH>`: *optional* filesystem-path option; used to specify the output SystemVerilog file (default if option specified with no `<PATH>`: `output.sv`).
* `--out-sv-lib <PATH>`: *optional* filesystem-path option; used to specify the output SystemVerilog file for generated operations library (default if option specified with no `<PATH>`: `output-lib.sv`).
* `--out-dfcir <PATH>`: *optional* filesystem-path option; used to specify the output unscheduled DFCIR file (default if option specified with no `<PATH>`: `dfcir.mlir`).
* `--out-scheduled-dfcir <PATH>`: *optional* filesystem-path option; used to specify the output scheduled DFCIR file (default if option specified with no `<PATH>`: `scheduler-dfcir.mlir`).
* `--out-firrtl <PATH>`: *optional* filesystem-path option; used to specify the output FIRRTL file (default if option specified with no `<PATH>`: `firrtl.mlir`).
* `--out-dot <PATH>`: *optional* filesystem-path option; used to specify the output DOT file (default if option specified with no `<PATH>`: `output.dot`).
* `-a` or `-l` or `--pipeline <STAGES>`: *optional* flag/option; used to specify the chosen scheduling strategy - either as-soon-as-possible or linear programming or combinational pipelining for the provided number of stages (default: `--pipeline 0`). **Exactly one or zero of these flags has to be specified**.

**At least one of the `out-*` options has to be specified.**

Here is an example of an Utopia HLS CLI call:

```bash
umain hls --config ~/utopia-user/config.json --out-sv ~/outFile.sv --out-dfcir ~/outFile2.mlir -a
```

`sim` mode is used to simulate the input DFCxx kernel. The list of arguments for `sim`-mode is presented below:

* `-h,--help`: *optional* flag; used to print the help-message about other arguments.
* `--in <PATH>`: *optional* filesystem-path option; used to specify the input file for simulation data (default: `sim.txt`). Its format is presented in `docs/simulation.md`.
* `--out <PATH>`: *optional* filesystem-path option; used to specify the output VCD file (default: `sim_out.vcd`).

### Examples

Root subdirectory `examples` contains different examples of DFCxx kernels, `start`-function definitions and JSON configuration files.

For example, given subdirectory `polynomial2`, the compilation and execution commands might look the following (setting `INCLUDE_DIRS` is not necessary, as the required headers reside right next to their executable):

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=~/firtool-1.110.0 -DSRC_FILES="~/utopia-hls/examples/polynomial2/polynomial2.cpp"
cmake --build build
build/src/umain hls --config examples/polynomial2/add_int_2_mul_int3.json -a --out-sv output
```

The execution command is going to pass a JSON configuration file (with latencies 2 and 3 for integer addition
and multiplication respectively) to Utopia HLS, resulting in the creation of the file `output`, containing a SystemVerilog
module for `Polynomial2` kernel with a greedy ASAP-scheduling.

The same kernel can be simulated with:

```bash
build/src/umain sim --in examples/polynomial2/sim.txt --out output.vcd
```

This command uses the simulation data from `sim.txt` file and stores the simulation trace in the output file `output.vcd`.

## DFCxx Documentation

WORK IN PROGRESS
