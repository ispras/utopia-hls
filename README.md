[//]: <> (SPDX-License-Identifier: Apache-2.0)

# Utopia HLS

Utopia is an open-source high-level synthesis tool for streaming dataflow computers.

With the use of its domain-specific language, DFCxx, a streaming computation dataflow can be specified. This specification is used to compile a generator for corresponding SystemVerilog modules.

DFCxx provides a library of C++ classes (like `Stream`, `Scalar`, `Kernel`), allowing relations between objects to represent the computational logic of a streaming computer.

By inheriting `dfcxx::Kernel` abstract class and implementing at least a single constructor this new class represents the main unit of computation, <i>kernel</i>.

```cpp
#include "dfcxx/DFCXX.h"

class Polynomial2 : public dfcxx::Kernel {
public:
  std::string_view getName() override {
    return "Polynomial2";
  }

  ~Polynomial2() override = default;

  Polynomial2() : dfcxx::Kernel() {
    using dfcxx::DFType;
    using dfcxx::DFVariable;

    const DFType &type = dfUInt(32);
    DFVariable x = io.input("x", type);
    DFVariable square = x * x;
    DFVariable result = square + x;
    DFVariable test = result + x;
    DFVariable out = io.output("out", type);
    out.connect(test);
  }
};
```
User-defined source files and include directories for a kernel are used in the compilation of Utopia HLS, so that the resulting binary executable becomes a generator for application-specific SystemVerilog modules. 

The computational logic of a streaming dataflow computer has to be statically scheduled. Currently there are two implementations of scheduling algorithms: Linear Programming-approach and a greedy as-soon-as-possible algorithm.

When the Utopia HLS binary executable is compiled, its integrated command line interface allows a filesystem path to a JSON latency configuration file to be specified, as well as the chosen scheduling strategy.

## Licensing and Distribution

Utopia is distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Coding Style

See `CODE_STYLE.md` for more details.

## Prerequisites

### Packages

It is recommended to use Utopia HLS on Debian-based operating systems (e.g. Ubuntu 20.04). The required dependencies' package names are specific to `apt` package manager, present in Debian-based operating systems, and are presented below:
* `cmake` ver. 3.13 or higher
* `build-essential`
* `clang` + `lld` or `g++` + `gcc` as a C/C++ compiler
* `liblpsolve55-dev`
* `ninja-build` (preferred) or `make`

The following command can be used to install all of these dependencies regardless of what exactly will be used to compile Utopia HLS:
```
sudo apt install cmake build-essential clang g++ gcc liblpsolve55-dev lld make ninja-build
```

### Precompiled CIRCT & MLIR Installation

Utopia HLS utilizes SystemVerilog-generation capabilities from CIRCT project, which itself is based on LLVM Multi-Level Intermediate Representation (MLIR) project.

**[!!!]Currently supported CIRCT release: 1.72.0, Apr 5 2024 ([link](https://github.com/llvm/circt/releases/tag/firtool-1.72.0))[!!!]**

For both CIRCT and MLIR projects it is possible to download precompiled libraries (this approach is more convenient and saves time), but both projects can also be compiled from sources.

Precompiled releases of CIRCT include the precompiled LLVM MLIR release as well: visit the corresponding [page](https://github.com/llvm/circt/releases) to see all available releases. 

**Note that <ins>it's the libraries that are required</ins>, not the binary executables.**<br>
Look for the archives which have the following names:
`circt-full-static-<ARCH>` or `circt-full-shared-<ARCH>` for static and dynamic libraries respectively.

**[!!!]Current releases have an inconsistency in their configuration files[!!!]:**<br>
after downloading the chosen release archive, extract the files inside and look for the file `lib/cmake/mlir/MLIRTargets.mlir`.

Open `MLIRTargets.mlir` with a text editor of your choice and look the `_NOT_FOUND_MESSAGE_targets` near the end of the file:<br>
for 1.72.0, it's line **3012** for `circt-full-static-linux-x64`-archive and line **3008** for `circt-full-shared-linux-x64.tar`-archive.

Remove all the `"CIRCT*"`-entries from the corresponding `foreach`-statement and save the file.

That's it: both CIRCT and LLVM can be used now.

### Compiling CIRCT & LLVM from scratch

MLIR is included in CIRCT in the form of a `git`-submodule, so the compilation process starts with cloning the CIRCT repository into some directory `CIRCT_DIR`.
```
cd <WORKDIR>
git clone --depth 1 --branch firtool-<VERSION> https://github.com/llvm/circt/ <CIRCT_DIR>
cd <CIRCT_DIR>
git submodule init
git submodule update
```

Now that `llvm`-subdirectory has been initialized, we may build MLIR in some directory `<LLVM_BUILD_DIR>`.

The following command is used to build the simplest MLIR revision, which allows debugging - `RelWithDebInfo` may be changed for `Release` or `Debug` if the user deems it so.

```
cd llvm
cmake -S llvm -B <LLVM_BUILD_DIR> -G Ninja -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build <LLVM_BUILD_DIR>
```

This process is going to take a while.
After building MLIR, it can be used to build CIRCT in some directory `<CIRCT_BUILD_DIR>`.

```
cd <CIRCT_DIR>
cmake -S . -B <CIRCT_BUILD_DIR> -G Ninja -DMLIR_DIR=<LLVM_BUILD_DIR>/lib/cmake/mlir -DLLVM_DIR=<LLVM_BUILD_DIR>/lib/cmake/llvm -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build <CIRCT_BUILD_DIR>
```
## Project compilation

WORK IN PROGRESS

## Usage

WORK IN PROGRESS

## Testing

WORK IN PROGRESS

## DFCxx documentation

WORK IN PROGRESS


