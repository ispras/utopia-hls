[//]: <> (SPDX-License-Identifier: Apache-2.0)

# Utopia EDA

Utopia is an open-source HLS-based EDA for digital hardware design.

The EDA takes the following inputs:
* an algorithmic description of the accelerator (IP core);
* a configuration of the target hardware (FPGA, ULA, or ASIC);
* custom constraints.

And produces the following outputs:
* an RTL model of the accelerator;
* recommendations for placing elements of the RTL model on a chip;
* an implementation of the API for interacting with the accelerator (if necessary).

## Licensing and Distribution

Utopia is distributed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Coding Style

See `CODE_STYLE.md` for more details.

## General Notes

### Environment variables

Several environment variables should be set during the Utopia building.
To keep the value of the environment variable in your system permanenly, add the
appropriate command to either `.profile` or `.bashrc` file. For example, to set
the `/usr` value to the `SOME_DIR` variable, the command should be as follows:
```
export SOME_DIR=/usr
```
To make this variable-value pair active in your terminal session you need either
to reboot your operating system or run the following command (it is supposed
that the variable was established in `.profile`):
```
source ~/.profile
```
To check if value is set, use `echo $SOME_DIR` command.

### Parallelization

To speed up building, several tools like `make` or `ninja` provide options
that aimed at running multiple jobs in parallel. For example, to run `make`
in 4 threads do the following:
```
make -j4
```
It is recommended to use such options upon building the project dependencies
(like CIRCT/LLVM), due to significant reduction of the compilation time.

### Working directory

In this guide `<workdir>` path appears several times. This string denotes
a path to user's working directory (e.g. `~/work`, `~/projects`). It is not
necessary for the project's building to have the same `<workdir>` all the
times it is used in this guide.

## System Requirements

The recommended operating system for Utopia is Ubuntu 20.04. The package names
below are specific to this operating system:

* `autoconf`
* `bison`
* `clang`
* `clang-tidy`
* `cmake`
* `flex`
* `g++`
* `gcc`
* `iverilog`
* `liblpsolve55-dev`
* `libtool`
* `libxerces-c3.2`
* `libxerces-c-dev`
* `lld`
* `make`
* `ninja-build`
* `python`
* `zlib1g`
* `zlib1g-dev`

To install them, do the following:
```
sudo apt install autoconf bison clang clang-tidy cmake flex g++ gcc \
    iverilog liblpsolve55-dev libtool libxerces-c3.2 libxerces-c-dev lld \
    make ninja-build python zlib1g zlib1g-dev
```

### CIRCT Installation

LLVM requires a significant amount of RAM (about 8 Gb or more) to build.
Please take this into account while moving through the guide.

#### Check out LLVM and CIRCT repos

```
cd <workdir>
git clone --recursive https://github.com/circt/circt.git
cd circt
git checkout 6de88ef7
git submodule update
```

#### LLVM/MLIR Installation

```
cd <workdir>/circt
mkdir llvm/build
cd llvm/build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
ninja
```
Set `MLIR_DIR` environment variable to directory with MLIR CMake files:
```
export MLIR_DIR=<workdir>/circt/llvm/build/lib/cmake/mlir/
```

#### CIRCT Installation

```
cd <workdir>/circt
mkdir build
cd build
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DVERILATOR_DISABLE=ON
ninja
```
Add `<workdir>/circt/build/bin` and `<workdir>/circt/llvm/build/bin`
to your `PATH` environment variable:
```
export PATH=<workdir>/circt/build/bin:<workdir>/circt/llvm/build/bin:$PATH
```

### Z3 Installation

```
cd <workdir>
git clone https://github.com/Z3Prover/z3.git
cd z3
python scripts/mk_make.py
cd build
make
sudo make install
```
If you would like to install Z3 to a non-standard location,
please set `Z3_DIR` environment variable to Z3 build/installation directory.

### C++ CTemplate Installation

```
cd <workdir>
git clone https://github.com/OlafvdSpek/ctemplate.git
cd ctemplate
./autogen.sh
./configure --prefix=/usr
make
sudo make install
```
If you would like to install CTemplate to a non-standard location, please
specify `--prefix` option of `configure` script to installation directory
you want and set `CT_DIR` environment variable to it too.


## Working in Command Line

### Clone Project Repository and Set Environment Variable

```
cd <workdir>
git clone --recursive https://github.com/ispras/utopia-hls
cd utopia-hls
export UTOPIA_HLS_HOME=<workdir>/utopia-hls
```
Please keep `UTOPIA_HLS_HOME` variable and its value in your system permanently.
### Building Project

```
cmake -S . -B build -G Ninja
cmake --build build
```
or simply run the following script:
```
./build.sh
```
If you've modified some of the project files, you can use `rebuild.sh` script
for incremental build.

### Running Utopia EDA

```
rm -rf $UTOPIA_HLS_HOME/output
./build/src/umain hls <file(s)>
```

### Running Tests

#### Run All Tests

```
rm -rf $UTOPIA_HLS_HOME/output
$UTOPIA_HLS_HOME/build/test/utest
```
or
```
./run-tests.sh
```
#### Run Specific Tests

```
./build/test/utest --gtest_filter=<test-pattern>
```
Test pattern accepts ```*``` and ```?``` wildcards.

#### List Available Tests

```
./build/test/utest --gtest_list_tests
```

## Working in Visual Studio Code

### Installing VS Studio w/ C/C++ and CMake Extensions
* Download the VS Studio package
  * Go to https://code.visualstudio.com/docs/?dv=linux64_deb
  * Wait until the package is downloaded
* Install the VS Studio package
  ```
  sudo apt install -f ~/Downloads/code_1.60.2-1632313585_amd64.deb
  ```
* Install the C/C++ and CMake extensions
  * Start VS Code
  * Press the `Ctrl+Shift+x` key combination
    * Find and install the `C/C++` extension
    * Find and install the `CMake Tools` extension
  * Click on the `No Kit Selected` text in the status bar
    * Select the kit to use (for example, `GCC 9.3.0 x86_64-linux-gnu`)

### Opening/Building Project

* Clone project repository (see above)
* Click on the `File` and `Open Folder...` menu items
  * Select the `<UTOPIA_HLS_HOME>/src` directory (or `<UTOPIA_HLS_HOME>`)
  * Press the `I trust the authors` button
* Click on the `Build` text in the status bar
