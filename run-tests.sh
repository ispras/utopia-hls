#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

# Build project with tests

rm -rf build

cmake -S . -B build -G Ninja \
      -DCMAKE_PREFIX_PATH="/home/ssedai/sources/circt.git/build;/home/ssedai/sources/circt.git/llvm/build" \
      -DSRC_FILES="~/projects/utopia-hls/examples/polynomial2/polynomial2.cpp" \
      -DBUILD_TESTS=ON #-DCMAKE_CXX_COMPILER=clang++

cmake --build build

# Run tests
build/test/utest
