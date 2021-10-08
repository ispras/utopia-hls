#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

rm -rf build

cmake -S . -B build -G Ninja
cmake --build build

./build/src/umain test/ril/test.ril test/hil/test.hil
./build/test/utest
