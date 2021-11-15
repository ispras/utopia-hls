#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

rm -rf build

cmake -S . -B build -G Ninja
cmake --build build
