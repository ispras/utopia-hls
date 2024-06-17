#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

rm -rf $UTOPIA_HLS_HOME/build

cmake -S . -B build -G Ninja
cmake --build build
