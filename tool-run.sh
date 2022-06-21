#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

./build/src/umain rtl test/data/ril/test.ril

./build/src/umain hls --output-dir ./output/ \
                      --output-dot  test.dot \
                      --output-mlir test.mlir \
                      --output-vlog test.v \
                      --output-test test_tb.v \
                      test/data/hil/test.hil

./build/src/umain hls --output-dir ./output/ \
                      --output-dot  idct.dot \
                      --output-mlir idct.mlir \
                      --output-vlog idct.v \
                      --output-test idct_tb.v \
                      test/data/hil/idct.hil
