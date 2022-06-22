#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

./build/src/umain rtl test/data/ril/test.ril

./build/src/umain hls --output-dir ./output/ \
                      --output-dot  test.dot \
                      --output-mlir test.mlir \
                      --output-lib  test_lib.v \
                      --output-top  test_top.v \
                      --output-test test_tb.v \
                      test/data/hil/test.hil

./build/src/umain hls --output-dir ./output/ \
                      --output-dot  idct.dot \
                      --output-mlir idct.mlir \
                      --output-lib  idct_lib.v \
                      --output-top  idct_top.v \
                      --output-test idct_tb.v \
                      test/data/hil/idct.hil

./build/src/umain hls --output-dir ./output/ \
                      --output-dot  feedback.dot \
                      --output-mlir feedback.mlir \
                      --output-lib  feedback_lib.v \
                      --output-top  feedback_top.v \
                      --output-test feedback_tb.v \
                      test/data/hil/feedback.hil
