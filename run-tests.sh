#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

rm -rf $UTOPIA_HLS_HOME/output

${UTOPIA_HLS_HOME}/build/test/utest
