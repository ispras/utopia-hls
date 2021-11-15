#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

./build/src/umain test/ril/test.ril test/hil/test.hil
./build/test/utest
