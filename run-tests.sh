#!/bin/sh

# SPDX-License-Identifier: Apache-2.0

<<<<<<< Updated upstream
./build/test/utest --gtest_filter=-*BigPartitionTest*
=======
rm -rf ./output
./build/test/utest #--gtest_filter=-CheckNetlistTest.*
>>>>>>> Stashed changes
