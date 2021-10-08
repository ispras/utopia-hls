#pragma once

#include <cassert>
#include <iostream>

#define FATAL "Fatal"
#define LOG(x) std::cout << ": "
#define CHECK(x) if (!(x)) std::cout << ": "
#define ICHECK(x) CHECK(x)
#define CHECK_EQ(x, y) if ((x) != (y)) std::cout << ": "
#define CHECK_LT(x, y) if ((x) >= (y)) std::cout << ": "
#define CHECK_GE(x, y) if ((x) < (y))  std::cout << ": "
