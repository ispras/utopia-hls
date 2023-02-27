//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>

namespace fs = std::filesystem;

bool stringIsInteger(std::string inputString) {
  return std::regex_match(inputString, std::regex("[(-|\\+)]?[0-9]+"));
}

void printInfo() {
    std::cout << "Karatsuba multiplexer generator. Usage:" << std::endl;
    std::cout << "kamulgen <outDirName> <outFileName> <inputWidth>"
              << std::endl;
    std::cout << "<outDirName>: A path to the output file."
              << " " <<  "Created if nonexistant."
              << std::endl;
    std::cout << "<outFileName>: The name of the output file."
              << std::endl;
    std::cout << "<inputWidth>: Width of the input. Must be a positive integer."
              << " " << "The output will have twice the size of the input."
              << std::endl;
}

static constexpr const char* indent = "    ";
static constexpr const char* mindent = "        ";
static constexpr const char* moduleName = "MUL";

int main(int argc, char* argv[]) {
  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    printInfo();
    return 0;
  }
  // Error handling
  if (argc < 4) {
    std::cout << "Not enough arguments!" << std::endl;
    return 1;
  }
  if (argc > 4) {
    std::cout << "Too many arguments!" << std::endl;
    return 1;
  }
  if (!stringIsInteger(argv[3]) || atoi(argv[3]) <= 0) {
    std::cout << "Input width must be positive integer!" << std::endl;
    return 1;
  }

  // Recursively create directories
  const fs::path fsPath = std::string(argv[1]);
  fs::create_directories(fsPath);

  // Open file
  std::ofstream outputFile;
  outputFile.open(fsPath / std::string(argv[2]));

  // Print interface
  outputFile << "module " << moduleName << "(" << std::endl;
  outputFile << mindent << "input clock," << std::endl;
  outputFile << mindent << "input reset," << std::endl;
  uint inputWidth = atoi(argv[3]);
  outputFile << mindent << "input " << "[" << inputWidth - 1 << ":" << 0
             << "] x,"  << std::endl;
  outputFile << mindent << "input " << "[" << inputWidth - 1 << ":" << 0
             << "] y,"  << std::endl;
  outputFile << mindent << "output " << "[" << inputWidth * 2 - 1 << ":" 
             << 0 << "] z);" << std::endl;
  outputFile << std::endl;
  // Print body
  outputFile << indent << "localparam h_w = " << inputWidth << " / 2" << ";"
             << std::endl;
  // Print wires
  outputFile << indent << "wire " << "[" << "h_w-1:0" << "]" << " " << "a, b;"
             << std::endl;
  outputFile << indent << "wire " << "[" << "h_w-1:0" << "]" << " " << "c, d;"
             << std::endl;
  outputFile << indent << "wire " << "[" << "h_w*2-1:0" << "]" << " " 
             << "ac, bd;" << std::endl;
  outputFile << indent << "wire " << "[" << "h_w*2:0" << "]" << " " << "t;"
            << std::endl;
  // Print assignments
  outputFile << indent << "assign " << "{a, b} " << "= " << "x;" << std::endl;
  outputFile << indent << "assign " << "{c, d} " << "= " << "y;" << std::endl;
  outputFile << indent << "assign " << "ac " << "= " << "a * c;" << std::endl;
  outputFile << indent << "assign " << "bd " << "= " << "b * d;" << std::endl;
  outputFile << indent << "assign " << "t " << "= " << "(a + b) * (c + d); "
             << std::endl;
  outputFile << indent << "assign " << "z " << "= " << "ac << " << inputWidth
             << " + (t - ac - bd) << h_w + bd;" << std::endl;
  // Print endmodule
  outputFile << "endmodule";
  // Close file
  outputFile.close();
}