#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <cstring>

bool isInt(char* input) {
  for (int i=0; input[i]; i++)
  if (isdigit(input[i]) == true) {
      return true;
  }
  return false;
}

static constexpr const char* indent = "    ";
static constexpr const char* mindent = "       ";
static constexpr const char* moduleName = "MUL";

int main(int argc, char* argv[]) {
  std::ofstream outputFile;
  if (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
    std::cout << "Karatsuba multiplexer generator. Usage:" << std::endl;
    std::cout << "kamulgen outDirName outFileName inputWidth"
              << std::endl;
    std::cout << "outDirName: path to output file. Created if nonexistant."
              << std::endl;
    std::cout << "outFileName: name of the output file."
              << std::endl;
    std::cout << "inputWidth: width of the input. Must be positive integer."
              << "Output will have twice the size."
              << std::endl;
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
  if (!isInt(argv[3]) || atoi(argv[3]) <= 0) {
    std::cout << "Input width must be positive integer!" << std::endl;
    return 1;
  }
  std::string outputDirectoryName = std::string(argv[1]);
  int start = 0;
  int end = outputDirectoryName.find("/");
  std::string dir = "";
  while (end != -1) {
  dir = dir + outputDirectoryName.substr(start, end - start) + "/";
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directory(dir);
  }
    start = end + 1;
    end = outputDirectoryName.find("/", start);
  }
  // Open file
  outputFile.open(std::string(argv[1]) + "/" + std::string(argv[2]));
  // Interface
  outputFile << "module " << moduleName << "(" << std::endl;
  outputFile << mindent << "input clock," << std::endl;
  outputFile << mindent << "input reset," << std::endl;
  outputFile << mindent << "input " << "[" << atoi(argv[3]) - 1 << ":" << 0
             << "] x," << std::endl;
  outputFile << mindent << "input " << "[" << atoi(argv[3]) - 1 << ":" << 0
             << "] y," << std::endl;
  outputFile << mindent << "output " << "[" << atoi(argv[3]) * 2 - 1 << ":" 
             << 0 << "] z);" << std::endl;
  outputFile << std::endl;
  // Body
  outputFile << indent << "localparam h_w = " << argv[3] << " / 2" << ";"
             << std::endl;
  // Wires
  outputFile << indent << "wire " << "[" << "h_w-1:0" << "]" << " " << "a, b;"
             << std::endl;
  outputFile << indent << "wire " << "[" << "h_w-1:0" << "]" << " " << "c, d;"
             << std::endl;
  outputFile << indent << "wire " << "[" << "h_w*2-1:0" << "]" << " " 
             << "ac, bd;" << std::endl;
  outputFile << indent << "wire " << "[" << "h_w*2:0" << "]" << " " << "t;"
            << std::endl;
  // Assignments
  outputFile << indent << "assign " << "{a, b} " << "= " << "x;" << std::endl;
  outputFile << indent << "assign " << "{c, d} " << "= " << "y;" << std::endl;
  outputFile << indent << "assign " << "ac " << "= " << "a * c;" << std::endl;
  outputFile << indent << "assign " << "bd " << "= " << "b * d;" << std::endl;
  outputFile << indent << "assign " << "t " << "= " << "(a + b) * (c + d); "
             << std::endl;
  outputFile << indent << "assign " << "z " << "= " << "ac << " << argv[3]
             << " + (t - ac - bd) << h_w + bd;" << std::endl;
  // Endmodule
  outputFile << "endmodule";
  // Close file
  outputFile.close();
}
