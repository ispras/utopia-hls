//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#include "ipxact_parser.h"

using namespace xercesc;
using namespace std;

namespace eda::hls::library {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); }
                  );
    return s;
}

bool isLetter(char c) {
  if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
    return true;
  }
  return false;
}

bool isDigit(char c) {
  if (c >= '0' && c <= '9') {
    return true;
  }
  return false;
}

void IPXACTParser::initialize() {
  XMLPlatformUtils::Initialize();
}

void IPXACTParser::finalize() {
  comp_fnames.clear();
  XMLPlatformUtils::Terminate();
}

void IPXACTParser::parseCatalog(const std::string &libraryPath,
                                const std::string &catalogPath) {
  this->libraryPath = libraryPath;
  //Initialization.
  //std::cout << "parseCatalog" << std::endl;
  //XMLCh tempStr[100];
  //XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(
    XMLString::transcode("LS"));
  DOMLSParser *parser = dynamic_cast<DOMImplementationLS*>(impl)->
    createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0);
  xercesc::DOMDocument *doc = nullptr;
  doc = parser->parseURI((this->libraryPath + "/" + catalogPath).c_str());
  XMLCh *ipxactFile_tag = XMLString::transcode("ipxact:ipxactFile");
  XMLCh *vlnv_tag = XMLString::transcode("ipxact:vlnv");
  XMLCh *name_tag = XMLString::transcode("ipxact:name");
  XMLCh *name_attr = XMLString::transcode("name");
  size_t ipxactFile_size = doc->getElementsByTagName(
    ipxactFile_tag)->getLength();
  for (size_t i = 0; i < ipxactFile_size; i++) {
    const DOMElement *ipxactFile = (const DOMElement*)(
      doc->getElementsByTagName(ipxactFile_tag)->item(i));
    const auto *vlnv = ipxactFile->getElementsByTagName(vlnv_tag)->item(0);
    std::string key = std::string(XMLString::transcode(
      vlnv->getAttributes()->getNamedItem(name_attr)->getNodeValue()));
    const auto *name = ipxactFile->getElementsByTagName(name_tag)->item(0);
    std::string value = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    comp_fnames.insert({key, value});
  }
  for (const auto &[key, value] : comp_fnames) {
    std::cout << key << std::endl << value << std::endl;
  }
  //Termination.
  delete ipxactFile_tag;
  delete vlnv_tag;
  delete name_tag;
  delete name_attr;
  parser->release();
}

bool IPXACTParser::hasComponent(const std::string &name) {
  return comp_fnames.count(toLower(name)) > 0 ? true : false;
}

std::shared_ptr<MetaElement> IPXACTParser::parseComponent(
    const std::string &name) {
  Parameters params;
  std::vector<Port> ports;
  /*for (const auto &[key, value] : comp_fnames) {
    std::cout << key << std::endl << value << std::endl;
  }*/
  //Initialization.
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(
    XMLString::transcode("LS"));
  DOMLSParser *parser = ((DOMImplementationLS*)impl)->createLSParser(
    DOMImplementationLS::MODE_SYNCHRONOUS, 0);
  xercesc::DOMDocument *doc = nullptr;
  //std::cout << name << std::endl;
  //std::cout << comp_fnames[toLower(name)] << std::endl;
  doc = parser->parseURI(comp_fnames[toLower(name)].c_str());
  //IP-XACT tags.
  XMLCh *port_tag = XMLString::transcode("ipxact:port");
  XMLCh *name_tag = XMLString::transcode("ipxact:name");
  XMLCh *vlnv_tag = XMLString::transcode("ipxact:vlnv");
  XMLCh *direction_tag = XMLString::transcode("ipxact:direction");
  XMLCh *vector_tag = XMLString::transcode("ipxact:vector");
  XMLCh *left_tag = XMLString::transcode("ipxact:left");
  XMLCh *right_tag = XMLString::transcode("ipxact:right");
  //IP-XACT vendor extensions tags.
  XMLCh *parameter_tag_v = XMLString::transcode("kactus2:parameter");
  XMLCh *name_tag_v = XMLString::transcode("kactus2:name");
  XMLCh *value_tag_v = XMLString::transcode("kactus2:value");
  XMLCh *left_tag_v = XMLString::transcode("kactus2:left");
  XMLCh *right_tag_v = XMLString::transcode("kactus2:right");
  //For component generators.
  XMLCh *componentGenerators_tag = XMLString::transcode("ipxact:componentGenerators");
  XMLCh *generatorExe_tag = XMLString::transcode("ipxact:generatorExe");
  //For static compomonents.
  //XMLCh *fileSet_tag = XMLString::transcode("ipxact:fileSet");
  XMLCh *file_tag = XMLString::transcode("ipxact:file");

  //IP-XACT tags parsing.
  size_t port_count = doc->getElementsByTagName(port_tag)->getLength();
  for (size_t i = 0; i < port_count; i++) {
    const DOMElement *port = (const DOMElement*)(
      doc->getElementsByTagName(port_tag)->item(i));
    const auto *name = port->getElementsByTagName(name_tag)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    const auto *direction = port->getElementsByTagName(
      direction_tag)->item(0);
    std::string direction_str = std::string(XMLString::transcode(
      direction->getFirstChild()->getNodeValue()));
    const auto *left = port->getElementsByTagName(left_tag)->item(0);
    int left_int = -1;
    std::string value;
    bool isParam = false;
    char param = ' ';
    if (left != nullptr) {
      value = XMLString::transcode(left->getFirstChild()->getNodeValue());
      std::string value = XMLString::transcode(
        left->getFirstChild()->getNodeValue());
      //If value is a parameter.
      if (value.size() == 1) {
        if (isLetter(value.c_str()[0])) {
          isParam = true;
          param = value.c_str()[0];
        } else {
          std::cout << "Error in IP-XACT specification!" << std::endl;
          return nullptr;
        }
      } else {
        for (size_t j = 0; j < value.size(); j++) {
          if (!isDigit(value.c_str()[j])) {
            std::cout << "Error in IP-XACT specification!" << std::endl;
            return nullptr;
          }
        }
      }
      if (!isParam) {
        left_int = std::stoi(XMLString::transcode(
          left->getFirstChild()->getNodeValue()));
      }
    }
    //Creating Port.
    if (direction_str == "in") {
      /*std::cout << "in " << name_str;
      if (left != nullptr) {
        std::cout << " [" << left_int << ":" << "0" << "]";
      }
      std::cout << std::endl;*/
      ports.push_back(library::Port(name_str,
                                    library::Port::IN,
                                    0,
                                    left_int + 1,
                                    isParam,
                                    param));
    }
    if (direction_str == "out") {
      /*std::cout << "out " << name_str;
      if (left != nullptr) {
        std::cout << " [" << left_int << ":" << "0" << "]";
      }*/
      ports.push_back(library::Port(name_str,
                                    library::Port::OUT,
                                    0,
                                    left_int + 1,
                                    isParam,
                                    param));
    }
  }
  //Vendor extensions tags parsing.
  size_t parameter_count = doc->getElementsByTagName(
    parameter_tag_v)->getLength();
  for (size_t i = 0; i < parameter_count; i++) {
    const DOMElement *parameter = (const DOMElement*)(
      doc->getElementsByTagName(parameter_tag_v)->item(i));
    const auto *name = parameter->getElementsByTagName(name_tag_v)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    const auto *value = parameter->getElementsByTagName(value_tag_v)->item(0);
    int value_int = std::stoi(XMLString::transcode(
      value->getFirstChild()->getNodeValue()));
    const auto *left = parameter->getElementsByTagName(left_tag_v)->item(0);
    int left_int = std::stoi(XMLString::transcode(
      left->getFirstChild()->getNodeValue()));
    const auto *right = parameter->getElementsByTagName(right_tag_v)->item(0);
    int right_int = std::stoi(XMLString::transcode(
      right->getFirstChild()->getNodeValue()));
    //Creating Parameter.
    params.add(library::Parameter(name_str,
                                  library::Constraint(left_int, right_int),
                                  value_int));
  }
  std::string type;
  std::string genPath;
  std::string comPath;
  size_t generator_count = doc->getElementsByTagName(
    componentGenerators_tag)->getLength();
  std::shared_ptr<MetaElement> metaElement;
  if (generator_count != 0) {
    const auto *genExe = doc->getElementsByTagName(generatorExe_tag)->item(0);
    genPath = std::string(XMLString::transcode(
      genExe->getFirstChild()->getNodeValue()));
    metaElement = std::shared_ptr<MetaElement>(new ElementGenerator(name,
                                                                    params,
                                                                    ports,
                                                                    genPath));
  } else {
    const DOMElement *file = (const DOMElement*)(doc->getElementsByTagName(
      file_tag)->item(0));
    const auto *name = file->getElementsByTagName(name_tag)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    comPath = libraryPath + "/" + name_str;
    metaElement = std::shared_ptr<MetaElement>(new ElementCore(name_str,
                                                               params,
                                                               ports,
                                                               comPath));
  }
  //Termination.
  delete port_tag;
  delete name_tag;
  delete vlnv_tag;
  delete direction_tag;
  delete vector_tag;
  delete left_tag;
  delete right_tag;
  delete parameter_tag_v;
  delete name_tag_v;
  delete value_tag_v;
  delete left_tag_v;
  delete right_tag_v;
  delete componentGenerators_tag;
  delete generatorExe_tag;
  delete file_tag;

  parser->release();

  return metaElement;
}

} // namespace eda::hls::library
