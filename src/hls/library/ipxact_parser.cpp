//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/ipxact_parser.h"
#include "hls/library/element_core.h"
#include "hls/library/element_generator.h"

#include "util/path.h"

#include <xercesc/dom/DOM.hpp>

using namespace eda::hls::mapper::config::hwconfig;
using namespace eda::utils;
using namespace xercesc;

namespace eda::hls::library {

// TODO: Duplication (see compiler). Move this function into utils.
std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); }
                  );
    return s;
}

void IPXACTParser::initialize() {
  XMLPlatformUtils::Initialize();
}

void IPXACTParser::finalize() {
  readFileNames.clear();
  XMLPlatformUtils::Terminate();
}

void IPXACTParser::parseCatalog(const std::string &libraryPath,
                                const std::string &catalogPath) {
  this->libraryPath = libraryPath;

  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(
                              XMLString::transcode("LS"));

  uassert(impl != nullptr, "DOMImplementation is not found!");

  DOMLSParser *parser =
    dynamic_cast<DOMImplementationLS*>(impl)->
      createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0);

  const DOMDocument *doc =
    parser->parseURI((libraryPath + "/" + catalogPath).c_str());

  size_t ipxactFileSize =
    doc->getElementsByTagName(ipxIpxFileTag)->getLength();

  for (size_t i = 0; i < ipxactFileSize; i++) {
    const DOMElement *ipxactFile = (const DOMElement*)(
      doc->getElementsByTagName(ipxIpxFileTag)->item(i));

    const auto *vlnv = ipxactFile->getElementsByTagName(ipxVlnvTag)->item(0);

    std::string key = std::string(XMLString::transcode(
      vlnv->getAttributes()->getNamedItem(nameAttr)->getNodeValue()));

    const auto *name = ipxactFile->getElementsByTagName(ipxNameTag)->item(0);

    std::string value = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));

    value = correctPath(libraryPath) + "/" + value;

    readFileNames.insert({key, value});
  }
  /*for (const auto &[key, value] : readFileNames) {
    std::cout << key << std::endl << value << std::endl;
  }*/
  parser->release();
}

bool IPXACTParser::hasComponent(const std::string &name,
                                const HWConfig &hwconfig) {
  return readFileNames.count(toLower(name)) > 0 ? true : false;
}

std::shared_ptr<MetaElement> IPXACTParser::parseComponent(
    const std::string &name/*, const std::string &libraryName*/) {
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
  /*std::cout << name << std::endl;
  std::cout << readFileNames[toLower(name)] << std::endl;*/
  doc = parser->parseURI(readFileNames[toLower(name)].c_str());

  //IP-XACT tags parsing.
  size_t port_count = doc->getElementsByTagName(ipxPortTag)->getLength();
  for (size_t i = 0; i < port_count; i++) {
    const DOMElement *port = (const DOMElement*)(
      doc->getElementsByTagName(ipxPortTag)->item(i));
    const auto *name = port->getElementsByTagName(ipxNameTag)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    const auto *direction = port->getElementsByTagName(
      ipxDirectTag)->item(0);
    std::string direction_str = std::string(XMLString::transcode(
      direction->getFirstChild()->getNodeValue()));
    const auto *left = port->getElementsByTagName(ipxLeftTag)->item(0);
    int left_int = -1;
    std::string value;
    bool isParam = false;
    std::string param;
    if (left != nullptr) {
      value = XMLString::transcode(left->getFirstChild()->getNodeValue());
      std::string value = XMLString::transcode(
        left->getFirstChild()->getNodeValue());
      //If value is a parameter.
      //TODO: Regular expressions.
      if (isalpha(value.c_str()[0])) {
          isParam = true;
          param = value;
      } else {
        left_int = std::stoi(XMLString::transcode(
          left->getFirstChild()->getNodeValue()));
      }
    }
    //Creating Port.
        /*std::cout << "in " << name_str;
        if (left != nullptr) {
          std::cout << " [" << left_int << ":" << "0" << "]";
        }
        std::cout << std::endl;*/
      if (isParam) {
        ports.push_back(library::Port(name_str,
                                      direction_str == "in" ? Port::IN : Port::OUT,
                                      left_int + 1,
                                      model::Parameter(std::string(param))));
      } else {
        ports.push_back(library::Port(name_str,
                                      direction_str == "in" ? Port::IN : Port::OUT,
                                      left_int + 1,
                                      model::Parameter(std::string("WIDTH"),
                                                       left_int + 1)));
      }
  }
  //Vendor extensions tags parsing.
  size_t parameter_count = doc->getElementsByTagName(
    k2ParamTag)->getLength();
  for (size_t i = 0; i < parameter_count; i++) {
    const DOMElement *parameter = (const DOMElement*)(
      doc->getElementsByTagName(k2ParamTag)->item(i));
    const auto *name = parameter->getElementsByTagName(k2NameTag)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    const auto *value = parameter->getElementsByTagName(k2ValueTag)->item(0);
    int value_int = std::stoi(XMLString::transcode(
      value->getFirstChild()->getNodeValue()));
    const auto *left = parameter->getElementsByTagName(k2LeftTag)->item(0);
    int left_int = std::stoi(XMLString::transcode(
      left->getFirstChild()->getNodeValue()));
    const auto *right = parameter->getElementsByTagName(k2RightTag)->item(0);
    int right_int = std::stoi(XMLString::transcode(
      right->getFirstChild()->getNodeValue()));
    //Creating Parameter.
    params.add(model::Parameter(name_str,
                                model::Constraint<unsigned>(left_int, right_int),
                                value_int));
  }
  std::string type;
  std::string genPath;
  std::string comPath;
  size_t generator_count = doc->getElementsByTagName(
    ipxCompGensTag)->getLength();
  std::shared_ptr<MetaElement> metaElement;
  if (generator_count != 0) {
    const auto *genExe = doc->getElementsByTagName(ipxGenExeTag)->item(0);
    genPath = std::string(XMLString::transcode(
      genExe->getFirstChild()->getNodeValue()));
    metaElement = std::shared_ptr<MetaElement>(new ElementGenerator(name,
                                                                    params,
                                                                    ports,
                                                                    genPath));
  } else {
    const DOMElement *file = (const DOMElement*)(doc->getElementsByTagName(
      ipxFileTag)->item(0));
    const auto *name = file->getElementsByTagName(ipxNameTag)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    comPath = libraryPath + "/" + name_str;
    metaElement = std::shared_ptr<MetaElement>(new ElementCore(name_str,
                                                               params,
                                                               ports,
                                                               comPath));
  }

  parser->release();

  return metaElement;
}

} // namespace eda::hls::library
