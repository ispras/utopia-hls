//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#include "hls/library/ipxact_parser.h"

using namespace xercesc;
using namespace std;

namespace eda::hls::library {

void IPXACTParser::parseCatalog(const std::string& catalog_fname) {
  //Initialization.
  //std::cout << "parseCatalog" << std::endl;
  XMLPlatformUtils::Initialize();
  //XMLCh tempStr[100];
  //XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(
    XMLString::transcode("LS"));
  DOMLSParser* parser = ((DOMImplementationLS*)impl)->createLSParser(
    DOMImplementationLS::MODE_SYNCHRONOUS, 0);
  xercesc::DOMDocument *doc = 0;
  doc = parser->parseURI(catalog_fname.c_str());
  XMLCh *ipxactFile_tag = XMLString::transcode("ipxact:ipxactFile");
  XMLCh *vlnv_tag = XMLString::transcode("ipxact:vlnv");
  XMLCh *name_tag = XMLString::transcode("ipxact:name");
  XMLCh *name_attr = XMLString::transcode("name");
  size_t ipxactFile_size = doc->getElementsByTagName(
    ipxactFile_tag)->getLength();
  for (size_t i = 0; i < ipxactFile_size; i++) {
    const DOMElement* ipxactFile = (const DOMElement*)(
      doc->getElementsByTagName(ipxactFile_tag)->item(i));
    const auto* vlnv = ipxactFile->getElementsByTagName(vlnv_tag)->item(0);
    std::string key = std::string(XMLString::transcode(
      vlnv->getAttributes()->getNamedItem(name_attr)->getNodeValue()));
    const auto* name = ipxactFile->getElementsByTagName(name_tag)->item(0);
    std::string value = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    comp_fnames.insert({key, value});
  }
  /*for (const auto &[key, value] : comp_fnames) {
    std::cout << key << std::endl << value << std::endl;
  }*/
  //Termination.
  delete ipxactFile_tag;
  delete vlnv_tag;
  delete name_tag;
  delete name_attr;
  parser->release();
  XMLPlatformUtils::Terminate();
  return;
}

void IPXACTParser::parseComponent(const std::string &name,
                                  library::Parameters &params,
                                  library::Ports &ports) {
  //Initialization.
  //std::cout << "parseComponent" << std::endl;
  XMLPlatformUtils::Initialize();
  //XMLCh tempStr[100];
  //XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(
    XMLString::transcode("LS"));
  DOMLSParser* parser = ((DOMImplementationLS*)impl)->createLSParser(
    DOMImplementationLS::MODE_SYNCHRONOUS, 0);
  xercesc::DOMDocument *doc = 0;
  //std::cout << name << std::endl;
  //std::cout << comp_fnames[name] << std::endl;
  doc = parser->parseURI(comp_fnames[name].c_str());
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
  //IP-XACT tags parsing.
  size_t port_count = doc->getElementsByTagName(port_tag)->getLength();
  for (size_t i = 0; i < port_count; i++) {
    const DOMElement* port = (const DOMElement*)(
      doc->getElementsByTagName(port_tag)->item(i));
    const auto* name = port->getElementsByTagName(name_tag)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    const auto* direction = port->getElementsByTagName(
      direction_tag)->item(0);
    std::string direction_str = std::string(XMLString::transcode(
      direction->getFirstChild()->getNodeValue()));
    const auto* left = port->getElementsByTagName(left_tag)->item(0);
    int left_int = 0;
    if (left != NULL) {
      left_int = std::stoi(XMLString::transcode(
        left->getFirstChild()->getNodeValue()));
    }
    //May be needed in future.
    /*const auto* right = port->getElementsByTagName()->item(0);
    std::string right = std::string(XMLString::transcode(
        right->getFirstChild()->getNodeValue()));*/
    //Creating Port.
    if (direction_str == "in") {
      /*std::cout << "in " << name_str;
      if (left != NULL) {
        std::cout << " [" << left_int << ":" << "0" << "]";
      }
      std::cout << std::endl;*/
      ports.push_back(library::Port(name_str,
                                    library::Port::IN,
                                    0,
                                    left_int + 1));
    }
    if (direction_str == "out") {
      /*std::cout << "out " << name_str;
      if (left != NULL) {
        std::cout << " [" << left_int << ":" << "0" << "]";
      }
      std::cout << std::endl;*/
      ports.push_back(library::Port(name_str,
                                    library::Port::OUT,
                                    0,
                                    left_int + 1));
    }
  }
  //Vendor extensions tags parsing.
  size_t parameter_count = doc->getElementsByTagName(
    parameter_tag_v)->getLength();
  for (size_t i = 0; i < parameter_count; i++) {
    const DOMElement* parameter = (const DOMElement*)(
      doc->getElementsByTagName(parameter_tag_v)->item(i));
    const auto* name = parameter->getElementsByTagName(name_tag_v)->item(0);
    std::string name_str = std::string(XMLString::transcode(
      name->getFirstChild()->getNodeValue()));
    const auto* value = parameter->getElementsByTagName(value_tag_v)->item(0);
    int value_int = std::stoi(XMLString::transcode(
      value->getFirstChild()->getNodeValue()));
    const auto* left = parameter->getElementsByTagName(left_tag_v)->item(0);
    int left_int = std::stoi(XMLString::transcode(
      left->getFirstChild()->getNodeValue()));
    const auto* right = parameter->getElementsByTagName(right_tag_v)->item(0);
    int right_int = std::stoi(XMLString::transcode(
      right->getFirstChild()->getNodeValue()));
    //Creating Parameter.
    params.add(library::Parameter(name_str,
                                  library::Constraint(left_int, right_int),
                                  value_int));
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
  parser->release();
  XMLPlatformUtils::Terminate();
  return;
}
} // namespace eda::hls::library
