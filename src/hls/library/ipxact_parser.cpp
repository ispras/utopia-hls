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

#include <filesystem>
#include <xercesc/dom/DOM.hpp>

using namespace eda::hls::mapper;
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

bool stringIsDigit(std::string str) {
  for (size_t i = 0; i < str.length(); i++) {
    if (!isdigit(str.at(i))) {
      return false;
    }
  }
  return true;
}

void IPXACTParser::initialize() {
  XMLPlatformUtils::Initialize();
}

void IPXACTParser::finalize() {
  readFileNames.clear();
  XMLPlatformUtils::Terminate();
}

std::string IPXACTParser::getStrValueFromTag(const DOMElement *element,
                                             const XMLCh *tagName) {
  size_t tagCount = element->getElementsByTagName(tagName)->getLength();
  std::string tagNameStr = XMLString::transcode(tagName);
  //std::cout << std::endl << tagCount << std::endl;
  uassert(tagCount >= 1,
          "Cannot find tag " + tagNameStr + "!");
  uassert(tagCount == 1,
          "Multiple tags " + tagNameStr + "!");
  const auto *tag = element->getElementsByTagName(
    tagName)->item(0);
  const auto *tagFirstChild = tag->getFirstChild();
  uassert(tagFirstChild != nullptr,
          "Missing value inside " + tagNameStr + "tag!");

  std::string tagValue = XMLString::transcode(
    tagFirstChild->getNodeValue());

  return tagValue;
}

int IPXACTParser::getIntValueFromTag(const DOMElement *element,
                                     const XMLCh *tagName) {
  std::string tagValueStr = getStrValueFromTag(element, tagName);
  std::string tagNameStr = XMLString::transcode(tagName);
  uassert(stringIsDigit(tagValueStr), tagNameStr + "value must be an integer!");
  int tagValueInt = std::stoi(tagValueStr);
  return tagValueInt;
}

std::string IPXACTParser::getStrAttributeFromTag(const DOMElement *element,
                                                 const XMLCh *tagName,
                                                 const XMLCh *attributeName) {

  size_t tagCount = element->getElementsByTagName(tagName)->getLength();
  std::string tagNameStr = XMLString::transcode(tagName);
  uassert(tagCount >= 1,
          "Cannot find tag " + tagNameStr + "!");
  uassert(tagCount == 1,
          "Multiple tags " + tagNameStr + "!");
  const auto *tag = element->getElementsByTagName(tagName)->item(0);
  const auto *item = tag->getAttributes()->getNamedItem(attributeName);
  uassert(item != nullptr,
          "Missing attribute inside " + tagNameStr + "tag!");

  std::string attributeValue = std::string(XMLString::transcode(
    item->getNodeValue()));
  return attributeValue;
}

void IPXACTParser::parseCatalog(const std::string &libraryPath,
                                const std::string &catalogPath) {
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(
    XMLString::transcode("LS"));
  uassert(impl != nullptr, "DOMImplementation is not found!");

  // Create Xercecs Parser
  DOMLSParser *parser = ((DOMImplementationLS*)impl)->createLSParser(
    DOMImplementationLS::MODE_SYNCHRONOUS, 0);
  uassert(parser != nullptr, "Cannot create LSParser!");

  // Calculate path to IP-XACT catalog
  this->libraryPath = libraryPath;
  std::filesystem::path path = libraryPath;
  path /= catalogPath;

  // Open IP-XACT catalog
  const DOMDocument *doc = parser->parseURI(path.c_str());
  uassert(doc != nullptr, "Cannot parse IP-XACT catalog!");

  // The number of IP-XACT files to be read
  const size_t ipxactFileCount =
    doc->getElementsByTagName(ipxIpxFileTag)->getLength();

  // Read IP-XACT files one-by-one
  for (size_t i = 0; i < ipxactFileCount; i++) {
    const DOMElement *ipxactFile = (const DOMElement*)(
      doc->getElementsByTagName(ipxIpxFileTag)->item(i));

    // Parse ipxact:vlnv tag (to get attribute, not value)
    const std::string key = getStrAttributeFromTag(ipxactFile, ipxVlnvTag, nameAttr);

    // Parse ipxact:name tag
    const std::string name = getStrValueFromTag(ipxactFile, ipxNameTag);

    // Construct filesystem path to the component
    std::filesystem::path value = libraryPath;
    value /= name;

    // Bind component's name and its filesystem path
    readFileNames.insert({key, value.string()});
  }

  // TODO: remove or mask; only for debug
  for (const auto &[key, value] : readFileNames) {
    std::cout << key << std::endl << value << std::endl;
  }

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
  //For debuggin purposes.
  std::cout << name << std::endl;
  /*for (const auto &[key, value] : readFileNames) {
    std::cout << key << std::endl << value << std::endl;
  }*/
  //---------------------------------------------------------------------------
  //Creating parser.
  DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(
    XMLString::transcode("LS"));
  uassert(impl != nullptr, "DOMImplementation is not found!");
  DOMLSParser *parser = ((DOMImplementationLS*)impl)->createLSParser(
    DOMImplementationLS::MODE_SYNCHRONOUS, 0);
  uassert(parser != nullptr, "Cannot create LSParser!");
  //---------------------------------------------------------------------------
  //Parsing a document.
  const DOMDocument *doc = parser->parseURI(readFileNames[toLower(name)].c_str());
  uassert(doc != nullptr, "Cannot parse IP-XACT component!");
  //---------------------------------------------------------------------------
  //ipxact:port tag(s) parsing.
  size_t portCount = doc->getElementsByTagName(ipxPortTag)->getLength();
  for (size_t i = 0; i < portCount; i++) {
    const DOMElement *port = (const DOMElement*)(
      doc->getElementsByTagName(ipxPortTag)->item(i));
    //ipxact:name tag parsing.
    std::string name = getStrValueFromTag(port, ipxNameTag);
    //-------------------------------------------------------------------------
    //ipxact:direction tag parsing.
    std::string direction = getStrValueFromTag(port, ipxDirectTag);
    //-------------------------------------------------------------------------
    //ipxact:left tag parsing (if ipxact:vector tag is present).
    size_t vectorCount = port->getElementsByTagName(ipxVectTag)->getLength();
    std::string leftStr = vectorCount ? getStrValueFromTag(port, ipxLeftTag)
                                      : "";
    //-------------------------------------------------------------------------
    //Creating a port and adding it to the multitude of ports.
    int leftInt = -1;
    //Value can be a parameter.
    if (leftStr == "") {
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt + 1,
                                    model::Parameter(std::string("WIDTH"),
                                                     leftInt + 1)));
    } else if (isalpha(leftStr.c_str()[0])) {
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt + 1,
                                    model::Parameter(std::string(leftStr))));
    } else if (stringIsDigit(leftStr)) {
      leftInt = std::stoi(leftStr);
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt + 1,
                                    model::Parameter(std::string("WIDTH"),
                                                     leftInt + 1)));
    }
  }
  //---------------------------------------------------------------------------
  //ipxact:vendorExtensions tag(s) parsing.
  size_t parameterCount = doc->getElementsByTagName(
    k2ParamTag)->getLength();
  for (size_t i = 0; i < parameterCount; i++) {
    const DOMElement *parameter = (const DOMElement*)(
      doc->getElementsByTagName(k2ParamTag)->item(i));
    //kactus2:name tag parsing.
    std::string name = getStrValueFromTag(parameter, k2NameTag);
    //-------------------------------------------------------------------------
    //kactus2:value tag parsing.
    int value = getIntValueFromTag(parameter, k2ValueTag);
    //-------------------------------------------------------------------------
    //kactus2:left tag parsing.
    int left = getIntValueFromTag(parameter, k2LeftTag);
    //-------------------------------------------------------------------------
    //kactus2:right tag parsing.
    int right = getIntValueFromTag(parameter, k2RightTag);
    //-------------------------------------------------------------------------
    //Creating Parameter.
    params.add(model::Parameter(name,
                                model::Constraint<unsigned>(left, right),
                                value));
  }
  //---------------------------------------------------------------------------
  //ipxact:componentGenerators tag parsing (if present).
  std::shared_ptr<MetaElement> metaElement;
  size_t generatorTagCount = doc->getElementsByTagName(
    ipxCompGensTag)->getLength();
  if (generatorTagCount != 0) {
    //ipxact:generatorExe tag parsing.
    const DOMElement *ipxCompGens = (const DOMElement*)(
      doc->getElementsByTagName(ipxCompGensTag)->item(0));
    std::string path = getStrValueFromTag(ipxCompGens, ipxGenExeTag);
    //-------------------------------------------------------------------------
    //Creating metaElement.
    metaElement = std::shared_ptr<MetaElement>(new ElementGenerator(name,
                                                                    params,
                                                                    ports,
                                                                    path));
    //-------------------------------------------------------------------------
  } else {
    const auto *file = (const DOMElement*)(doc->getElementsByTagName(
      ipxFileTag)->item(0));
    //ipxact:name tag parsing.
    std::string name = getStrValueFromTag(file, ipxNameTag);
    //-------------------------------------------------------------------------
    //Constructing complete path.
    std::filesystem::path filesystemPath = libraryPath;
    //For debuggin purposes.
    //std::cout << libraryPath << std::endl;
    //-------------------------------------------------------------------------
    filesystemPath /= name;
    std::string path = filesystemPath.string();
    //For debuggin purposes.
    //std::cout << comPath << std::endl;
    //-------------------------------------------------------------------------
    //Creating metaElement.
    metaElement = std::shared_ptr<MetaElement>(new ElementCore(name,
                                                               params,
                                                               ports,
                                                               path));
    //-------------------------------------------------------------------------
  }
  //---------------------------------------------------------------------------
  //Releasing parser.
  parser->release();
  //---------------------------------------------------------------------------
  //Returns created element.
  return metaElement;
  //---------------------------------------------------------------------------
}

} // namespace eda::hls::library
