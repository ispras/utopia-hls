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
#include "util/string.h"

#include <filesystem>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>

using namespace eda::hls::mapper;
using namespace eda::utils;
using namespace xercesc;

namespace eda::hls::library {

bool stringIsInteger(std::string str) {
  size_t startIndex = (str.at(0) == '-' || str.at(0) == '+') ? 1 : 0;
  for (size_t i = startIndex; i < str.length(); i++) {
    if (!isdigit(str.at(i))) {
      return false;
    }
  }
  return true;
}

std::vector<std::shared_ptr<MetaElement>> IPXACTParser::getDelivery(
    const std::string &libraryPath, const std::string &catalogPath) {

  const auto &fileNames = parseCatalog(libraryPath, catalogPath);
  const auto &metaElements = parseComponents(libraryPath, fileNames);
  return metaElements;
}

bool IPXACTParser::tryToParseXML(const char* fileName) {
  try {
    parser->parse(fileName);
  } 
  catch (const XMLException &xmlException) {
      char* message = XMLString::transcode(xmlException.getMessage());
      std::cout << "Exception message is: \n"
                << message << "\n";
      XMLString::release(&message);
      return false;
  }
  catch (const DOMException &domException) {
    char* message = XMLString::transcode(domException.getMessage());
    std::cout << "Exception message is: \n"
              << message << "\n";
    XMLString::release(&message);
    return false;
  }
  catch (const SAXException &saxException) {
    char* message = XMLString::transcode(saxException.getMessage());
    std::cout << "Exception message is: \n"
              << message << "\n";
    XMLString::release(&message);
    return false;
  }
  catch (...) {
    std::cout << "Unexpected Exception \n" ;
    return false;   
  }
  return true;
}

std::string IPXACTParser::getStrValueFromTag(const DOMElement *element,
                                             const XMLCh      *tagName) {

  size_t tagCount = element->getElementsByTagName(tagName)->getLength();
  std::string tagNameStr = XMLString::transcode(tagName);
  uassert(tagCount >= 1, "Cannot find tag " + tagNameStr + "!\n");
  const auto *tag = element->getElementsByTagName(tagName)->item(0);
  const auto *tagFirstChild = tag->getFirstChild();
  uassert(tagFirstChild != nullptr,
          "Missing value inside " + tagNameStr + "tag!\n");
  std::string tagValue = XMLString::transcode(
    tagFirstChild->getNodeValue());
  return tagValue;
}

int IPXACTParser::getIntValueFromTag(const DOMElement *element,
                                     const XMLCh      *tagName) {

  std::string tagValueStr = getStrValueFromTag(element, tagName);
  std::string tagNameStr = XMLString::transcode(tagName);
  uassert(stringIsInteger(tagValueStr), tagNameStr + 
          " value must be an integer!\n");
  int tagValueInt = std::stoi(tagValueStr);
  return tagValueInt;
}

std::string IPXACTParser::getStrAttributeFromTag(const DOMElement *element,
                                                 const XMLCh *tagName,
                                                 const XMLCh *attributeName) {

  size_t tagCount = element->getElementsByTagName(tagName)->getLength();
  std::string tagNameStr = XMLString::transcode(tagName);
  uassert(tagCount >= 1, "Cannot find tag " + tagNameStr + "!\n");
  uassert(tagCount == 1, "Multiple tags " + tagNameStr + "!\n");
  const auto *tag = element->getElementsByTagName(tagName)->item(0);
  const auto *item = tag->getAttributes()->getNamedItem(attributeName);
  uassert(item != nullptr, "Missing attribute inside " + tagNameStr + "tag!\n");
  std::string attributeValue = std::string(XMLString::transcode(
    item->getNodeValue()));
  return attributeValue;
}

std::vector<std::string> IPXACTParser::parseCatalog(
    const std::string &libraryPath,
    const std::string &catalogPath) {

  // Calculate path to IP-XACT catalog
  std::filesystem::path path = libraryPath;
  path /= catalogPath;

  // Trying to parse IP-XACT catalog
  bool isParsed = tryToParseXML(path.c_str());
  uassert(isParsed, "Error while parsing an IP-XACT catalog!\n");

  // Get DOM document representing IP-XACT catalog
  const DOMDocument *doc = parser->getDocument();
  uassert(doc != nullptr, "Cannot parse IP-XACT catalog!\n");

  // Get number of IP-XACT files to be read
  const size_t ipxactFileCount =
    doc->getElementsByTagName(ipxIpxFileTag)->getLength();

  // Read IP-XACT files one-by-one
  std::vector<std::string> fileNames;
  for (size_t i = 0; i < ipxactFileCount; i++) {
    const DOMElement *ipxactFile = (const DOMElement*)(
      doc->getElementsByTagName(ipxIpxFileTag)->item(i));

    // Parse tags
    const std::string name = getStrValueFromTag(ipxactFile, ipxNameTag);

    // Bind component's name and it's path relative to IP-XACT library path
    fileNames.push_back(name);
  }
  // Reset parser for further use
  parser->reset();

  return fileNames;
}

std::vector<std::shared_ptr<MetaElement>> IPXACTParser::parseComponents(
    const std::string              &libraryPath,
    const std::vector<std::string> &componentPaths) {

  std::vector<std::shared_ptr<MetaElement>> metaElements;
  for (const auto &componentPath : componentPaths) {
    metaElements.push_back(parseComponent(libraryPath, componentPath));
  }
  return metaElements;
}

std::shared_ptr<MetaElement> IPXACTParser::parseComponent(
    const std::string &libraryPath,
    const std::string &componentPath) {

  Parameters params;
  std::vector<Port> ports;

  // Calculate path to IP-XACT component
  std::filesystem::path path = libraryPath;
  path /= componentPath;

  // Trying to parse IP-XACT component
  bool isParsed = tryToParseXML(path.c_str());
  uassert(isParsed, "Error while parsing an IP-XACT component!\n");

  // Get DOM document representing IP-XACT component
  const DOMDocument *doc = parser->getDocument();
  uassert(doc != nullptr, "Cannot parse IP-XACT component!\n");

  // Parse tags
  const DOMElement *comp = (const DOMElement*)(
      doc->getElementsByTagName(ipxCompTag)->item(0));
  const std::string name = getStrValueFromTag(comp, ipxModuleNameTag);
  const std::string library = getStrValueFromTag(comp, ipxLibTag);

  // Parse ports
  size_t portCount = doc->getElementsByTagName(ipxPortTag)->getLength();
  for (size_t i = 0; i < portCount; i++) {
    const DOMElement *port = (const DOMElement*)(
      doc->getElementsByTagName(ipxPortTag)->item(i));
  
    // Parse tags
    std::string name = getStrValueFromTag(port, ipxNameTag);
    std::string direction = getStrValueFromTag(port, ipxDirectTag);

    // TODO: This is a temporal solution. Any HW language may be used
    uassert(direction == "in" || direction == "out" || direction == "inout",
        "Direction of the port " + name + "must be 'in', 'out' or 'inout'!\n");

    size_t vectorCount = port->getElementsByTagName(ipxVectTag)->getLength();
    std::string leftStr = vectorCount ? getStrValueFromTag(port, ipxLeftTag)
                                      : "";

    // Create port and adding it to multitude of ports
    int leftInt = 0;
    if (leftStr == "") {
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt + 1,
                                    model::Parameter(std::string("WIDTH"),
                                                     leftInt + 1)));
    } else if (isalpha(leftStr.c_str()[0])) {
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt,
                                    model::Parameter(std::string(leftStr))));
    } else if (stringIsInteger(leftStr)) {
      leftInt = std::stoi(leftStr);
      uassert(leftInt >= 0, "Port width margin value cannot be negative!\n");
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt + 1,
                                    model::Parameter(std::string("WIDTH"),
                                                     leftInt + 1)));
    }
  }

  // Parse vendor extensions
  size_t parameterCount = doc->getElementsByTagName(
    k2ParamTag)->getLength();
  for (size_t i = 0; i < parameterCount; i++) {
    const DOMElement *parameter = (const DOMElement*)(
      doc->getElementsByTagName(k2ParamTag)->item(i));
    // Parse tags
    std::string name = getStrValueFromTag(parameter, k2NameTag);

    int value = getIntValueFromTag(parameter, k2ValueTag);

    int left = getIntValueFromTag(parameter, k2LeftTag);

    int right = getIntValueFromTag(parameter, k2RightTag);

    uassert(left <= right, "Incorrect margins!\n");

    // Creating Parameter
    params.add(model::Parameter(name,
                                model::Constraint<unsigned>(left, right),
                                value));
  }

  // Check if component is generator
  std::shared_ptr<MetaElement> metaElement;
  size_t generatorTagCount = doc->getElementsByTagName(
    ipxCompGensTag)->getLength();
  if (generatorTagCount != 0) {
    // Parse tag
    const DOMElement *ipxCompGens = (const DOMElement*)(
      doc->getElementsByTagName(ipxCompGensTag)->item(0));
    
    std::string path = std::string(std::getenv("UTOPIA_HOME"));
    std::string genPath = getStrValueFromTag(ipxCompGens, ipxGenExeTag);
    path = path + "/" + genPath;

    // Create metaElement
    metaElement = std::shared_ptr<MetaElement>(new ElementGenerator(name,
                                                                    library,
                                                                    params,
                                                                    ports,
                                                                    path));

  } else {
    const auto *file = (const DOMElement*)(doc->getElementsByTagName(
      ipxFileTag)->item(0));
    // Parse tag
    std::string filePath = getStrValueFromTag(file, ipxNameTag);

    // Construct complete path
    std::filesystem::path filesystemPath = libraryPath;

    filesystemPath /= filePath;
    std::string path = filesystemPath.string();

    // Create metaElement
    metaElement = std::shared_ptr<MetaElement>(new ElementCore(name,
                                                               library,
                                                               params,
                                                               ports,
                                                               path));
  }
  // Reset parser for further use
  parser->reset();
  
  return metaElement;
}

} // namespace eda::hls::library
