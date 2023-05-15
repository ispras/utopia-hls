//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/ipxact_parser.h"

#include "hls/library/element_core.h"
#include "hls/library/element_generator.h"
#include "util/string.h"

#include <filesystem>
#include <regex>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>

namespace eda::hls::library {

bool stringIsInteger(std::string &inputString) {
  return std::regex_match(inputString, std::regex("[(-|\\+)]?[0-9]+"));
}

bool stringIsId(std::string &inputString) {
  return std::regex_match(inputString, std::regex("[A-Za-z][A-Za-z0-9]*"));
}

bool stringIsBool(std::string &inputString) {
  return (inputString == "true" || inputString == "false");
}

bool stringToBool(std::string &inputString) {
  return inputString == "true" ? true : false;
}

std::vector<std::shared_ptr<MetaElement>> IPXACTParser::getDelivery(
    const std::string &libraryPath, const std::string &catalogPath) {
  const auto &fileNames = parseCatalog(libraryPath, catalogPath);
  const auto &metaElements = parseComponents(libraryPath, fileNames);
  return metaElements;
}

bool IPXACTParser::tryToParseXML(const char* fileName) {
  // Check wheter the file exists.
  if (!std::filesystem::exists(fileName)) {
    std::cout << "File" << fileName << " not found!" << std::endl;
    return false;
  }
  // Trying to parse the file.
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
                                             const XMLCh *tagName) {
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
                                     const XMLCh *tagName) {
  std::string tagValueStr = getStrValueFromTag(element, tagName);
  std::string tagNameStr = XMLString::transcode(tagName);
  uassert(stringIsInteger(tagValueStr), tagNameStr + 
          " value must be an integer!\n");
  int tagValueInt = std::stoi(tagValueStr);
  return tagValueInt;
}

bool IPXACTParser::getBoolValueFromTag(const DOMElement *element,
                                       const XMLCh *tagName) {
  std::string tagValueStr = getStrValueFromTag(element, tagName);
  std::string tagNameStr = XMLString::transcode(tagName);
  uassert(stringIsBool(tagValueStr), tagNameStr + 
          " value must be bool!\n");
  bool tagValueBool = stringToBool(tagValueStr);
  return tagValueBool;
}


std::string IPXACTParser::getStrAttrFromTag(const DOMElement *element,
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
  // Calculate path to IP-XACT catalog.
  std::filesystem::path path = libraryPath;
  path /= catalogPath;

  // Trying to parse IP-XACT catalog.
  bool isParsed = tryToParseXML(path.c_str());
  uassert(isParsed, "Error while parsing an IP-XACT catalog!\n");

  // Get DOM document representing IP-XACT catalog.
  const DOMDocument *doc = parser->getDocument();
  uassert(doc != nullptr, "Cannot parse IP-XACT catalog!\n");

  // Get number of IP-XACT files to be read.
  const size_t ipxactFileCount =
    doc->getElementsByTagName(ipxIpxFileTag)->getLength();

  // Read IP-XACT files one-by-one.
  std::vector<std::string> fileNames;
  for (size_t i = 0; i < ipxactFileCount; i++) {
    const DOMElement *ipxactFile = (const DOMElement*)(
      doc->getElementsByTagName(ipxIpxFileTag)->item(i));

    // Parse tags.
    const std::string name = getStrValueFromTag(ipxactFile, ipxNameTag);

    // Bind component's name and it's path relative to IP-XACT library path.
    fileNames.push_back(name);
  }
  // Reset parser for further use.
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

  // Calculate path to IP-XACT component.
  std::filesystem::path path = libraryPath;
  path /= componentPath;

  // Trying to parse IP-XACT component.
  bool isParsed = tryToParseXML(path.c_str());
  uassert(isParsed, "Error while parsing an IP-XACT component!\n");

  // Get DOM document representing IP-XACT component.
  const DOMDocument *doc = parser->getDocument();
  uassert(doc != nullptr, "Cannot parse IP-XACT component!\n");

  // Parse tags.
  const DOMElement *comp = (const DOMElement*)(
      doc->getElementsByTagName(ipxCompTag)->item(0));
  const std::string name = getStrValueFromTag(comp, ipxModuleNameTag);
  const std::string libraryName = getStrValueFromTag(comp, ipxLibTag);

  // Parse ports.
  size_t portCount = doc->getElementsByTagName(ipxPortTag)->getLength();
  for (size_t i = 0; i < portCount; i++) {
    const DOMElement *port = (const DOMElement*)(
      doc->getElementsByTagName(ipxPortTag)->item(i));
  
    // Parse tags.
    std::string name = getStrValueFromTag(port, ipxNameTag);
    std::string direction = getStrValueFromTag(port, ipxDirectTag);
    // Default port type is DATA.
    Port::Type type = Port::Type::DATA;

    uassert(direction == "in" || direction == "out" || direction == "inout",
        "Direction of the port " + name + "must be 'in', 'out' or 'inout'!\n");

    size_t vectorCount = port->getElementsByTagName(ipxVectTag)->getLength();
    std::string leftStr = vectorCount ? getStrValueFromTag(port, ipxLeftTag)
                                      : "";
    
    // Port can be of special type: clock or reset.
    size_t qualifierTagCount = port->getElementsByTagName(
      ipxQualifierTag)->getLength();
    if (qualifierTagCount != 0) {
      size_t ipxIsDataTagCount = port->getElementsByTagName(
          ipxIsDataTag)->getLength();
      size_t ipxIsClockTagCount = port->getElementsByTagName(
          ipxIsClockTag)->getLength();
      size_t ipxIsResetTagCount = port->getElementsByTagName(
          ipxIsResetTag)->getLength();
      if (ipxIsDataTagCount  == 1 &&
          ipxIsClockTagCount == 0 &&
          ipxIsResetTagCount == 0) {
        bool isDataType = getBoolValueFromTag(port, ipxIsDataTag);
        uassert(isDataType, "Value of " + 
            std::string(XMLString::transcode(ipxIsDataTag)) + 
            "must be 'true'!\n");
        type = Port::Type::DATA;
      } else if (ipxIsDataTagCount  == 0 &&
                 ipxIsClockTagCount == 1 &&
                 ipxIsResetTagCount == 0) {
        bool isClockType = getBoolValueFromTag(port, ipxIsClockTag);
        uassert(isClockType, "Value of " + 
            std::string(XMLString::transcode(ipxIsClockTag)) + 
            "must be 'true'!\n");
        type = Port::Type::CLOCK;
      } else if (ipxIsDataTagCount  == 0 &&
                 ipxIsClockTagCount == 0 &&
                 ipxIsResetTagCount == 1) {
        bool isResetType = getBoolValueFromTag(port, ipxIsClockTag);
        uassert(isResetType, "Value of " + 
            std::string(XMLString::transcode(ipxIsDataTag)) + 
            "must be 'true'!\n");
        type = Port::Type::RESET;
      }
    }      
    // Create port and adding it to multitude of ports.
    int leftInt = 0;
    if (leftStr == "") {
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt + 1,
                                    model::Parameter(std::string("WIDTH"),
                                                     leftInt + 1),
                                    type));
    } else if (stringIsId(leftStr)) {
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt,
                                    model::Parameter(std::string(leftStr)),
                                    type));
    } else if (stringIsInteger(leftStr)) {
      leftInt = std::stoi(leftStr);
      uassert(leftInt >= 0, "Port width margin value cannot be negative!\n");
      ports.push_back(library::Port(name,
                                    direction == "in" ? Port::IN : Port::OUT,
                                    leftInt + 1,
                                    model::Parameter(std::string("WIDTH"),
                                                     leftInt + 1),
                                    type));
    }
  }

  // Parse vendor extensions.
  size_t parameterCount = doc->getElementsByTagName(k2ParamTag)->getLength();
  for (size_t i = 0; i < parameterCount; i++) {
    const DOMElement *parameter = (const DOMElement*)(
      doc->getElementsByTagName(k2ParamTag)->item(i));
    // Parse tags.
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

  // Check if component is generator.
  std::shared_ptr<MetaElement> metaElement;
  size_t generatorTagCount = doc->getElementsByTagName(
      ipxCompGensTag)->getLength();
  if (generatorTagCount != 0) {
    // Parse tag.
    const DOMElement *ipxCompGens = (const DOMElement*)(
      doc->getElementsByTagName(ipxCompGensTag)->item(0));
    
    std::string path = std::string(std::getenv("UTOPIA_HLS_HOME"));
    std::string genPath = getStrValueFromTag(ipxCompGens, ipxGenExeTag);
    path = path + "/" + genPath;

    // Create metaElement.
    metaElement = std::shared_ptr<MetaElement>(new ElementGenerator(name,
                                                                    libraryName,
                                                                    false,
                                                                    params,
                                                                    ports,
                                                                    path));

  } else {
    const auto *file = (const DOMElement*)(doc->getElementsByTagName(
      ipxFileTag)->item(0));
    // Parse tag.
    std::string filePath = getStrValueFromTag(file, ipxNameTag);

    // Construct complete path.
    std::filesystem::path filesystemPath = libraryPath;

    filesystemPath /= filePath;
    std::string path = filesystemPath.string();

    // Create metaElement.
    metaElement = std::shared_ptr<MetaElement>(new ElementCore(name,
                                                               libraryName,
                                                               false,
                                                               params,
                                                               ports,
                                                               path));
  }
  // Reset parser for further use.
  parser->reset();
  
  return metaElement;
}

} // namespace eda::hls::library