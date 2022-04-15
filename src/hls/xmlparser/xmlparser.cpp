//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/DOMLSParserImpl.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <ctemplate/template.h>

#include <iostream>

#include "hls/xmlparser/xmlparser.h"

using namespace xercesc;
using namespace std;

namespace eda::hls::xmlparser {
  void XMLParser::parseIPXACT() {
    try {
      XMLPlatformUtils::Initialize();
    }
    catch(XMLException& e) {
      char* message = XMLString::transcode( e.getMessage() );
      cout << "XML toolkit initialization error: " << message << endl;
      XMLString::release( &message );
    }
    parseCatalog();
    std::cout << comp_fnames.size() << std::endl;
    for (size_t i = 0; i < 1; ++i) {
      std::cout << "???" << std::endl;
      parseComponent(i);
    }
    XMLPlatformUtils::Terminate();
  }

  void XMLParser::parseCatalog() {
    XMLCh tempStr[100];
    XMLString::transcode("LS", tempStr, 99);
    DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(tempStr);
    DOMLSParser* parser = ((DOMImplementationLS*)impl)->createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0);
    xercesc::DOMDocument *doc = 0;
    doc = parser->parseURI(catalog_fname);
    XMLCh *ipxactFile_tag = XMLString::transcode("ipxact:ipxactFile");
    XMLCh *vlnv_tag = XMLString::transcode("ipxact:vlnv");
    XMLCh *name_tag = XMLString::transcode("ipxact:name");
    XMLCh *name_attr = XMLString::transcode("name");
    //std::vector<const XMLCh*> comp_fnames;
    for (size_t i = 0; i < doc->getElementsByTagName(ipxactFile_tag)->getLength(); ++i) {
      const DOMElement* ipxactFile = (const DOMElement*)(doc->getElementsByTagName(ipxactFile_tag)->item(i));
      /*if (ipxactFile == NULL) {
        std::cout << "???" << std::endl;
      }*/
      std::cout << XMLString::transcode(ipxactFile->getNodeName()) << std::endl;
      const auto* vlnv = ipxactFile->getElementsByTagName(vlnv_tag)->item(0);
      std::cout << XMLString::transcode(vlnv->getAttributes()->getNamedItem(name_attr)->getNodeValue()) << std::endl;
      const auto* name = ipxactFile->getElementsByTagName(name_tag)->item(0);
      //const auto* fname = new XMLCh[sizeof(name->getFirstChild()->getNodeValue())/sizeof(XMLCh)];
      //memcpy((void *)fname, (void *)name->getFirstChild()->getNodeValue(), sizeof(name->getFirstChild()->getNodeValue()));
      comp_fnames.push_back(XMLString::transcode(name->getFirstChild()->getNodeValue()));
      //std::cout << comp_fnames.size() << std::endl;
      std::cout << comp_fnames.back() << std::endl;
    }
    //std::cout << comp_fnames.size() << std::endl;
    //doc->release();
    delete ipxactFile_tag;
    delete vlnv_tag;
    delete name_tag;
    delete name_attr;
    parser->release();
    return;
  }

  void XMLParser::parseComponent(int index) {
    XMLCh tempStr[100];
    XMLString::transcode("LS", tempStr, 99);
    DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(tempStr);
    DOMLSParser* parser = ((DOMImplementationLS*)impl)->createLSParser(DOMImplementationLS::MODE_SYNCHRONOUS, 0);
    xercesc::DOMDocument *doc = 0;
    try {
      doc = parser->parseURI(comp_fnames[index]);
    } catch(...) {
      std::cout << "Exception!" << std::endl;
      return;
    }
    std::cout <<comp_fnames[index] << std::endl;

    XMLCh *port_tag = XMLString::transcode("ipxact:port");
    XMLCh *name_tag = XMLString::transcode("ipxact:name");
    XMLCh *vlnv_tag = XMLString::transcode("ipxact:vlnv");
    XMLCh *name_attr = XMLString::transcode("name");
    for (size_t i = 0; i < doc->getElementsByTagName(port_tag)->getLength(); ++i) {
      const DOMElement* port = (const DOMElement*)(doc->getElementsByTagName(port_tag)->item(i));
      const auto* name = port->getElementsByTagName(name_tag)->item(0);
      std::cout << "Port name: " << XMLString::transcode(name->getFirstChild()->getNodeValue()) << std::endl;
      /*std::cout << XMLString::transcode(ipxactFile->getNodeName()) << std::endl;
      const auto* vlnv = ipxactFile->getElementsByTagName(vlnv_tag)->item(0);
      std::cout << XMLString::transcode(vlnv->getAttributes()->getNamedItem(name_attr)->getNodeValue()) << std::endl;
      const auto* name = ipxactFile->getElementsByTagName(name_tag)->item(0);
      std::cout << XMLString::transcode(name->getFirstChild()->getNodeValue()) << std::endl;*/
    }
    //doc->release();
    parser->release();
    return;
  }

  XMLParser::XMLParser(const char* catalog_fname) {
    this->catalog_fname = catalog_fname;
  }

  XMLParser::~XMLParser() {
    //delete catalog_fname;
  }
} // namespace eda::hls::xmlparser





      /*XercesDOMParser* parser = new XercesDOMParser();
      parser->setValidationScheme(XercesDOMParser::Val_Always);
      parser->setDoNamespaces(true);
      ErrorHandler* errHandler = (ErrorHandler*) new HandlerBase();
      parser->setErrorHandler(errHandler);

      char* xmlFile = "~/utopia/src/hls/library/IP-XACT/ispras/ip.hw/library_catalog/1.0/library_catalog.xml";
      try {
        parser->parseURI(xmlFile);
      }
      catch (const XMLException& toCatch) {
        char* message = XMLString::transcode(toCatch.getMessage());
        std::cout << "Exception message is: \n"
               << message << "\n";
        XMLString::release(&message);
        return;
      }
      catch (const DOMException& toCatch) {
        char* message = XMLString::transcode(toCatch.msg);
        std::cout << "Exception message is: \n"
               << message << "\n";
        XMLString::release(&message);
        return;
      }*/
      /*catch (...) {
        std::cout << "Unexpected Exception \n" ;
        return;
      }*/
      /*std::cout << "?" << std::endl;
      DOMDocument *doc = parser->getDocument();
      if (doc == NULL) {
        std::cout << "Error in reading the document!**********************" << std::endl;
        return;
      }
      std::cout << "?" << std::endl;
      //XMLCh *component_tag = XMLString::transcode("component");
      XMLCh *ipxactFile_tag = XMLString::transcode("ipxactFile");
      XMLCh *vlnv_tag = XMLString::transcode("vlnv");
      XMLCh *name = XMLString::transcode("name");
	    //const auto *component_doc = doc->getElementsByTagName(name);
	    for (size_t i = 0; i < doc->getElementsByTagName(ipxactFile_tag)->getLength(); ++i) {
        const auto* ipxactFile = doc->getElementsByTagName(ipxactFile_tag)->item(i);
		    const auto* vlnv = ipxactFile->getOwnerDocument()->getElementsByTagName(vlnv_tag)->item(0);
		    std::cout << vlnv->getAttributes()->getNamedItem(name) << std::endl;
	    }
      doc->release();
      delete parser;
      XMLPlatformUtils::Terminate();
    }*/
