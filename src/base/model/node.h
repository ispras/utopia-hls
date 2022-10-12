//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "base/model/link.h"
#include "base/model/signal.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace eda::base::model {

/**
 * \brief Represents a net node (a gate or a higher-level unit).
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
template <typename F>
class Node {
public:
  //===--------------------------------------------------------------------===//
  // Types
  //===--------------------------------------------------------------------===//

  using Id = unsigned;
  using List = std::vector<Node<F>*>;
  using Link = eda::base::model::Link<Id>;
  using LinkList = Link::List;
  using Signal = eda::base::model::Signal<Id>;
  using SignalList = Signal::List;

  //===--------------------------------------------------------------------===//
  // Constants
  //===--------------------------------------------------------------------===//

  static constexpr Id INVALID = -1u;

  //===--------------------------------------------------------------------===//
  // Accessor
  //===--------------------------------------------------------------------===//

  /// Returns the node w/ the given id from the storage.
  static Node<F> *get(Id id) { return _storage[id]; }
  /// Returns the next node identifier.
  static Id nextId() { return _storage.size(); }

  //===--------------------------------------------------------------------===//
  // Properties
  //===--------------------------------------------------------------------===//

  Id id() const { return _id; }
  F kind() const { return _kind; }
/*
  bool isSource() const {
    return _kind == GateSymbol::NOP && _inputs.empty();
  }

  bool isValue() const {
    return _kind == GateSymbol::ONE || _kind == GateSymbol::ZERO;
  }

  bool isTrigger() const {
    for (const auto &input: _inputs) {
      if (!input.isAlways())
        return true;
    }
    return false;
  }

  bool isComb() const {
    return !isSource() && !isTrigger();
  }
*/

  //===--------------------------------------------------------------------===//
  // Connections
  //===--------------------------------------------------------------------===//

  size_t arity() const { return _inputs.size(); }
  const SignalList &inputs() const { return _inputs; }
  const Signal &input(size_t i) const { return _inputs[i]; }

  size_t fanout() const { return _links.size(); }
  const LinkList &links() const { return _links; }
  const Link &link(size_t i) const { return _links[i]; }

protected:
  /// Creates a node w/ the given function and the inputs.
  Node(F kind, const SignalList inputs):
    _id(_storage.size()), _kind(kind), _inputs(inputs) {
    // Register the node in the storage.
    if (_id >= _storage.size()) {
      _storage.resize(_id + 1);
      _storage[_id] = this;
    }
    appendLinks();
  }

  /// Creates a source node.
  // FIXME: Gate(): Gate(GateSymbol::NOP, {}) {}

  void setKind(F kind) {
    _kind = kind;
  }

  void setInputs(const SignalList &inputs) {
    removeLinks();
    _inputs.assign(inputs.begin(), inputs.end());
    appendLinks();
  }

  void appendLink(Id to, size_t i) {
    Link link(_id, to, i);
    _links.push_back(link);
  }

  void removeLink(Id to, size_t input) {
    Link link(_id, to, input);
    auto i = std::remove(_links.begin(), _links.end(), link);
    _links.erase(i, _links.end());
  }

  void appendLinks() {
    for (size_t i = 0; i < _inputs.size(); i++) {
      auto *node = Node<F>::get(_inputs[i].node());
      node->appendLink(_id, i);
    }
  }

  void removeLinks() {
    for (size_t i = 0; i < _inputs.size(); i++) {
      auto *node = Node<F>::get(_inputs[i].node());
      node->removeLink(_id, i);
    }
  }

  const Id _id;
  F _kind;
  SignalList _inputs;
  LinkList _links;

  /// Node storage.
  static List _storage;
};

template <typename F>
typename Node<F>::List Node<F>::_storage = []{
  Node<F>::List storage;
  storage.reserve(1024*1024);
  return storage;
}();

} // namespace eda::base::model
