//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "base/model/hash.h"
#include "base/model/link.h"
#include "base/model/signal.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace eda::base::model {

/**
 * \brief Represents a net node (a gate or a higher-level unit).
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
template <typename Func, bool StructHash = true>
class Node {
public:
  //===--------------------------------------------------------------------===//
  // Constants and Types
  //===--------------------------------------------------------------------===//

  /// Node identifier.
  using Id = uint32_t;
  static constexpr Id INVALID = -1u;

  using List = std::vector<Node<Func, StructHash>*>;
  using Link = eda::base::model::Link<Id>;
  using LinkList = Link::List;
  using Signal = eda::base::model::Signal<Id>;
  using SignalList = Signal::List;
  using StructHashKey = NodeHashKey<Func, Id>;
  using StructHashMap = std::unordered_map<StructHashKey, Id>;

  //===--------------------------------------------------------------------===//
  // Accessor
  //===--------------------------------------------------------------------===//

  /// Returns the node w/ the given id from the storage.
  static Node<Func, StructHash> *get(Id id) { return _storage[id]; }
  /// Returns the next node identifier.
  static Id nextId() { return _storage.size(); }

  /// Returns the node w/ the given function/inputs from the storage.
  static Node<Func, StructHash> *get(
      uint32_t netId, Func func, const SignalList &inputs);
  /// Saves the node w/ the given function/inputs to the hash table.
  static void add(
      uint32_t netId, Node<Func, StructHash> *node);

  //===--------------------------------------------------------------------===//
  // Properties
  //===--------------------------------------------------------------------===//

  Id id() const { return _id; }
  Func func() const { return _func; }

  bool hasSignature(Func func, const SignalList &inputs) {
    if (func != _func || inputs.size() != _inputs.size()) {
      return false;
    }

    std::unordered_set<Id> inputSet;
    for (size_t i = 0; i < inputs.size(); i++) {
      inputSet.insert(inputs[i].node());
      inputSet.insert(_inputs[i].node());
    }

    return inputSet.size() == inputs.size();
  }

  //===--------------------------------------------------------------------===//
  // Connections
  //===--------------------------------------------------------------------===//

  size_t arity() const { return _inputs.size(); }
  const SignalList &inputs() const { return _inputs; }
  const Signal &input(size_t i) const { return _inputs[i]; }

  size_t fanout() const { return _links.size(); }
  const LinkList &links() const { return _links; }
  const Link &link(size_t i) const { return _links[i]; }

  //===--------------------------------------------------------------------===//
  // Signal Wrappers
  //===--------------------------------------------------------------------===//

  Signal posedge() const { return Signal::posedge(_id); }
  Signal negedge() const { return Signal::negedge(_id); }
  Signal level0()  const { return Signal::level0(_id); }
  Signal level1()  const { return Signal::level1(_id); }
  Signal always()  const { return Signal::always(_id); }

protected:
  /// Creates a node w/ the given function/inputs and
  /// allocates this node in the storage.
  Node(Func func, const SignalList &inputs):
    _id(_storage.size()), _func(func), _inputs(inputs) {
    // Register the node in the storage.
    if (_id >= _storage.size()) {
      _storage.resize(_id + 1);
      _storage[_id] = this;
    }
    appendLinks();
  }

  /// Creates a node w/ the given function/inputs and
  /// stores this node in the existing position.
  Node(Id id, Func func, const SignalList &inputs, const LinkList &links):
    _id(id), _func(func), _inputs(inputs), _links(links) {
    assert(_id < _storage.size());
    _storage[_id] = this;
    appendLinks();
  }

  void setFunc(Func func) {
    _func = func;
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
      auto *node = Node<Func, StructHash>::get(_inputs[i].node());
      node->appendLink(_id, i);
    }
  }

  void removeLinks() {
    for (size_t i = 0; i < _inputs.size(); i++) {
      auto *node = Node<Func, StructHash>::get(_inputs[i].node());
      node->removeLink(_id, i);
    }
  }

  const Id _id;
  Func _func;
  SignalList _inputs;
  LinkList _links;

  /// Node storage.
  static List _storage;
  /// Structural hashing.
  static StructHashMap _hashing;
};

template <typename Func, bool StructHash>
Node<Func, StructHash> *Node<Func, StructHash>::get(
    uint32_t netId, Func func, const SignalList &inputs) {
  // Structural hashing is disabled.
  if constexpr(!StructHash) {
    return nullptr;
  }

  // Source nodes look identical, but they are not.
  if (!func.isConstant() && inputs.empty()) {
    return nullptr;
  }

  // Ignore identity nodes.
  if (func.isIdentity()) {
    assert(inputs.size() == 1);
    return get(inputs[0].node());
  }

  // Search for the same node.
  StructHashKey key(netId, func, inputs);
  auto i = _hashing.find(key);

  // If the same node exists, return it.
  if (i != _hashing.end()) {
    auto *node = get(i->second);
    if (node->hasSignature(func, inputs)) {
      return node;
    }
  }

  return nullptr;
}

template <typename Func, bool StructHash>
void Node<Func, StructHash>::add(uint32_t netId, Node<Func, StructHash> *node) {
  // Structural hashing is disabled.
  if constexpr(!StructHash) {
    return;
  }

  StructHashKey key(netId, node->func(), node->inputs());
  _hashing.insert({key, node->id()});
}

template <typename Func, bool StructHash>
typename Node<Func, StructHash>::List Node<Func, StructHash>::_storage = []{
  Node<Func, StructHash>::List storage;
  storage.reserve(1024*1024);
  return storage;
}();

template <typename Func, bool StructHash>
typename Node<Func, StructHash>::StructHashMap Node<Func, StructHash>::_hashing = []{
  Node<Func, StructHash>::StructHashMap hashing;
  hashing.reserve(1024*1024);
  return hashing;
}();

} // namespace eda::base::model
