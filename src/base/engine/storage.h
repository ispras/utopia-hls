//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>

namespace eda::base::engine {

/**
 * \brief Implements a container that store arbitrary objects.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Storage final {
public:
  Storage() = default;

  Storage(const Storage&) = delete;
  Storage &operator=(const Storage&) = delete;

  /// Stores an object and transfers the ownership.
  template<typename T>
  void add(const std::string &name, T &&object);

  /// Gets access to the stored object.
  template<typename T>
  const T &get(const std::string &name) const;

  /// Checks whether there exists an object w/ the given name.
  bool exists(const std::string &name) const;

private:
  /// Type-erased value holder for move-constructible types.
  struct IHolder {
    virtual ~IHolder() = default;
    virtual const std::type_info &type() const = 0;
  };

  template<typename T,
           typename =
               std::enable_if_t<std::is_nothrow_move_constructible<T>::value>>
  struct Holder : public IHolder {
    Holder(T &&v): value(std::move(v)) {}
    const std::type_info &type() const { return typeid(T); }

    T value;
  };

  std::unordered_map<std::string, std::unique_ptr<IHolder>> _store;
};

template<typename T>
inline void Storage::add(const std::string &name, T &&object) {
  assert(_store.find(name) == _store.end());
  _store.emplace(name, std::make_unique<Holder<T>>(std::forward<T>(object)));
}

template<typename T>
inline const T &Storage::get(const std::string &name) const {
  auto i = _store.find(name);
  assert(i != _store.end());

  const IHolder *holder = i->second.get();
  const auto *casted = dynamic_cast<const Holder<T>*>(holder);
  assert(casted != nullptr);

  return casted->value;
}

inline bool Storage::exists(const std::string &name) const {
  return _store.find(name) != _store.end();
}

} // namespace eda::base::engine


