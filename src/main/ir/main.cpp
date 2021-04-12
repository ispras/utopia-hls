/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include "net.h"

using namespace utopia;

int main() {
  Net net;

  Variable clk = Variable::var("clk", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *clknode = net.add_src(clk);

  Variable s1 = Variable::var("s1", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *s1node = net.add_src(s1);

  Variable s2 = Variable::var("s2", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *s2node = net.add_src(s2);

  Variable c = Variable::var("c", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *cnode = net.add_src(c);

  Variable x = Variable::var("x", Variable::WIRE, Variable::INPUT, Type::uint(8));
  VNode *xnode = net.add_src(x);

  Variable y = Variable::var("y", Variable::WIRE, Variable::INPUT, Type::uint(8));
  VNode *ynode = net.add_src(y);

  Variable z = Variable::var("z", Variable::WIRE, Variable::OUTPUT, Type::uint(8));
  VNode *znode = net.add_fun(z, Function::ADD, { xnode, ynode });

  Variable u = Variable::var("u", Variable::WIRE, Variable::INNER, Type::uint(8));
  VNode *unode = net.add_mux(u, { s1node, s2node, xnode, ynode });

  Variable r = Variable::var("r", Variable::REG, Variable::INNER, Type::uint(8));
  VNode *rnode = net.add_reg(r, Event::posedge(clknode), znode);

  net.add_cmb({ cnode }, { unode });
  net.add_seq(Event::posedge(clknode), { cnode }, { rnode });

  return 0;
}

