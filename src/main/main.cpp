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

#include <iostream>

#include "gate/flibrary.h"
#include "gate/gate.h"
#include "gate/netlist.h"
#include "rtl/net.h"

using namespace eda::gate;
using namespace eda::rtl;

int main() {
  Net net;

  // module(
  //   input  wire       clk,
  //   input  wire       rst,
  //   input  wire       c,
  //   input  wire [7:0] x,
  //   input  wire [7:0] y,
  //   output wire [7:0] u,
  //   output wire [7:0] v
  // );
  //   reg  [7:0] r;
  //   wire [7:0] f;
  //   wire [7:0] g;
  //   wire [7:0] w;
  //
  //   assign f = x + y;
  //   assign g = x - y;
  //
  //   always @(*) begin
  //     if (c) w <= f;
  //     else   w <= g; 
  //   end
  //
  //   always @(posedge clk, posedge rst) begin
  //     if (rst)    r <= 0;
  //     else if (c) r <= f;
  //     else        r <= g;
  //   end
  //
  //   assign u = w;
  //   assign v = r;
  // endmodule

  Variable clk("clk", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *clknode = net.add_src(clk);

  Variable rst("rst", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *rstnode = net.add_src(rst);

  Variable c("c", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *cnode = net.add_src(c);

  Variable n("n", Variable::WIRE, Type::uint(1));
  VNode *nnode = net.add_fun(n, FuncSymbol::NOT, { cnode });

  Variable x("x", Variable::WIRE, Variable::INPUT, Type::uint(8));
  VNode *xnode = net.add_src(x);

  Variable y("y", Variable::WIRE, Variable::INPUT, Type::uint(8));
  VNode *ynode = net.add_src(y);

  Variable f("f", Variable::WIRE, Type::uint(8));
  VNode *fnode = net.add_fun(f, FuncSymbol::ADD, { xnode, ynode });

  Variable g("g", Variable::WIRE, Type::uint(8));
  VNode *gnode = net.add_fun(g, FuncSymbol::SUB, { xnode, ynode });

  Variable w("w", Variable::WIRE, Type::uint(8));
  VNode *wnode1 = net.add_fun(w, FuncSymbol::NOP, { fnode });
  VNode *wnode2 = net.add_fun(w, FuncSymbol::NOP, { gnode });
  VNode *w_phi = net.add_phi(w);

  Variable z("z", Variable::WIRE, Type::uint(8));
  VNode *znode = net.add_val(z, { false, false, false, false, false, false, false, false });

  Variable r("r", Variable::REG, Type::uint(8));
  VNode *rnode0 = net.add_reg(r, /* Event::level1(rstnode)  */ znode);
  VNode *rnode1 = net.add_reg(r, /* Event::posedge(clknode) */ fnode);
  VNode *rnode2 = net.add_reg(r, /* Event::posedge(clknode) */ gnode);
  VNode *r_phi = net.add_phi(r);

  Variable u("u", Variable::WIRE, Variable::OUTPUT, Type::uint(8));
  VNode *unode = net.add_fun(u, FuncSymbol::NOP, { w_phi });

  Variable v("v", Variable::WIRE, Variable::OUTPUT, Type::uint(8));
  VNode *vnode = net.add_fun(v, FuncSymbol::NOP, { r_phi });

  net.add_cmb({ cnode }, { wnode1 });
  net.add_cmb({ nnode }, { wnode2 });
  net.add_cmb({}, { unode });
  net.add_cmb({}, { vnode });

  net.add_seq(Event::level1(rstnode), {}, { rnode0 });
  net.add_seq(Event::posedge(clknode), { cnode }, { rnode1 });
  net.add_seq(Event::posedge(clknode), { nnode }, { rnode2 });

  net.create();
  std::cout << net;

  std::cout << "------------------------------------------"  << std::endl;

  Netlist netlist;
  netlist.create(net, FLibraryDefault::instance());
  std::cout << netlist;

  return 0;
}

