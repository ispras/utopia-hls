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

using namespace eda::ir;

int main() {
  Net net;

  // module(
  //   input  wire       clk,
  //   input  wire       c,
  //   input  wire [7:0] x,
  //   input  wire [7:0] y,
  //   output wire [7:0] f,
  //   output wire [7:0] g
  // );
  //   reg  [7:0] r;
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
  //   always @(posedge clk) begin
  //     if (c) r <= f;
  //     else   r <= g;
  //   end
  // endmodule

  Variable clk("clk", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *clknode = net.add_src(clk);

  Variable c("c", Variable::WIRE, Variable::INPUT, Type::uint(1));
  VNode *cnode = net.add_src(c);

  Variable n("~c", Variable::WIRE, Type::uint(1));
  VNode *nnode = net.add_fun(n, Function::NOT, { cnode });

  Variable x("x", Variable::WIRE, Variable::INPUT, Type::uint(8));
  VNode *xnode = net.add_src(x);

  Variable y("y", Variable::WIRE, Variable::INPUT, Type::uint(8));
  VNode *ynode = net.add_src(y);

  Variable f("f", Variable::WIRE, Variable::OUTPUT, Type::uint(8));
  VNode *fnode = net.add_fun(f, Function::ADD, { xnode, ynode });

  Variable g("g", Variable::WIRE, Variable::OUTPUT, Type::uint(8));
  VNode *gnode = net.add_fun(g, Function::SUB, { xnode, ynode });

  Variable w("w", Variable::WIRE, Type::uint(8));
  VNode *wnode1 = net.add_fun(w, Function::NOP, { fnode });
  VNode *wnode2 = net.add_fun(w, Function::NOP, { gnode });

  Variable r("r", Variable::REG, Type::uint(8));
  VNode *rnode1 = net.add_reg(r, Event::posedge(clknode), fnode);
  VNode *rnode2 = net.add_reg(r, Event::posedge(clknode), gnode);

  net.add_cmb({ cnode }, { wnode1 });
  net.add_cmb({ nnode }, { wnode2 });

  net.add_seq(Event::posedge(clknode), { cnode }, { rnode1 });
  net.add_seq(Event::posedge(clknode), { nnode }, { rnode2 });

  net.create();

  return 0;
}

