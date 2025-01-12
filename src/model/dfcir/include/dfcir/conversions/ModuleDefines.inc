//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

// Module names for operations.
#define ADD_MODULE "__ADD"
#define SUB_MODULE "__SUB"
#define MUL_MODULE "__MUL"
#define DIV_MODULE "__DIV"
#define NEG_MODULE "__NEG"

#define AND_MODULE "__AND"
#define OR_MODULE "__OR"
#define XOR_MODULE "__XOR"
#define NOT_MODULE "__NOT"

#define LESS_MODULE "__LESS"
#define LESSEQ_MODULE "__LESSEQ"
#define GREATER_MODULE "__GREATER"
#define GREATEREQ_MODULE "__GREATEREQ"
#define EQ_MODULE "__EQ"
#define NEQ_MODULE "__NEQ"

// Specification names for operation's types.
#define FLOAT_SPEC "FLOAT"
#define INT_SPEC "INT"
#define UINT_SPEC "UINT"
#define SINT_SPEC "SINT"

// Utility definitions.
#define BUF_MODULE "__FIFO"
#define STAGES_PARAM "stages"
#define CLOCK_ARG "clk"

#define TYPE_SIZE_PARAM "size"
#define INSTANCE_LATENCY_ATTR "__latency"
#define CONNECT_OFFSET_ATTR "__offset"
