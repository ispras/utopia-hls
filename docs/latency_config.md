## Latency Configuration

Latency configuration is a JSON-based file describing characteristics of computational operations for the specific DFCxx kernel.

Currently each operation has two specifications based on the types of its arguments: for integer values (`INT`) and floating point (`FLOAT`) values respectively. 

All the supported computational operations are listed below:

* `ADD` - Addition
* `SUB` - Subtraction
* `MUL` - Multiplication
* `DIV` - Division
* `AND` - Logical conjunction
* `OR` - Logical disjunction
* `XOR` - Exclusive logical disjunction
* `NOT` - Logical inversion
* `NEG` - Negation
* `LESS` - "less" comparison
* `LESSEQ` - "less or equal" comparison
* `GREATER` - "greater" comparison 
* `GREATEREQ` - "greater or equal" comparison
* `EQ` - "equal" comparison
* `NEQ` - "not equal" comparison

JSON configuration structure states that for every operation with a specific configuration (each pair is represented as a separate JSON-field with `_` between pair's elements) present in the kernel, operation's latency has to be provided. 

Here is an example of a JSON configuration file, containing latencies for multiplication, addition and subtraction of integer numbers:

```json
{
  "MUL_INT": 3,
  "ADD_INT": 1,
  "SUB_INT": 1
}
```
