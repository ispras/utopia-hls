## DFCxx Simulation

DFCxx kernels can be simulated to check that they describe computations as expected. The simulation doesn't take scheduling into account and requires every computational node to use **and** accept some values at every simulation tick.

### Input format

The format to provide simulation input data is the following:

* input data is divided into blocks separated with a newline character (`\n`) - one block for each simulation step
* every block has a number of lines, each of which has the case-sensitive name of some **input** stream/scalar variable and its hex-value (these values do not have an explicit type - they will be casted to the types of related computational nodes)
* stream/scalar value name and the hex-value are separated with a single space character (` `)
* the provided value must be a valid hex-value: with or without `0x`, with either lower- or uppercase letters
* if some stream/scalar hex-value is present twice or more in the same block - its latest described value is used

Here is an example of an input simulation file for `MuxMul` kernel, which has two input streams: `x` (unsigned 32-bit integer values) and `ctrl` (unsigned 1-bit integer values).

```txt
x 0x32
ctrl 0x1

x 0x45
ctrl 0x0

x 0x56
ctrl 0x1
```

In this example, `x` accepts values `50` (`0x32`), `69` (`0x45`) and `86` (`0x56`), while `ctrl` accepts `1`, `0` and `1`. This means that 3 simulation ticks will be performed for the provided kernel.

### Not Supported Constructions

* *offsets*
