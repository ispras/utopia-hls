# ==================== COMPILATION RELATED SETTINGS ====================
# Path to the kernel sources (from "./driver", if relative path is used)
KSOURCE_DIR?=../../../kernel

# Cross compiler "prepend" string
CROSS_COMPILE?=arm-linux-gnueabihf-

# Architecture
ARCH=arm

# Compile with debug information
CMA_DEBUG?=0

# ==================== DRIVER RELATED SETTINGS ====================
# Node name used in "/dev" folder
DRIVER_NODE_NAME="cma"

# Unique (across system) ioctl magic number. Every ioctl interface should have one.
CMA_IOC_MAGIC=0xf2
