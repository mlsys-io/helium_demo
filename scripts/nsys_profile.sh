#!/bin/bash

set -e

# delay=225 when KV and prompt cache are enabled
# delay=150 when only prompt cache is enabled
nsys profile -o report.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node "$@"
