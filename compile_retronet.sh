#!/usr/bin/env bash
# Compile retronet for J722S with the 11_01_06_00 tools in this folder.
# Uses the 11.01 onnxruntime_tidl wheel from ./pylib and tidl_tools from
# ./tools/AM67A, so the system Python install is not touched.
#
# Usage: ./compile_retronet.sh [model.onnx] [out_dir]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export SOC=am67a
export TIDL_TOOLS_PATH=$HERE/tools/AM67A/tidl_tools
export LD_LIBRARY_PATH=$TIDL_TOOLS_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$HERE/pylib${PYTHONPATH:+:$PYTHONPATH}
cd "$HERE"
exec python3 compile_retronet.py "$@"
