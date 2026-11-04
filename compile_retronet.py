#!/usr/bin/env python3
"""
Compile retronet_v10_tidl_safe.onnx into TIDL artifacts for J722S (AM67A) with
edgeai-tidl-tools 11_01_06_00, matching Processor SDK RTOS 11.01.00.04 on the
board (TIDL 11_01_06_00).

Same compile options as the 11.02 basic_example config.yaml used before
(8-bit, accuracy_level 1, 2 calibration frames x 5 iterations, seeded random
calibration input), minus the TI-internal ti_internal_nc_flag that 11.01 does
not have.

--cores 1 (default) compiles for a single C7x (inference_mode 0).

--cores 2 compiles for high-throughput mode (inference_mode 1,
TIDL_inferenceModeHighThroughput) on both J722S C7x cores: each core runs its
own copy of the network on one batch item. TIDL needs the batch size to be a
multiple of num_cores, and retronet is exported with batch 1, so a batch-N copy
of the model is written to <out_dir>/../<name>_b<N>.onnx and compiled instead.
retronet_main must be given that batch-N .onnx at run time.

--rows R stacks R independent items along the height axis instead of the
batch axis: input [B, 4, R, 520], output [B, 5, R, 520]. This is exact for
retronet because every op works along width only (1xk convs and pools, and
the nearest Resize + MaxPool(2x1) pairs give back the rows they started
with), and it is much faster than --batch on the C7x: TIDL runs the layers of
each batch item one after another and pays its ~30-40k-cycle per-layer setup
each time, but a taller tensor pays it once. Item r of batch b is row r, so
the runner must lay the input out channel-major per batch: x[b][c][r][w].

Usage (from this folder):
    ./compile_retronet.sh [model.onnx] [out_dir] [--cores N] [--batch B] [--rows R]
"""
import argparse
import glob
import os
import shutil
import struct
import sys

import numpy as np
import onnx
import onnxruntime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "retronet_v10_tidl_safe.onnx")
TIDL_TOOLS = os.environ["TIDL_TOOLS_PATH"]
CALIB_FRAMES = 2
J722S_NUM_C7X = 2
# sizeof(sTIDL_IOBufDesc_t) for TIDL 11_01 built with SOC_J722S
# (TIDL_MAX_NUM_CORES = 2). The C7x firmware rejects any other size, e.g.
# 94616 (AM62A/J721E/J721S2) or 378392 (J784S4/J742S2).
J722S_IO_DESC_SIZE = 189208
INFERENCE_MODES = {0: "default (single core)", 1: "high throughput", 2: "low latency"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
ap.add_argument("out_dir", nargs="?", default=None)
ap.add_argument("--cores", type=int, default=1, choices=range(1, J722S_NUM_C7X + 1),
                help="C7x cores; >1 selects high-throughput mode (default 1)")
ap.add_argument("--batch", type=int, default=None,
                help="batch size, a multiple of --cores (default: --cores)")
ap.add_argument("--rows", type=int, default=1,
                help="items stacked along height per batch entry (default 1)")
args = ap.parse_args()

CORES = args.cores
BATCH = args.batch if args.batch is not None else CORES
if BATCH < 1 or BATCH % CORES != 0:
    sys.exit(f"ERROR: --batch {BATCH} must be a positive multiple of --cores {CORES}")
ROWS = args.rows
if ROWS < 1:
    sys.exit(f"ERROR: --rows {ROWS} must be positive")
INFERENCE_MODE = 1 if CORES > 1 else 0

MODEL = os.path.abspath(args.model)
name = os.path.splitext(os.path.basename(MODEL))[0]
suffix = (f"_ht{CORES}" if INFERENCE_MODE == 1 else "") + \
         (f"_b{BATCH}" if BATCH != CORES else "") + \
         (f"_r{ROWS}" if ROWS != 1 else "")
OUT = os.path.abspath(args.out_dir or
                      os.path.join(HERE, f"model-artifacts/{name}{suffix}/artifacts"))

options = {
    "tidl_tools_path": TIDL_TOOLS,
    "artifacts_folder": OUT,
    "platform": "J7",
    "debug_level": 0,
    "tensor_bits": 8,
    "accuracy_level": 1,
    "max_num_subgraphs": 16,
    "advanced_options:calibration_frames": CALIB_FRAMES,
    "advanced_options:calibration_iterations": 5,
    "advanced_options:mixed_precision_factor": -1,
    "advanced_options:quantization_scale_type": 0,
    "advanced_options:high_resolution_optimization": 0,
    "advanced_options:pre_batchnorm_fold": 1,
    "advanced_options:activation_clipping": 1,
    "advanced_options:weight_clipping": 1,
    "advanced_options:bias_calibration": 1,
    "advanced_options:channel_wise_quantization": 0,
    "advanced_options:add_data_convert_ops": 3,
    "advanced_options:inference_mode": INFERENCE_MODE,
    "advanced_options:num_cores": CORES,
}


def reshape_io(src, dst, batch, rows):
    """Write a copy of src with every graph input/output batch dim set to
    batch and height dim set to rows. Fails if the graph does not infer to
    those, e.g. because a Reshape hardcodes them."""
    m = onnx.load(src)
    for v in list(m.graph.input) + list(m.graph.output):
        v.type.tensor_type.shape.dim[0].dim_value = batch
        v.type.tensor_type.shape.dim[2].dim_value = rows
    del m.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    for v in m.graph.output:
        dims = v.type.tensor_type.shape.dim
        if dims[0].dim_value != batch or dims[2].dim_value != rows:
            sys.exit(f"ERROR: {src} does not reshape cleanly ({v.name} "
                     f"{[d.dim_value for d in dims]})")
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return dst


shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

model_path = MODEL
src_dims = [d.dim_value for d in onnx.load(MODEL).graph.input[0].type.tensor_type.shape.dim]
if src_dims[2] != 1 and ROWS != 1:
    sys.exit(f"ERROR: --rows needs a model with height 1, {MODEL} has {src_dims}")
if src_dims[0] != BATCH or (ROWS != 1 and src_dims[2] != ROWS):
    tag = f"_b{BATCH}" + (f"_r{ROWS}" if ROWS != 1 else "")
    model_path = reshape_io(MODEL, os.path.join(os.path.dirname(OUT), f"{name}{tag}.onnx"),
                            BATCH, ROWS if ROWS != 1 else src_dims[2])

so = rt.SessionOptions()
sess = rt.InferenceSession(model_path, providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
                           provider_options=[options, {}], sess_options=so)

inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
print(f"model {model_path}\ninput {inp.name} {shape}\nartifacts {OUT}")
print(f"inference_mode {INFERENCE_MODE} ({INFERENCE_MODES[INFERENCE_MODE]}), num_cores {CORES}")

# Calibration runs: TIDL collects activation ranges from these frames
for i in range(CALIB_FRAMES):
    np.random.seed(i)
    x = np.random.randn(*shape).astype(np.float32)
    out = sess.run(None, {inp.name: x})
    print(f"calibration frame {i}: output {out[0].shape}")

del sess

io_bins = sorted(glob.glob(os.path.join(OUT, "*_tidl_io_*.bin")))
if not io_bins:
    sys.exit(f"ERROR: no *_tidl_io_*.bin in {OUT}")
status = 0
for f in io_bins:
    size = os.path.getsize(f)
    if size != J722S_IO_DESC_SIZE:
        print(f"ERROR: {os.path.basename(f)} is {size} bytes, J722S needs {J722S_IO_DESC_SIZE}")
        status = 1
        continue
    # sTIDL_IOBufDesc_t starts with numInputBuf, numOutputBuf, numCores,
    # numVirtualCores, numSuperBatches, inferenceMode (int32 each)
    with open(f, "rb") as fh:
        _, _, n_cores, n_vcores, n_super, mode = struct.unpack("<6i", fh.read(24))
    print(f"{os.path.basename(f)}: {size} bytes, inferenceMode {mode}, numCores {n_cores}, "
          f"numVirtualCores {n_vcores}, numSuperBatches {n_super}")
    if mode != INFERENCE_MODE or n_cores != CORES:
        print(f"ERROR: expected inferenceMode {INFERENCE_MODE} with numCores {CORES}")
        status = 1
if status:
    sys.exit("artifacts do not match the J722S TIDL 11_01 firmware or the requested mode "
             "(check SOC=am67a and tools version)")
if model_path != MODEL:
    print(f"run with: --model {os.path.basename(model_path)} (batch {BATCH}, rows {ROWS})")
print("done:", sorted(os.listdir(OUT)))
