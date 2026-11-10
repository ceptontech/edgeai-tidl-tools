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

--labels ends the model with the label rule of verify_onnx_example.py, so
the C7x returns the label per position, [B, 1, H, W] int64 (0 bg, 1 retro,
2 car, 3 cateye, 4 bloom), instead of the [B, 5, H, W] float logits:
    p = softmax over the 5 classes
    label = argmax(p[1:4]) + 1 where max(p[1:4]) > 0.2, 4 where p4 > 0.5
    (bloom wins), else 0
TIDL has no Softmax across channels and no compare/select, so it is built as
    Reshape [B,5,H,W] -> [B,1,5,H*W] -> Softmax(height) -> Reshape  -> p
    1x1 Conv + Clip(0, 1): [e, r1, r2, r3, s]
        rm = pm - .2, s = 16 (p4 - .5), e = 0.002 (16-bit) / 0.03 (8-bit)
    1x1 Conv + Relu:       [e, r1 - s, r2 - s, r3 - s, s]
    ArgMax(axis -3)
Without bloom (s = 0) the retro scores are rm at full resolution and the
largest wins over bg if above e; with bloom s pushes them down and wins.
Weights stay within 16x of each other (8-bit weights lose a 1 next to a 1000)
and bg wins rounding noise through e. Differences from the float rule: retro
needs pm > .2 + e, bloom p4 > .5 + e/16, and a retro class over .2 still beats
bloom if p4 < ~.51. The
model is written next to the artifacts as <name>..._labels.onnx and must be
given to retronet_main as --model. The compile fails if any node of it was
left to the ARM.

--calib example_frame.npz calibrates on a real frame instead of random input:
the verify_onnx_example.py preprocessing (d0/308, i0/432, d1/308, i1/432) cut
into the model's items -- one column per row of the item for 520-wide models
(retronet_v10, --rows), blocks of neighbouring columns for 520-high ones
(retro_dilated2d) -- with --calib-frames batches spread evenly over the frame.

--bits 16 quantizes to 16 bits instead of 8: closer to the float model,
slower on the C7x.

Usage (from this folder):
    ./compile_retronet.sh [model.onnx] [out_dir] [--cores N] [--batch B] [--rows R] [--labels]
                          [--calib frame.npz [--calib-frames N]] [--bits 8|16]
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
CALIB_FRAMES = 2          # random calibration input
CALIB_FRAMES_REAL = 20    # --calib: batches cut from the frame
J722S_NUM_C7X = 2
# sizeof(sTIDL_IOBufDesc_t) for TIDL 11_01 built with SOC_J722S
# (TIDL_MAX_NUM_CORES = 2). The C7x firmware rejects any other size, e.g.
# 94616 (AM62A/J721E/J721S2) or 378392 (J784S4/J742S2).
J722S_IO_DESC_SIZE = 189208
INFERENCE_MODES = {0: "default (single core)", 1: "high throughput", 2: "low latency"}
# --labels output tensor name and thresholds (verify_onnx_example.py)
LABELS_OUTPUT = "retronet_label"
RETRO_P = 0.2   # retro / car / cateye: probability over this
BLOOM_P = 0.5   # bloom: probability over this, wins over retro
BLOOM_K = 16.0    # --labels: bloom score s = BLOOM_K * (p4 - BLOOM_P), clipped to 1
LABEL_EPS = {16: 0.002, 8: 0.03}   # --labels: bg score, per --bits

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("model", nargs="?", default=DEFAULT_MODEL)
ap.add_argument("out_dir", nargs="?", default=None)
ap.add_argument("--cores", type=int, default=1, choices=range(1, J722S_NUM_C7X + 1),
                help="C7x cores; >1 selects high-throughput mode (default 1)")
ap.add_argument("--batch", type=int, default=None,
                help="batch size, a multiple of --cores (default: --cores)")
ap.add_argument("--rows", type=int, default=1,
                help="items stacked along height per batch entry (default 1)")
ap.add_argument("--labels", action="store_true",
                help="end with the softmax + threshold label rule on the C7x: labels "
                     "[B, 1, H, W] int64 instead of [B, 5, H, W] logits")
ap.add_argument("--calib", default=None,
                help="calibrate on a real frame (example_frame.npz) instead of random input")
ap.add_argument("--calib-frames", type=int, default=None,
                help=f"calibration batches (default {CALIB_FRAMES}, "
                     f"{CALIB_FRAMES_REAL} with --calib)")
ap.add_argument("--bits", type=int, default=8, choices=(8, 16),
                help="tensor_bits: 8 (default) or 16")
args = ap.parse_args()

CORES = args.cores
BATCH = args.batch if args.batch is not None else CORES
if BATCH < 1 or BATCH % CORES != 0:
    sys.exit(f"ERROR: --batch {BATCH} must be a positive multiple of --cores {CORES}")
ROWS = args.rows
if ROWS < 1:
    sys.exit(f"ERROR: --rows {ROWS} must be positive")
INFERENCE_MODE = 1 if CORES > 1 else 0
LABELS = args.labels
CALIB = os.path.abspath(args.calib) if args.calib else None
N_CALIB = args.calib_frames or (CALIB_FRAMES_REAL if CALIB else CALIB_FRAMES)

MODEL = os.path.abspath(args.model)
name = os.path.splitext(os.path.basename(MODEL))[0]
suffix = (f"_ht{CORES}" if INFERENCE_MODE == 1 else "") + \
         (f"_b{BATCH}" if BATCH != CORES else "") + \
         (f"_r{ROWS}" if ROWS != 1 else "") + \
         ("_labels" if LABELS else "") + \
         (f"_{args.bits}bit" if args.bits != 8 else "")
OUT = os.path.abspath(args.out_dir or
                      os.path.join(HERE, f"model-artifacts/{name}{suffix}/artifacts"))

options = {
    "tidl_tools_path": TIDL_TOOLS,
    "artifacts_folder": OUT,
    "platform": "J7",
    "debug_level": 0,
    "tensor_bits": args.bits,
    "accuracy_level": 1,
    "max_num_subgraphs": 16,
    "advanced_options:calibration_frames": N_CALIB,
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


def load_for_tidl(src):
    """Load src and fix what the 11.01 TIDL import cannot take. Returns the
    model and the list of fixes made (empty: src can be compiled as is).
     - nodes out of execution order (some exporters append them, e.g.
       retro_dilated2d_best_fp16.onnx lists its input Cast after the Convs
       that read it): shape inference rejects that
     - IR version 10: onnxruntime_tidl 1.15 reads IR <= 9
     - float16 weights ("data type 10 not supported"): converted to float32,
       exact, and TIDL quantizes them anyway; the fp32 <-> fp16 Casts around
       them become no-ops and are removed"""
    m = onnx.load(src)
    fixes = []

    ready = {i.name for i in m.graph.input} | {i.name for i in m.graph.initializer} | {""}
    nodes, order = list(m.graph.node), []
    while nodes:
        runnable = [n for n in nodes if all(i in ready for i in n.input)]
        if not runnable:
            sys.exit(f"ERROR: {src} has nodes whose inputs are never produced")
        for n in runnable:
            order.append(n)
            ready.update(n.output)
            nodes.remove(n)
    if [n.name for n in order] != [n.name for n in m.graph.node]:
        fixes.append("nodes sorted")
    del m.graph.node[:]
    m.graph.node.extend(order)

    if m.ir_version > 9:
        fixes.append(f"IR {m.ir_version} -> 9")
        m.ir_version = 9

    fp16 = [i for i in m.graph.initializer if i.data_type == onnx.TensorProto.FLOAT16]
    if fp16:
        fixes.append(f"{len(fp16)} fp16 weights -> fp32")
        for i in fp16:
            i.CopyFrom(onnx.numpy_helper.from_array(
                onnx.numpy_helper.to_array(i).astype(np.float32), i.name))
        outputs = {o.name for o in m.graph.output}
        casts = 0
        for n in [n for n in m.graph.node if n.op_type == "Cast"]:
            to = onnx.helper.get_attribute_value(n.attribute[0])
            if to not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
                continue
            # Bypass: a graph output keeps its name (its producer is renamed),
            # anything else is read from the Cast's input instead
            old, new = (n.input[0], n.output[0]) if n.output[0] in outputs else \
                       (n.output[0], n.input[0])
            m.graph.node.remove(n)
            casts += 1
            for k in m.graph.node:
                k.input[:] = [new if x == old else x for x in k.input]
                k.output[:] = [new if x == old else x for x in k.output]
        if casts:
            fixes.append(f"{casts} fp16 Casts removed")
    del m.graph.value_info[:]
    return m, fixes


def calib_items(path, item):
    """Items of shape item = [4, H, W] cut from a real frame, preprocessed as
    verify_onnx_example.py: channels d0/308, i0/432, d1/308, i1/432 over the
    frame's 520 rows x columns. A 520-wide item holds one column per row (H
    columns), a 520-high one W neighbouring columns."""
    d = np.load(path)
    f = np.stack([d["dist_mat"] / 308.0, d["int_mat"] / 432.0,
                  d["dist1_mat"] / 308.0, d["int1_mat"] / 432.0]).astype(np.float32)
    c, h, w = item
    rows, cols = f.shape[1:]
    if c != 4:
        sys.exit(f"ERROR: --calib needs 4 input channels, the model has {c}")
    if w == rows:
        n = cols // h
        return f.transpose(2, 0, 1)[:n * h].reshape(n, h, c, w).transpose(0, 2, 1, 3)
    if h == rows:
        n = cols // w
        return f[:, :, :n * w].reshape(c, h, n, w).transpose(2, 0, 1, 3)
    sys.exit(f"ERROR: --calib: frame is {rows} rows, the model item is {item}")


def set_io_shape(m, batch, rows):
    """Set every graph input/output batch dim to batch and height dim to
    rows."""
    for v in list(m.graph.input) + list(m.graph.output):
        v.type.tensor_type.shape.dim[0].dim_value = batch
        v.type.tensor_type.shape.dim[2].dim_value = rows


def add_labels(m, bits):
    """End m in the label rule (see --labels above), from ops the C7x runs:
    [B, 5, H, W] logits -> [B, 1, H, W] int64 labels."""
    if len(m.graph.output) != 1:
        sys.exit(f"ERROR: --labels needs a model with one output, it has {len(m.graph.output)}")
    logits = m.graph.output[0]
    dims = [d.dim_value for d in logits.type.tensor_type.shape.dim]
    if dims[1] != 5:
        sys.exit(f"ERROR: --labels needs 5 classes, {logits.name} has {dims[1]}")

    # bg score: above the rounding floor of the other scores. TIDL keeps the
    # bloom score >= 1 output step (~0.00027 at 16 bits, not 1/32767) where
    # float has 0, and rounds a smaller bg score to 0: then bloom wins every
    # bg pixel
    eps = LABEL_EPS[bits]
    # 1x1 conv + clip(0, 1) on p: c = [eps, r1, r2, r3, s]
    w1 = np.zeros((5, 5, 1, 1), np.float32)
    b1 = np.zeros(5, np.float32)
    b1[0] = eps
    for c in (1, 2, 3):
        w1[c, c] = 1.0
        b1[c] = -RETRO_P
    w1[4, 4] = BLOOM_K
    b1[4] = -BLOOM_K * BLOOM_P
    # 1x1 conv + relu on c: [eps, r1 - s, r2 - s, r3 - s, s]
    w2 = np.zeros((5, 5, 1, 1), np.float32)
    w2[0, 0] = 1.0
    for c in (1, 2, 3):
        w2[c, c] = 1.0
        w2[c, 4] = -1.0
    w2[4, 4] = 1.0
    def const(name, a):
        m.graph.initializer.append(onnx.numpy_helper.from_array(np.asarray(a), name))
        return name
    const("lbl_w1", w1)
    const("lbl_b1", b1)
    const("lbl_w2", w2)
    const("lbl_b2", np.zeros(5, np.float32))
    const("lbl_lo", np.array(0.0, np.float32))
    const("lbl_hi", np.array(1.0, np.float32))
    # Classes onto the height axis, a free reshape: TIDL's Softmax runs along
    # height or width only, and after a Transpose its import loses the shape
    const("lbl_shape_h", np.array([dims[0], 1, 5, dims[2] * dims[3]], np.int64))
    const("lbl_shape_c", np.array(dims, np.int64))

    make = onnx.helper.make_node
    m.graph.node.extend([
        make("Reshape", [logits.name, "lbl_shape_h"], ["lbl_lh"], name="lbl_lh"),
        make("Softmax", ["lbl_lh"], ["lbl_ph"], name="lbl_softmax", axis=-2),
        make("Reshape", ["lbl_ph", "lbl_shape_c"], ["lbl_p"], name="lbl_p"),
        make("Conv", ["lbl_p", "lbl_w1", "lbl_b1"], ["lbl_conv1"], name="lbl_conv1"),
        make("Clip", ["lbl_conv1", "lbl_lo", "lbl_hi"], ["lbl_c"], name="lbl_c"),
        make("Conv", ["lbl_c", "lbl_w2", "lbl_b2"], ["lbl_conv2"], name="lbl_conv2"),
        make("Relu", ["lbl_conv2"], ["lbl_score"], name="lbl_score"),
        make("ArgMax", ["lbl_score"], [LABELS_OUTPUT], name=LABELS_OUTPUT, axis=-3, keepdims=1),
    ])
    del m.graph.output[:]
    m.graph.output.append(onnx.helper.make_tensor_value_info(
        LABELS_OUTPUT, onnx.TensorProto.INT64, [dims[0], 1, dims[2], dims[3]]))


def save_checked(m, dst, batch, rows):
    """Shape-infer m and save it to dst. Fails if the outputs do not infer to
    batch and rows, e.g. because a Reshape hardcodes them."""
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    for v in m.graph.output:
        dims = v.type.tensor_type.shape.dim
        if dims[0].dim_value != batch or dims[2].dim_value != rows:
            sys.exit(f"ERROR: the model does not reshape cleanly ({v.name} "
                     f"{[d.dim_value for d in dims]})")
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return dst


shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

model_path = MODEL
m, fixes = load_for_tidl(MODEL)
src_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
if src_dims[2] != 1 and ROWS != 1:
    sys.exit(f"ERROR: --rows needs a model with height 1, {MODEL} has {src_dims}")
height = ROWS if ROWS != 1 else src_dims[2]
reshape = src_dims[0] != BATCH or src_dims[2] != height
if fixes or reshape or LABELS:
    if fixes:
        print(f"model fixes for TIDL 11.01: {', '.join(fixes)}")
    set_io_shape(m, BATCH, height)
    if LABELS:
        add_labels(m, args.bits)
    tag = f"_b{BATCH}" + (f"_r{ROWS}" if ROWS != 1 else "") + ("_labels" if LABELS else "")
    model_path = save_checked(m, os.path.join(os.path.dirname(OUT), f"{name}{tag}.onnx"),
                              BATCH, height)

so = rt.SessionOptions()
sess = rt.InferenceSession(model_path, providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
                           provider_options=[options, {}], sess_options=so)

inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
print(f"model {model_path}\ninput {inp.name} {shape}\nartifacts {OUT}")
print(f"inference_mode {INFERENCE_MODE} ({INFERENCE_MODES[INFERENCE_MODE]}), num_cores {CORES}")

# Calibration runs: TIDL collects activation ranges from these frames
if CALIB:
    items = calib_items(CALIB, shape[1:])
    pick = np.linspace(0, len(items) - 1, N_CALIB * shape[0]).round().astype(int)
    print(f"calibration: {N_CALIB} batches of {shape[0]} from {len(items)} items of {CALIB}")
for i in range(N_CALIB):
    if CALIB:
        x = items[pick[i * shape[0]:(i + 1) * shape[0]]]
    else:
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
# allowedNode.txt: number of subgraphs, then per subgraph its node count and
# node indices. --labels needs every node on the C7x, the label one included
if LABELS:
    with open(os.path.join(OUT, "allowedNode.txt")) as fh:
        vals = [int(v) for v in fh.read().split()]
    offloaded, i = 0, 1
    for _ in range(vals[0]):
        offloaded += vals[i]
        i += vals[i] + 1
    total = len(onnx.load(model_path).graph.node)
    if offloaded != total:
        print(f"ERROR: only {offloaded} of {total} nodes run on the C7x: the label rule "
              "would be split with the ARM")
        status = 1
    elif not any(LABELS_OUTPUT.encode() in open(f, "rb").read() for f in io_bins):
        print(f"ERROR: {LABELS_OUTPUT} is not a TIDL subgraph output")
        status = 1
if status:
    sys.exit("artifacts do not match the J722S TIDL 11_01 firmware or the requested mode "
             "(check SOC=am67a and tools version)")
if model_path != MODEL:
    print(f"run with: --model {os.path.basename(model_path)} (batch {BATCH}, rows {ROWS}"
          f"{', labels' if LABELS else ''})")
print("done:", sorted(os.listdir(OUT)))
