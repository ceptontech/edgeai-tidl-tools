# retronet on J722S (TIDL 11_01_06_00)

Compile and run `retronet_v10_tidl_safe.onnx` on the Cepton J722S board, which runs
Processor SDK RTOS 11.01.00.04 with TIDL `11_01_06_00` on the C7x cores.

The model takes a `[1, 4, 1, 520]` float32 tensor and returns `[1, 5, 1, 520]`
logits. There is no image pre/post-processing.

## Artifact size

Every `*_tidl_io_*.bin` must be exactly `sizeof(sTIDL_IOBufDesc_t)` for J722S,
which is **189208 bytes** (`TIDL_MAX_NUM_CORES = 2`). Other sizes mean the
artifacts were built for another SOC or TIDL release, and the C7x rejects them
("Reference is NULL" / `TIVX_CMD_NODE_CREATE` failed):

| SOC | io.bin size |
|---|---|
| J722S (am67a) | 189208 |
| AM62A / J721E / J721S2 | 94616 |
| J784S4 / J742S2 | 378392 |

Both `compile_retronet.py` and `retronet_main` check this and fail otherwise.

## 1. Compile artifacts (x86 host)

From the repo root, with `tools/AM67A/tidl_tools` and `pylib/` in place:

```bash
R=https://software-dl.ti.com/jacinto7/esd/tidl-tools/11_01_06_00
mkdir -p tools/AM67A && wget -qO- $R/TIDL_TOOLS/AM67A/tidl_tools.tar.gz | tar -xz -C tools/AM67A
pip install --no-deps --target pylib \
    $R/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl

./compile_retronet.sh <retronet_v10_tidl_safe.onnx> [out_dir]
```

The default output is `model-artifacts/retronet_v10_tidl_safe/artifacts`.

### High-throughput mode (both C7x cores)

```bash
./compile_retronet.sh --cores 2
```

This compiles with `advanced_options:inference_mode 1` (TIDL_inferenceModeHighThroughput)
and `advanced_options:num_cores 2`. Each C7x runs its own copy of the network on one
batch item, so one `Run()` handles 2 frames in parallel. Latency per frame does not
improve, only throughput.

TIDL needs the batch size to be a multiple of `num_cores`, but retronet is exported
with batch 1. The script therefore writes a batch-2 copy of the model and compiles
that:

| File | |
|---|---|
| `model-artifacts/retronet_v10_tidl_safe_ht2/retronet_v10_tidl_safe_b2.onnx` | `[2, 4, 1, 520]` -> `[2, 5, 1, 520]`, pass it as `--model` |
| `model-artifacts/retronet_v10_tidl_safe_ht2/artifacts/` | io.bin header: `inferenceMode 1, numCores 2` |

`--batch 4` (any multiple of `--cores`) gives 2 items per core per `Run()`.

High-throughput mode needs vision_apps firmware that gives C7x_2 the same heaps as
C7x_1 (3+1 MB non-cacheable, 8+8 MB cacheable each), with its matching DTB.

### Stacked rows (fastest)

```bash
./compile_retronet.sh --rows 16                       # one C7x, 16 items per Run()
./compile_retronet.sh --cores 2 --batch 2 --rows 32   # both C7x, 64 items per Run()
```

Every retronet op works along width only, so `--rows R` stacks R independent items
along height: `[B, 4, R, 520]` -> `[B, 5, R, 520]`, item r in row r. The result is
identical to running the items one by one (checked on CPU). It is much faster than
`--batch`, because TIDL pays each layer's setup once per batch entry, not once per
row. Measured on the board with `TIDL_RT_DDR_STATS=0` (batch 1 with it on, which
costs about 0.16 ms per `Run()`):

| Artifacts | Items per `Run()` | ms per `Run()` | items/s |
|---|---|---|---|
| batch 1 | 1 | 1.69 | 591 |
| `--batch 4` | 4 | 3.53 | 1,134 |
| `--rows 16` | 16 | 2.04 | 7,859 |
| `--cores 2 --batch 2 --rows 16` | 32 | 2.27 | 14,113 |
| `--cores 2 --batch 2 --rows 32` | 64 | 2.78 | 23,062 |

The input for item (b, r) is the strided slice `x[b][:][r][:]`. `retronet_main` here
does not lay inputs out that way; use the 11.02 `runtimes/examples/cpp/retronet_example`,
which detects the row count from the model.

### Labels on the C7x

```bash
./compile_retronet.sh retro_dilated2d_best_fp16.onnx --labels --bits 16 --calib example_frame.npz
```

`--labels` ends the model with the label rule of `verify_onnx_example.py`, so the C7x
returns the label per position, `[B, 1, H, W]` int64 (0 bg, 1 retro, 2 car, 3 cateye,
4 bloom), instead of `[B, 5, H, W]` float logits:

```
p     = softmax over the 5 classes
label = argmax(p[1:4]) + 1   where max(p[1:4]) > 0.2
        4                    where p[4] > 0.5   (bloom wins)
        0                    otherwise
```

TIDL has no Softmax across channels and no compare/select, so the rule is built from
ops it does run:

```
Reshape [B,5,H,W] -> [B,1,5,H*W], Softmax(height), Reshape back      -> p
1x1 Conv + Clip(0, 1): [e, p1-.2, p2-.2, p3-.2, s = 16 (p4-.5)]      -> c
1x1 Conv + Relu:       [e, c1 - s, c2 - s, c3 - s, s]
ArgMax (axis -3)
```

Without bloom the retro scores keep their full resolution; with bloom `s` pushes them
down and wins. `e` (0.002 at 16 bits) keeps bg above TIDL's rounding floor: the bloom
score never drops below one output step, so a smaller bg score lost every bg pixel to
bloom. The cost against the float rule: retro needs `p > 0.202`, bloom `p4 > 0.500125`,
and a retro class over 0.2 still beats bloom while `p4 < ~0.51`. The model is written as
`<name>_b<B>[_r<R>]_labels[_16bit].onnx` next to the artifacts; pass it as `--model`.
The compile fails if any node was left to the ARM.

Use it at 16 bits: at 8 bits the network's own logits are too coarse (cateye is lost
completely) and this head was not made to work there.

### Calibration and precision

`--calib example_frame.npz` calibrates on a real frame (the `verify_onnx_example.py`
preprocessing, cut into the model's items, `--calib-frames` batches spread over the
frame, default 20) instead of random input. `--bits 16` quantizes to 16 bits. For
`retro_dilated2d_best_fp16.onnx` on `example_frame.npz` (TIDL host emulation, labels vs
the same model in float, all 907,200 pixels; calibrated and tested on the same frame):

| Build | Overall | bg | retro | cateye | bloom |
|---|---|---|---|---|---|
| 8-bit, random calibration | 34.18% | 34.7% | 1.9% | 0% | 0% |
| 8-bit, `--calib` | 98.05% | 99.7% | 84.3% | 0% | 78.5% |
| 16-bit, `--calib`, rule on the ARM | 99.70% | 99.9% | 97.6% | 85.1% | 96.9% |
| 16-bit, `--calib`, `--labels` on the C7x | 99.71% | 99.9% | 97.6% | 85.0% | 96.9% |

(1 car pixel in the frame.) 16-bit runs slower on the C7x than 8-bit; measure on the board.

### Models the 11.01 import cannot take as exported

`compile_retronet.py` writes a fixed copy of the model (and says so) when it has:

- nodes out of execution order (`retro_dilated2d_best_fp16.onnx` lists its input
  `Cast` after the Convs that read it): sorted
- IR version 10: written as 9 (onnxruntime_tidl 1.15 reads up to 9)
- float16 weights (TIDL: "data type 10 not supported"): converted to float32, which
  is exact, and the fp32 <-> fp16 `Cast`s around them are removed

`retro_dilated2d_best_fp16.onnx` takes `[B, 4, 520, 4]` and returns `[B, 5, 520, 4]`;
`--rows` does not apply to it (height is 520, not 1).

## 2. Build `retronet_main` (cross-compile for aarch64)

This needs the ARM GNU 13.2 toolchain in
`tools/AM67A/arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu`, and the
SDK target filesystem.

```bash
export SOC=am67a
export TIDL_TOOLS_PATH=$PWD/tools/AM67A/tidl_tools
export TARGET_FS_PATH=<ti-processor-sdk-rtos-j722s-evm-11_01_00_04>/targetfs
mkdir -p examples/build && cd examples/build
cmake -DTARGET_CPU=arm ..
make retronet_main            # -> bin/Release/retronet_main
```

## 3. Run on the board

The C7x and mcu2_0 cores must be running the vision_apps firmware (see
`recover-tidl.sh` in vision_apps).

```bash
./retronet_main --artifacts ./artifacts --model retronet_v10_tidl_safe.onnx
```

| Option | Meaning |
|---|---|
| `--input <f32.bin>` | Raw float32 input; seeded random input if omitted |
| `--output <f32.bin>` | Raw float32 logits (default `retronet_logits.bin`) |
| `--frames N` | Number of timed `Run()` calls, one full batch each |
| `--warmup N` | Untimed `Run()` calls first |
| `--quiet` | Timing summary only |
| `--debug-level N` | TIDL `debug_level`; 2 prints the C7x memory table to the remote log |
| `--core N` | C7x core, 1 (default) or 2, for single-core artifacts. C7x_2 needs the vision_apps memory map with equal C7x heaps (C7x heaps 0xA6800000 40 MB, shared 0xA9000000 16 MB) |
| `--no-offload` | ARM-only run, to compare against TIDL |

The runner reads the inference mode from the artifacts' io.bin. For high-throughput
artifacts it checks that the model batch is a multiple of the core count, runs on
C7x_1..C7x_N, and reports per-item results plus avg/min/max `Run()` latency and
items/s:

```bash
./retronet_main --artifacts ./artifacts_ht2 --model retronet_v10_tidl_safe_b2.onnx \
                --warmup 5 --frames 200 --quiet
```

The 11.01 ORT C API (`c_api_tidl_options`) has no `core_start_idx`, so high-throughput
runs always start at C7x_1. On J722S with 2 cores that is the only option anyway.
`--input` accepts either one item (copied to every batch slot) or a whole batch.
