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
| `--frames N` | Number of inferences |
| `--debug-level N` | TIDL `debug_level`; 2 prints the C7x memory table to the remote log |
| `--core N` | C7x core, 1 (default) or 2. The current vision_apps memory map gives C7x_2 too little heap for this network |
| `--no-offload` | ARM-only run, to compare against TIDL |
