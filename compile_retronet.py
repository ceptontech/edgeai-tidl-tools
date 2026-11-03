#!/usr/bin/env python3
"""
Compile retronet_v10_tidl_safe.onnx into TIDL artifacts for J722S (AM67A) with
edgeai-tidl-tools 11_01_06_00, matching Processor SDK RTOS 11.01.00.04 on the
board (TIDL 11_01_06_00).

Same compile options as the 11.02 basic_example config.yaml used before
(8-bit, accuracy_level 1, 2 calibration frames x 5 iterations, 1 core,
seeded random calibration input), minus the TI-internal ti_internal_nc_flag
that 11.01 does not have.

Usage (from this folder):
    ./compile_retronet.sh [model.onnx] [out_dir]
"""
import glob
import os
import shutil
import sys

import numpy as np
import onnxruntime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                        "/home/qcs4tracy/github/vision_apps/retronet_v10_tidl_safe.onnx")
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else
                      os.path.join(HERE, "model-artifacts/retronet_v10_tidl_safe/artifacts"))
TIDL_TOOLS = os.environ["TIDL_TOOLS_PATH"]
CALIB_FRAMES = 2
# sizeof(sTIDL_IOBufDesc_t) for TIDL 11_01 built with SOC_J722S
# (TIDL_MAX_NUM_CORES = 2). The C7x firmware rejects any other size, e.g.
# 94616 (AM62A/J721E/J721S2) or 378392 (J784S4/J742S2).
J722S_IO_DESC_SIZE = 189208

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
    "advanced_options:inference_mode": 0,
    "advanced_options:num_cores": 1,
}

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

so = rt.SessionOptions()
sess = rt.InferenceSession(MODEL, providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
                           provider_options=[options, {}], sess_options=so)

inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
print(f"model {MODEL}\ninput {inp.name} {shape}\nartifacts {OUT}")

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
bad = [(f, os.path.getsize(f)) for f in io_bins if os.path.getsize(f) != J722S_IO_DESC_SIZE]
for f, size in bad:
    print(f"ERROR: {os.path.basename(f)} is {size} bytes, J722S needs {J722S_IO_DESC_SIZE}")
if bad:
    sys.exit("artifacts do not match the J722S TIDL 11_01 firmware (check SOC=am67a and tools version)")
for f in io_bins:
    print(f"{os.path.basename(f)}: {J722S_IO_DESC_SIZE} bytes, ok for J722S")
print("done:", sorted(os.listdir(OUT)))
