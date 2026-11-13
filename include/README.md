# ONNX Runtime headers for the J722S unit

The ONNX Runtime C API headers of the TI ONNX Runtime 1.15 build
(`ORT_API_VERSION 15`, with the TIDL execution provider) that the unit runs as
`/usr/lib/libonnxruntime.so.1.15.0`, copied from the Processor SDK RTOS
11.01.00.04 target filesystem
(`targetfs/usr/include/onnxruntime/include/onnxruntime/core/`):

| File | |
|---|---|
| `onnxruntime/core/session/onnxruntime_c_api.h` | ONNX Runtime C API, Copyright (c) Microsoft Corporation, MIT License |
| `onnxruntime/core/providers/tidl/tidl_provider_factory.h` | TIDL execution provider: `c_api_tidl_options`, `OrtSessionOptionsAppendExecutionProvider_Tidl` |

chameleon-b4 (`app-b4/source/retro_net.cpp` in the parent repo) builds against
these without the SDK: it loads the library on the unit with `dlopen()`. They
must match that library, so update them together with the SDK.
