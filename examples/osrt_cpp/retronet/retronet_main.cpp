/*
 * Standalone runner for the retronet_v10_tidl_safe model (11_01 release
 * layout: ONNX Runtime C++ API with the TIDL execution provider).
 *
 * The model is not an image network: it takes a [1, 4, 1, 520] float tensor and
 * produces [1, 5, 1, 520] logits, i.e. a 5-class score for each of the 520
 * positions. There is therefore no image pre/post-processing here -- input is
 * either read verbatim from a raw float32 binary or generated, and the output
 * is written back as a raw float32 binary alongside an argmax class summary.
 *
 * Before the session is created, every *_tidl_io_*.bin in the artifacts folder
 * is checked against sizeof(sTIDL_IOBufDesc_t) as built for this SOC (J722S:
 * 189208 bytes). Artifacts compiled for another SOC or TIDL release fail there
 * with a clear message instead of "Reference is NULL" on the C7x.
 *
 * The same descriptor records the TIDL inference mode the artifacts were
 * compiled for. High-throughput artifacts (inference_mode 1, built with
 * compile_retronet.sh --cores 2) run one copy of the network per C7x core, one
 * batch item per core, so the model must be the batch-N .onnx the compile step
 * wrote. Every Run() then processes N items in parallel on C7x_1..C7x_N.
 *
 * Usage:
 *   retronet_main --artifacts <dir> [--model <onnx>] [--input <f32.bin>]
 *                 [--output <f32.bin>] [--frames N] [--warmup N] [--quiet]
 *                 [--debug-level N] [--core N] [--no-offload]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

#include "itidl_rt.h"
#include "itidl_ti.h"

#if !defined(SOC_J722S)
#error "retronet_main is built for J722S only (define SOC_J722S)"
#endif

namespace
{

/* sTIDL_IOBufDesc_t is sized by TIDL_MAX_NUM_CORES, which is 2 on J722S. */
const size_t kIoDescSize = sizeof(sTIDL_IOBufDesc_t);
static_assert(sizeof(sTIDL_IOBufDesc_t) == 189208,
              "sTIDL_IOBufDesc_t does not match TIDL 11_01 for J722S");

struct Args
{
    std::string model      = "retronet_v10_tidl_safe.onnx";
    std::string artifacts;
    std::string inputFile;
    std::string outputFile = "retronet_logits.bin";
    int32_t     frames     = 1;
    int32_t     warmup     = 0;
    int32_t     debugLevel = 0;
    int32_t     core       = 1;
    bool        coreSet    = false;
    bool        offload    = true;
    bool        quiet      = false;
};

/* Leading fields of sTIDL_IOBufDesc_t, shared by every subgraph. */
struct ArtifactInfo
{
    int32_t inferenceMode   = TIDL_inferenceModeDefault;
    int32_t numCores        = 1;
    int32_t numVirtualCores = 1;
    int32_t numSuperBatches = 1;
};

const char *modeName(int32_t mode)
{
    switch (mode)
    {
        case TIDL_inferenceModeDefault:        return "default (single core)";
        case TIDL_inferenceModeHighThroughput: return "high throughput";
        case TIDL_inferenceModeLowLatency:     return "low latency";
        default:                               return "unknown";
    }
}

void usage(const char *prog)
{
    printf("Usage: %s --artifacts <dir> [options]\n\n", prog);
    printf("  --artifacts <dir>   Directory holding the compiled TIDL artifacts (required)\n");
    printf("  --model <path>      Path to retronet_v10_tidl_safe.onnx\n");
    printf("  --input <file>      Raw float32 input; generated if omitted\n");
    printf("  --output <file>     Raw float32 output (default: retronet_logits.bin)\n");
    printf("  --frames <N>        Number of timed Run() calls (default: 1); each runs a\n");
    printf("                      full batch, i.e. one item per core in high-throughput mode\n");
    printf("  --warmup <N>        Untimed Run() calls before the timed ones (default: 0)\n");
    printf("  --quiet             Print only the timing summary, not per-frame results\n");
    printf("  --debug-level <N>   TIDL debug_level (default: 0); 2 prints the C7x memory\n");
    printf("                      requirement table to the remote core log\n");
    printf("  --core <N>          C7x core to run on, 1 or 2 (default: 1). Single-core\n");
    printf("                      artifacts only; high-throughput ones use C7x_1..C7x_N\n");
    printf("  --no-offload        Run on ARM only, without TIDL offload\n");
    printf("  --help              Show this message\n");
}

int32_t parseArgs(int argc, char *argv[], Args &args)
{
    for (int32_t i = 1; i < argc; i++)
    {
        std::string a(argv[i]);
        bool needsValue = (a != "--no-offload" && a != "--quiet" && a != "--help" && a != "-h");

        if (needsValue && (i + 1) >= argc)
        {
            printf("[ERROR] Missing value for %s\n", a.c_str());
            return -1;
        }

        if      (a == "--model")       args.model      = argv[++i];
        else if (a == "--artifacts")   args.artifacts  = argv[++i];
        else if (a == "--input")       args.inputFile  = argv[++i];
        else if (a == "--output")      args.outputFile = argv[++i];
        else if (a == "--frames")      args.frames     = std::atoi(argv[++i]);
        else if (a == "--warmup")      args.warmup     = std::atoi(argv[++i]);
        else if (a == "--debug-level") args.debugLevel = std::atoi(argv[++i]);
        else if (a == "--core")      { args.core       = std::atoi(argv[++i]); args.coreSet = true; }
        else if (a == "--no-offload")  args.offload    = false;
        else if (a == "--quiet")       args.quiet      = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 1; }
        else
        {
            printf("[ERROR] Unknown option: %s\n", a.c_str());
            return -1;
        }
    }

    if (args.artifacts.empty() && args.offload)
    {
        printf("[ERROR] --artifacts is required\n");
        return -1;
    }
    if (args.frames < 1)
    {
        printf("[ERROR] --frames must be >= 1\n");
        return -1;
    }
    if (args.warmup < 0)
    {
        printf("[ERROR] --warmup must be >= 0\n");
        return -1;
    }
    if (args.core < 1 || args.core > 2)
    {
        printf("[ERROR] --core must be 1 or 2\n");
        return -1;
    }
    return 0;
}

/* Every TIDL subgraph has one <name>_tidl_io_<n>.bin, a raw sTIDL_IOBufDesc_t.
 * Its size is fixed per SOC and TIDL release, so it is a cheap way to catch
 * artifacts that were compiled with the wrong tools. */
int32_t checkArtifacts(const std::string &dir, ArtifactInfo &info)
{
    DIR *d = opendir(dir.c_str());
    if (d == NULL)
    {
        printf("[ERROR] Could not open artifacts folder: %s\n", dir.c_str());
        return -1;
    }

    int32_t found  = 0;
    int32_t status = 0;
    for (struct dirent *e = readdir(d); e != NULL; e = readdir(d))
    {
        const std::string name(e->d_name);
        if (name.find("_tidl_io_") == std::string::npos ||
            name.size() < 4 || name.compare(name.size() - 4, 4, ".bin") != 0)
        {
            continue;
        }

        struct stat st;
        const std::string path = dir + "/" + name;
        if (stat(path.c_str(), &st) != 0)
        {
            printf("[ERROR] Could not stat %s\n", path.c_str());
            status = -1;
            continue;
        }

        if ((size_t)st.st_size != kIoDescSize)
        {
            printf("[ERROR] %s is %ld bytes, J722S TIDL 11_01 expects %zu.\n"
                   "        Recompile with edgeai-tidl-tools 11_01_06_00 and SOC=am67a.\n",
                   path.c_str(), (long)st.st_size, kIoDescSize);
            status = -1;
            continue;
        }

        /* Only the leading scalars are needed; the struct is ~185 KB. */
        int32_t head[6];
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.read(reinterpret_cast<char *>(head), sizeof(head)))
        {
            printf("[ERROR] Could not read %s\n", path.c_str());
            status = -1;
            continue;
        }
        static_assert(offsetof(sTIDL_IOBufDesc_t, numCores)        == 2 * sizeof(int32_t), "layout");
        static_assert(offsetof(sTIDL_IOBufDesc_t, numVirtualCores) == 3 * sizeof(int32_t), "layout");
        static_assert(offsetof(sTIDL_IOBufDesc_t, numSuperBatches) == 4 * sizeof(int32_t), "layout");
        static_assert(offsetof(sTIDL_IOBufDesc_t, inferenceMode)   == 5 * sizeof(int32_t), "layout");
        ArtifactInfo cur;
        cur.numCores        = head[2];
        cur.numVirtualCores = head[3];
        cur.numSuperBatches = head[4];
        cur.inferenceMode   = head[5];

        printf("Artifact  : %s (%ld bytes, ok) mode %d %s, %d core(s)\n",
               name.c_str(), (long)st.st_size, cur.inferenceMode,
               modeName(cur.inferenceMode), cur.numCores);

        if (found > 0 && (cur.inferenceMode != info.inferenceMode || cur.numCores != info.numCores))
        {
            printf("[ERROR] %s was compiled for a different inference mode or core count "
                   "than the other subgraphs\n", name.c_str());
            status = -1;
        }
        info = cur;
        found++;
    }
    closedir(d);

    if (found == 0)
    {
        printf("[ERROR] No *_tidl_io_*.bin in %s\n", dir.c_str());
        status = -1;
    }
    if (status == 0 && info.inferenceMode == TIDL_inferenceModeLowLatency)
    {
        printf("[ERROR] Low-latency (inference_mode 2) artifacts are not supported here\n");
        status = -1;
    }
    if (status == 0 && (info.numCores < 1 || info.numCores > (int32_t)TIDL_MAX_NUM_CORES))
    {
        printf("[ERROR] Artifacts ask for %d cores, J722S has %u\n",
               info.numCores, (uint32_t)TIDL_MAX_NUM_CORES);
        status = -1;
    }
    return status;
}

/* Shared memory when TIDL offload is active, plain aligned memory otherwise. */
void *allocMem(size_t size, bool offload, int32_t alignment = 64)
{
    void *ptr = NULL;

    if (offload)
    {
        ptr = TIDLRT_allocSharedMem(alignment, (int32_t)size);
    }
    if (ptr == NULL)
    {
        if (posix_memalign(&ptr, alignment, size) != 0)
        {
            ptr = NULL;
        }
    }
    return ptr;
}

void freeMem(void *ptr, bool offload)
{
    if (ptr == NULL)
    {
        return;
    }
    if (offload)
    {
        TIDLRT_freeSharedMem(ptr);
    }
    else
    {
        free(ptr);
    }
}

int64_t numElements(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); i++)
    {
        n *= (shape[i] > 0) ? shape[i] : 1;
    }
    return n;
}

/* Fills a batch of `batch` items of `itemElems` floats each, either from a
 * raw float32 file or with deterministic pseudo-random values. The file holds
 * either the whole batch or a single item, which is then copied to every batch
 * slot. Returns 0 on success. */
int32_t fillInput(float *dst, int64_t batch, int64_t itemElems,
                  const std::string &file, int32_t frame)
{
    const int64_t elems = batch * itemElems;

    if (!file.empty())
    {
        std::ifstream ifs(file, std::ios::binary | std::ios::ate);
        if (!ifs)
        {
            printf("[ERROR] Could not open input file: %s\n", file.c_str());
            return -1;
        }

        const std::streamsize bytes     = ifs.tellg();
        const std::streamsize batchSize = (std::streamsize)(elems * sizeof(float));
        const std::streamsize itemSize  = (std::streamsize)(itemElems * sizeof(float));
        if (bytes != batchSize && bytes != itemSize)
        {
            printf("[ERROR] %s holds %ld bytes, model expects %ld (batch of %ld) or %ld (one item)\n",
                   file.c_str(), (long)bytes, (long)batchSize, (long)batch, (long)itemSize);
            return -1;
        }

        ifs.seekg(0);
        ifs.read(reinterpret_cast<char *>(dst), bytes);
        if (!ifs)
        {
            printf("[ERROR] Short read from %s\n", file.c_str());
            return -1;
        }
        if (bytes == itemSize)
        {
            for (int64_t b = 1; b < batch; b++)
            {
                std::copy(dst, dst + itemElems, dst + b * itemElems);
            }
        }
    }
    else
    {
        /* Seeded per item so runs are reproducible; with batch 1 this is the
         * same sequence as the single-core runner always used. */
        for (int64_t b = 0; b < batch; b++)
        {
            std::mt19937 gen(1234u + (uint32_t)(frame * batch + b));
            std::normal_distribution<float> dist(0.0f, 1.0f);
            float *item = dst + b * itemElems;
            for (int64_t i = 0; i < itemElems; i++)
            {
                item[i] = dist(gen);
            }
        }
    }
    return 0;
}

/* One output item is [numClasses, 1, numPositions] (shape is the full
 * [batch, numClasses, 1, numPositions]); report the winning class per position
 * so the run says something more useful than "it did not crash". */
void summariseOutput(const float *src, const std::vector<int64_t> &shape)
{
    if (shape.size() != 4)
    {
        printf("[WARN] Unexpected output layout; skipping summary\n");
        return;
    }

    const int64_t numClasses   = shape[1];
    const int64_t numPositions = shape[3];

    std::vector<int64_t> histogram(numClasses, 0);

    for (int64_t p = 0; p < numPositions; p++)
    {
        int64_t best      = 0;
        float   bestScore = src[p];
        for (int64_t c = 1; c < numClasses; c++)
        {
            const float score = src[c * numPositions + p];
            if (score > bestScore)
            {
                bestScore = score;
                best      = c;
            }
        }
        histogram[best]++;
    }

    printf("Argmax class distribution over %ld positions:\n", (long)numPositions);
    for (int64_t c = 0; c < numClasses; c++)
    {
        printf("  class %ld : %6ld  (%5.1f%%)\n",
               (long)c, (long)histogram[c],
               100.0 * (double)histogram[c] / (double)numPositions);
    }
}

int32_t writeOutput(const float *src, int64_t elems, const std::string &file)
{
    std::ofstream ofs(file, std::ios::binary);
    if (!ofs)
    {
        printf("[ERROR] Could not open output file: %s\n", file.c_str());
        return -1;
    }
    ofs.write(reinterpret_cast<const char *>(src), (std::streamsize)(elems * sizeof(float)));
    if (!ofs)
    {
        printf("[ERROR] Failed writing %s\n", file.c_str());
        return -1;
    }
    printf("Output written to %s (%ld elements, float32)\n", file.c_str(), (long)elems);
    return 0;
}

double nowMs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

int32_t run(const Args &args, const ArtifactInfo &info)
{
    const bool highThroughput = args.offload &&
                                (info.inferenceMode == TIDL_inferenceModeHighThroughput);

    Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "retronet");
    Ort::SessionOptions sessionOptions;

    /* Must outlive the session: the TIDL provider keeps a pointer to it. */
    c_api_tidl_options tidlOptions;
    if (args.offload)
    {
        Ort::ThrowOnError(OrtSessionsOptionsSetDefault_Tidl(&tidlOptions));
        snprintf(tidlOptions.artifacts_folder, sizeof(tidlOptions.artifacts_folder),
                 "%s", args.artifacts.c_str());
        tidlOptions.debug_level = args.debugLevel;
        /* In high-throughput mode TIDL runs on C7x_{core_start_idx} onwards.
         * The 11.01 C API has no core_start_idx field, so it stays at its
         * default of 1 (C7x_1..C7x_N), and core_number keeps its default too. */
        tidlOptions.core_number = highThroughput ? 1 : args.core;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tidl(sessionOptions, &tidlOptions));
    }
    else
    {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, false));
    }
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.SetLogSeverityLevel(3);

    Ort::Session session(env, args.model.c_str(), sessionOptions);

    if (session.GetInputCount() != 1 || session.GetOutputCount() < 1)
    {
        printf("[ERROR] Expected 1 input and at least 1 output, got %zu and %zu\n",
               session.GetInputCount(), session.GetOutputCount());
        return -1;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr inName  = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr outName = session.GetOutputNameAllocated(0, allocator);

    Ort::TypeInfo inType  = session.GetInputTypeInfo(0);
    Ort::TypeInfo outType = session.GetOutputTypeInfo(0);
    std::vector<int64_t> inShape  = inType.GetTensorTypeAndShapeInfo().GetShape();
    std::vector<int64_t> outShape = outType.GetTensorTypeAndShapeInfo().GetShape();

    if (inType.GetTensorTypeAndShapeInfo().GetElementType()  != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        outType.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        printf("[ERROR] Expected float32 input and output\n");
        return -1;
    }
    for (size_t i = 0; i < inShape.size(); i++)  { if (inShape[i]  < 1) inShape[i]  = 1; }
    for (size_t i = 0; i < outShape.size(); i++) { if (outShape[i] < 1) outShape[i] = 1; }

    const int64_t inElems  = numElements(inShape);
    const int64_t outElems = numElements(outShape);
    const int64_t batch    = inShape.empty() ? 1 : inShape[0];
    printf("Tensors   : %s %ld floats -> %s %ld floats, batch %ld\n",
           inName.get(), (long)inElems, outName.get(), (long)outElems, (long)batch);

    if (outShape.empty() || outShape[0] != batch)
    {
        printf("[ERROR] Output batch does not match input batch %ld\n", (long)batch);
        return -1;
    }
    if (highThroughput && (batch % info.numCores) != 0)
    {
        printf("[ERROR] High-throughput artifacts for %d cores need a batch that is a multiple\n"
               "        of %d, but %s has batch %ld. Use the _b%d.onnx written by\n"
               "        compile_retronet.sh --cores %d.\n",
               info.numCores, info.numCores, args.model.c_str(), (long)batch,
               info.numCores, info.numCores);
        return -1;
    }
    if (args.offload && !highThroughput && batch > 1)
    {
        printf("[WARN] Batch %ld on single-core artifacts: all items run on C7x_%d.\n"
               "       Compile with --cores 2 to spread them over both C7x cores.\n",
               (long)batch, args.core);
    }
    const int64_t inItem  = inElems  / batch;
    const int64_t outItem = outElems / batch;
    const std::vector<int64_t> outItemShape(outShape.begin(), outShape.end());

    float *inData  = static_cast<float *>(allocMem(inElems  * sizeof(float), args.offload));
    float *outData = static_cast<float *>(allocMem(outElems * sizeof(float), args.offload));
    int32_t status = (inData != NULL && outData != NULL) ? 0 : -1;
    if (status != 0)
    {
        printf("[ERROR] Tensor allocation failed\n");
    }

    if (status == 0)
    {
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inTensor  = Ort::Value::CreateTensor<float>(memInfo, inData, inElems,
                                                               inShape.data(), inShape.size());
        Ort::Value outTensor = Ort::Value::CreateTensor<float>(memInfo, outData, outElems,
                                                               outShape.data(), outShape.size());
        Ort::IoBinding binding(session);
        binding.BindInput(inName.get(), inTensor);
        binding.BindOutput(outName.get(), outTensor);
        Ort::RunOptions runOptions;

        for (int32_t w = 0; (status == 0) && (w < args.warmup); w++)
        {
            status = fillInput(inData, batch, inItem, args.inputFile, w);
            if (status == 0)
            {
                session.Run(runOptions, binding);
            }
        }

        std::vector<double> runMs;
        runMs.reserve(args.frames);
        for (int32_t frame = 0; (status == 0) && (frame < args.frames); frame++)
        {
            /* Input generation stays outside the timed region. */
            status = fillInput(inData, batch, inItem, args.inputFile, frame);
            if (status != 0)
            {
                break;
            }

            const double t0 = nowMs();
            session.Run(runOptions, binding);
            const double t1 = nowMs();
            runMs.push_back(t1 - t0);

            if (!args.quiet)
            {
                printf("\n--- frame %d ---\n", frame);
                printf("  run            : %.2f ms (%ld item(s))\n", t1 - t0, (long)batch);
                for (int64_t b = 0; b < batch; b++)
                {
                    if (batch > 1)
                    {
                        printf("item %ld: ", (long)b);
                    }
                    summariseOutput(outData + b * outItem, outItemShape);
                }
            }
        }

        if (status == 0 && !runMs.empty())
        {
            double total = 0.0;
            for (size_t i = 0; i < runMs.size(); i++)
            {
                total += runMs[i];
            }
            const double avg = total / (double)runMs.size();
            printf("\n--- summary: %zu run(s) x batch %ld", runMs.size(), (long)batch);
            if (highThroughput)
            {
                printf(", high throughput on C7x_1..C7x_%d", info.numCores);
            }
            printf(" ---\n");
            printf("  Run() latency  : avg %.2f ms, min %.2f ms, max %.2f ms\n", avg,
                   *std::min_element(runMs.begin(), runMs.end()),
                   *std::max_element(runMs.begin(), runMs.end()));
            printf("  throughput     : %.1f items/s\n", (double)batch * runMs.size() * 1000.0 / total);
        }

        if (status == 0)
        {
            status = writeOutput(outData, outElems, args.outputFile);
        }
    }

    freeMem(inData, args.offload);
    freeMem(outData, args.offload);
    return status;
}

} /* anonymous namespace */

int main(int argc, char *argv[])
{
    Args args;
    const int32_t parsed = parseArgs(argc, argv, args);
    if (parsed != 0)
    {
        return (parsed > 0) ? 0 : -1;
    }

    printf("Model     : %s\n", args.model.c_str());
    printf("Artifacts : %s\n", args.artifacts.c_str());
    printf("Offload   : %s\n", args.offload ? "enabled" : "disabled");
    if (args.offload)
    {
        printf("Core      : C7x_%d%s\n", args.core, args.coreSet ? "" : " (default)");
    }
    printf("Input     : %s\n", args.inputFile.empty() ? "generated (seeded)"
                                                      : args.inputFile.c_str());

    ArtifactInfo info;
    if (args.offload && checkArtifacts(args.artifacts, info) != 0)
    {
        printf("\nFailed.\n");
        return -1;
    }
    if (args.offload && info.inferenceMode == TIDL_inferenceModeHighThroughput)
    {
        if (args.coreSet && args.core != 1)
        {
            printf("[ERROR] --core %d: high-throughput artifacts always start at C7x_1 "
                   "(the 11.01 C API cannot set core_start_idx)\n", args.core);
            printf("\nFailed.\n");
            return -1;
        }
        printf("Mode      : high throughput, C7x_1..C7x_%d, %d item(s) per core per Run()\n",
               info.numCores, info.numSuperBatches);
    }

    int32_t status;
    try
    {
        status = run(args, info);
    }
    catch (const Ort::Exception &e)
    {
        printf("[ERROR] ONNX Runtime: %s\n", e.what());
        status = -1;
    }

    printf("\n%s\n", (status == 0) ? "Done." : "Failed.");
    return status;
}
