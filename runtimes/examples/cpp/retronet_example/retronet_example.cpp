/*
 * Standalone runner for the retronet_v10_tidl_safe model.
 *
 * The model is not an image network: it takes a [B, 4, 1, 520] float tensor and
 * produces [B, 5, 1, 520] logits, i.e. a 5-class score for each of the 520
 * positions. There is therefore no image pre/post-processing here -- input is
 * either read verbatim from a raw float32 binary or generated, and the output
 * is written back as a raw float32 binary alongside an argmax class summary.
 *
 * High-throughput mode: artifacts compiled with inference_mode 1 and
 * num_cores N (edgeai-tidl-tools-11_01_06_00/compile_retronet.sh --cores 2)
 * run one copy of the network on each of C7x_1..C7x_N, one batch item per core.
 * TIDL needs the batch to be a multiple of N, so --model must be the batch-N
 * .onnx the compile step wrote (retronet_v10_tidl_safe_b2.onnx). Every
 * runInfer() then processes N items in parallel. The inference mode is read
 * from the artifacts' *_tidl_io_*.bin, so no extra flag is needed.
 *
 * Stacked rows: a model compiled with compile_retronet.sh --rows R takes
 * [B, 4, R, 520] and returns [B, 5, R, 520], with one independent item per
 * row. That is exact for retronet (every op works along width only) and much
 * cheaper on the C7x than batching, because TIDL pays each layer's setup once
 * per batch entry rather than once per item. This runner treats any height
 * above 1 as R items, so each runInfer() processes B x R items. Item (b, r)
 * is the strided slice x[b][:][r][:]; an --input file must already hold the
 * whole tensor in that layout, or hold a single [4, 1, 520] item.
 *
 * Usage:
 *   retronet_example --artifacts <dir> [--model <onnx>] [--input <f32.bin>]
 *                    [--output <f32.bin>] [--frames N] [--warmup N] [--quiet]
 *                    [--debug-level N] [--core N] [--no-offload]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <dirent.h>

#include "onnxrt_wrapper.h"
#include "itidl_rt.h"

using namespace onnxrt_wrapper;
using namespace dl_tensor;

namespace
{

/* eTIDL_inferenceMode (itidl_ti.h) */
const int32_t kModeDefault        = 0;
const int32_t kModeHighThroughput = 1;
const int32_t kModeLowLatency     = 2;
const int32_t kMaxCores           = 2; /* J722S: C7x_1, C7x_2 */

struct Args
{
    std::string model     = "../../data/models/retronet_v10_tidl_safe.onnx";
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

/* Leading int32 fields of sTIDL_IOBufDesc_t, which every *_tidl_io_*.bin
 * starts with: numInputBuf, numOutputBuf, numCores, numVirtualCores,
 * numSuperBatches, inferenceMode. */
struct ArtifactInfo
{
    int32_t inferenceMode   = kModeDefault;
    int32_t numCores        = 1;
    int32_t numVirtualCores = 1;
    int32_t numSuperBatches = 1;
};

const char *modeName(int32_t mode)
{
    switch (mode)
    {
        case kModeDefault:        return "default (single core)";
        case kModeHighThroughput: return "high throughput";
        case kModeLowLatency:     return "low latency";
        default:                  return "unknown";
    }
}

void usage(const char *prog)
{
    printf("Usage: %s --artifacts <dir> [options]\n\n", prog);
    printf("  --artifacts <dir>   Directory holding the compiled TIDL artifacts (required)\n");
    printf("  --model <path>      Path to the .onnx; for high-throughput artifacts, the\n");
    printf("                      batch-N copy written by compile_retronet.sh --cores N\n");
    printf("  --input <file>      Raw float32 input, one item or a whole batch; generated\n");
    printf("                      if omitted\n");
    printf("  --output <file>     Raw float32 output of the last run, whole batch\n");
    printf("                      (default: retronet_logits.bin)\n");
    printf("  --frames <N>        Number of timed runInfer() calls (default: 1); each runs\n");
    printf("                      a full batch, i.e. one item per core in high-throughput mode\n");
    printf("  --warmup <N>        Untimed runInfer() calls before the timed ones (default: 0)\n");
    printf("  --quiet             Print only the timing summary, not per-frame results\n");
    printf("  --debug-level <N>   TIDL debug_level (default: 0); 2 prints the C7x memory\n");
    printf("                      requirement table to the remote core log\n");
    printf("  --core <N>          C7x core, 1 or 2 (default: 1). Single-core artifacts only;\n");
    printf("                      high-throughput artifacts always use C7x_1..C7x_N\n");
    printf("  --no-offload        Run on ARM only, without TIDL offload\n");
    printf("  --help              Show this message\n");
}

int32_t parseArgs(int argc, char *argv[], Args &args)
{
    for (int32_t i = 1; i < argc; i++)
    {
        std::string a(argv[i]);
        bool needsValue = (a != "--no-offload" && a != "--quiet" &&
                           a != "--help" && a != "-h");

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

    if (args.artifacts.empty())
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
    if (args.core < 1 || args.core > kMaxCores)
    {
        printf("[ERROR] --core must be 1 or 2\n");
        return -1;
    }
    return 0;
}

/* Reads the inference mode and core count from every *_tidl_io_*.bin; all
 * subgraphs must agree. */
int32_t readArtifactInfo(const std::string &dir, ArtifactInfo &info)
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

        const std::string path = dir + "/" + name;
        int32_t head[6];
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.read(reinterpret_cast<char *>(head), sizeof(head)))
        {
            printf("[ERROR] Could not read %s\n", path.c_str());
            status = -1;
            continue;
        }

        ArtifactInfo cur;
        cur.numCores        = head[2];
        cur.numVirtualCores = head[3];
        cur.numSuperBatches = head[4];
        cur.inferenceMode   = head[5];
        printf("Artifact  : %s mode %d %s, %d core(s)\n", name.c_str(),
               cur.inferenceMode, modeName(cur.inferenceMode), cur.numCores);

        if (found > 0 && (cur.inferenceMode != info.inferenceMode ||
                          cur.numCores      != info.numCores))
        {
            printf("[ERROR] %s was compiled for a different inference mode or core "
                   "count than the other subgraphs\n", name.c_str());
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
    if (status == 0 && info.inferenceMode == kModeLowLatency)
    {
        printf("[ERROR] Low-latency (inference_mode 2) artifacts are not supported here\n");
        status = -1;
    }
    if (status == 0 && (info.numCores < 1 || info.numCores > kMaxCores))
    {
        printf("[ERROR] Artifacts ask for %d cores, J722S has %d\n", info.numCores, kMaxCores);
        status = -1;
    }
    return status;
}

/* Mirrors the allocation strategy of the shipped examples: shared memory when
 * TIDL offload is active, plain aligned memory otherwise. */
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

int32_t allocTensors(const std::vector<DlTensor> *detail,
                     std::vector<DlTensor>       &tensors,
                     std::vector<DlTensor *>     &ptrs,
                     bool                         offload)
{
    for (size_t i = 0; i < detail->size(); i++)
    {
        const DlTensor *info = &(detail->at(i));
        ptrs[i] = &tensors[i];

        ptrs[i]->name      = info->name;
        ptrs[i]->allocSize = info->allocSize;
        ptrs[i]->validSize = info->validSize;
        ptrs[i]->shape     = info->shape;
        ptrs[i]->numDim    = info->numDim;
        ptrs[i]->type      = info->type;
        ptrs[i]->numElem   = info->numElem;
        ptrs[i]->elemSize  = info->elemSize;
        ptrs[i]->typeName  = info->typeName;

        ptrs[i]->data = allocMem((size_t)info->allocSize, offload);
        if (ptrs[i]->data == NULL)
        {
            printf("[ERROR] Allocation of %ld bytes failed for '%s'\n",
                   (long)info->allocSize, info->name);
            return -1;
        }
        memset(ptrs[i]->data, 0, (size_t)info->allocSize);
    }
    return 0;
}

int64_t batchOf(const DlTensor *tensor)
{
    return (tensor->numDim > 0 && tensor->shape[0] > 0) ? tensor->shape[0] : 1;
}

/* Items stacked along height (compile_retronet.sh --rows); 1 for the original
 * height-1 model. */
int64_t rowsOf(const DlTensor *tensor)
{
    return (tensor->numDim == 4 && tensor->shape[2] > 0) ? tensor->shape[2] : 1;
}

/* Copies one [C, 1, W] item into row r of batch entry b of a [B, C, R, W]
 * tensor. */
void scatterItem(float *dst, const float *item, const DlTensor *tensor, int64_t b, int64_t r)
{
    const int64_t channels = tensor->shape[1];
    const int64_t rows     = tensor->shape[2];
    const int64_t width    = tensor->shape[3];
    for (int64_t c = 0; c < channels; c++)
    {
        std::copy(item + c * width, item + (c + 1) * width,
                  dst + ((b * channels + c) * rows + r) * width);
    }
}

/* Fills the input tensor, either from a raw float32 file or with deterministic
 * pseudo-random values. The file holds either the whole tensor or one item,
 * which is then copied to every item slot. Returns 0 on success. */
int32_t fillInput(DlTensor *tensor, const std::string &file, int32_t frame)
{
    const int64_t elems     = tensor->numElem;
    const int64_t batch     = batchOf(tensor);
    const int64_t rows      = rowsOf(tensor);
    const int64_t items     = batch * rows;
    const int64_t itemElems = elems / items;

    if (tensor->elemSize != (int32_t)sizeof(float))
    {
        printf("[ERROR] Expected a float32 input, got '%s'\n", tensor->typeName.c_str());
        return -1;
    }
    if (rows > 1 && tensor->numDim != 4)
    {
        printf("[ERROR] Expected a 4-D input\n");
        return -1;
    }

    float *dst = static_cast<float *>(tensor->data);

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
            printf("[ERROR] %s holds %ld bytes, model expects %ld (%ld items) "
                   "or %ld (one item)\n", file.c_str(), (long)bytes, (long)batchSize,
                   (long)items, (long)itemSize);
            return -1;
        }

        ifs.seekg(0);
        ifs.read(reinterpret_cast<char *>(dst), bytes);
        if (!ifs)
        {
            printf("[ERROR] Short read from %s\n", file.c_str());
            return -1;
        }
        if (bytes == itemSize && items > 1)
        {
            const std::vector<float> item(dst, dst + itemElems);
            for (int64_t b = 0; b < batch; b++)
            {
                for (int64_t r = 0; r < rows; r++)
                {
                    scatterItem(dst, item.data(), tensor, b, r);
                }
            }
        }
    }
    else
    {
        /* Seeded per item so runs are reproducible. Item n of a frame gets the
         * same values whether it sits in the batch axis or in a row, so a
         * --rows R model and a --batch R model see identical inputs; with one
         * item this is the sequence the single-core runner always used. */
        std::vector<float> item(itemElems);
        for (int64_t n = 0; n < items; n++)
        {
            std::mt19937 gen(1234u + (uint32_t)(frame * items + n));
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (int64_t i = 0; i < itemElems; i++)
            {
                item[i] = dist(gen);
            }
            if (rows > 1)
            {
                scatterItem(dst, item.data(), tensor, n / rows, n % rows);
            }
            else
            {
                std::copy(item.begin(), item.end(), dst + n * itemElems);
            }
        }
    }
    return 0;
}

/* The output is [B, numClasses, R, numPositions]; report the winning class per
 * position for each item so the run says something more useful than "it did
 * not crash". */
void summariseOutput(const DlTensor *tensor)
{
    if (tensor->numDim != 4 || tensor->elemSize != (int32_t)sizeof(float))
    {
        printf("[WARN] Unexpected output layout; skipping summary\n");
        return;
    }

    const int64_t batch        = batchOf(tensor);
    const int64_t rows         = rowsOf(tensor);
    const int64_t numClasses   = tensor->shape[1];
    const int64_t numPositions = tensor->shape[3];
    const int64_t classStride  = rows * numPositions;

    for (int64_t n = 0; n < batch * rows; n++)
    {
        const int64_t b = n / rows;
        const int64_t r = n % rows;
        const float *src = static_cast<const float *>(tensor->data) +
                           (b * numClasses * rows + r) * numPositions;
        std::vector<int64_t> histogram(numClasses, 0);

        for (int64_t p = 0; p < numPositions; p++)
        {
            int64_t best      = 0;
            float   bestScore = src[p];
            for (int64_t c = 1; c < numClasses; c++)
            {
                const float score = src[c * classStride + p];
                if (score > bestScore)
                {
                    bestScore = score;
                    best      = c;
                }
            }
            histogram[best]++;
        }

        if (batch * rows > 1)
        {
            printf("item %ld: ", (long)n);
        }
        printf("Argmax class distribution over %ld positions:\n", (long)numPositions);
        for (int64_t c = 0; c < numClasses; c++)
        {
            printf("  class %ld : %6ld  (%5.1f%%)\n",
                   (long)c, (long)histogram[c],
                   100.0 * (double)histogram[c] / (double)numPositions);
        }
    }
}

int32_t writeOutput(const DlTensor *tensor, const std::string &file)
{
    std::ofstream ofs(file, std::ios::binary);
    if (!ofs)
    {
        printf("[ERROR] Could not open output file: %s\n", file.c_str());
        return -1;
    }
    ofs.write(static_cast<const char *>(tensor->data),
              (std::streamsize)(tensor->numElem * tensor->elemSize));
    if (!ofs)
    {
        printf("[ERROR] Failed writing %s\n", file.c_str());
        return -1;
    }
    printf("Output written to %s (%ld elements, float32)\n",
           file.c_str(), (long)tensor->numElem);
    return 0;
}

double nowMs()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
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
    printf("Input     : %s\n", args.inputFile.empty() ? "generated (seeded)"
                                                      : args.inputFile.c_str());

    ArtifactInfo info;
    if (args.offload && readArtifactInfo(args.artifacts, info) != 0)
    {
        printf("\nFailed.\n");
        return -1;
    }
    const bool highThroughput = args.offload && (info.inferenceMode == kModeHighThroughput);

    std::map<std::string, std::string> options;
    options["artifacts_folder"] = args.artifacts;
    options["debug_level"]      = std::to_string(args.debugLevel);

    if (highThroughput)
    {
        /* TIDL runs high-throughput artifacts on C7x_{core_start_idx} onwards.
         * The ORT C API's c_api_tidl_options has no core_start_idx field, so it
         * stays at its default of 1: C7x_1..C7x_N. core_number is not set. */
        if (args.coreSet && args.core != 1)
        {
            printf("[ERROR] --core %d: high-throughput artifacts always start at C7x_1\n",
                   args.core);
            printf("\nFailed.\n");
            return -1;
        }
        printf("Mode      : high throughput on C7x_1..C7x_%d\n", info.numCores);
    }
    else if (args.offload)
    {
        options["core_number"] = std::to_string(args.core);
        printf("Core      : C7x_%d\n", args.core);
    }

    ONNXRT session(args.model, args.offload);

    if (session.createInfer(options) != 0)
    {
        printf("[ERROR] createInfer failed\n");
        return -1;
    }
    session.dumpInfo();

    const std::vector<DlTensor> *inputInfo  = session.getInputDetails();
    const std::vector<DlTensor> *outputInfo = session.getOutputDetails();

    if (inputInfo->empty() || outputInfo->empty())
    {
        printf("[ERROR] Model has no inputs or no outputs\n");
        return -1;
    }

    const int64_t batch = batchOf(&inputInfo->at(0));
    const int64_t rows  = rowsOf(&inputInfo->at(0));
    const int64_t items = batch * rows;
    if (batchOf(&outputInfo->at(0)) != batch || rowsOf(&outputInfo->at(0)) != rows)
    {
        printf("[ERROR] Output batch/rows %ld/%ld do not match input %ld/%ld\n",
               (long)batchOf(&outputInfo->at(0)), (long)rowsOf(&outputInfo->at(0)),
               (long)batch, (long)rows);
        return -1;
    }
    if (rows > 1)
    {
        printf("Items     : %ld per run (batch %ld x %ld stacked rows)\n",
               (long)items, (long)batch, (long)rows);
    }
    if (highThroughput && (batch % info.numCores) != 0)
    {
        printf("[ERROR] High-throughput artifacts for %d cores need a batch that is a\n"
               "        multiple of %d, but %s has batch %ld. Use the _b%d.onnx written\n"
               "        by compile_retronet.sh --cores %d.\n",
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

    std::vector<DlTensor>   inputs(inputInfo->size());
    std::vector<DlTensor *> inputPtrs(inputInfo->size(), nullptr);
    std::vector<DlTensor>   outputs(outputInfo->size());
    std::vector<DlTensor *> outputPtrs(outputInfo->size(), nullptr);

    int32_t status = allocTensors(inputInfo, inputs, inputPtrs, args.offload);
    if (status == 0)
    {
        status = allocTensors(outputInfo, outputs, outputPtrs, args.offload);
    }

    for (int32_t w = 0; (status == 0) && (w < args.warmup); w++)
    {
        for (size_t i = 0; (status == 0) && (i < inputPtrs.size()); i++)
        {
            status = fillInput(inputPtrs[i], args.inputFile, w);
        }
        if (status == 0 && session.runInfer(inputPtrs, outputPtrs) != 0)
        {
            printf("[ERROR] runInfer failed on warm-up run %d\n", w);
            status = -1;
        }
    }

    std::vector<double> runMs;
    runMs.reserve(args.frames);
    for (int32_t frame = 0; (status == 0) && (frame < args.frames); frame++)
    {
        /* Input generation stays outside the timed region. */
        for (size_t i = 0; (status == 0) && (i < inputPtrs.size()); i++)
        {
            status = fillInput(inputPtrs[i], args.inputFile, frame);
        }
        if (status != 0)
        {
            break;
        }

        const double t0 = nowMs();
        status = session.runInfer(inputPtrs, outputPtrs);
        const double t1 = nowMs();
        if (status != 0)
        {
            printf("[ERROR] runInfer failed on frame %d\n", frame);
            break;
        }
        runMs.push_back(t1 - t0);

        if (args.quiet)
        {
            continue;
        }

        printf("\n--- frame %d ---\n", frame);
        printf("  run            : %.2f ms (%ld item(s))\n", t1 - t0, (long)items);

        const std::map<std::string, std::pair<float, std::string>> perf =
            session.getPerformance();
        if (perf.empty())
        {
            printf("Performance counters unavailable "
                   "(built with ORT_TI_BENCHMARK_API=OFF)\n");
        }
        else
        {
            for (std::map<std::string, std::pair<float, std::string>>::const_iterator
                     it = perf.begin(); it != perf.end(); ++it)
            {
                printf("  %-15s: %.2f %s\n", it->first.c_str(),
                       it->second.first, it->second.second.c_str());
            }
        }

        summariseOutput(outputPtrs[0]);
    }

    if ((status == 0) && !runMs.empty())
    {
        double total = 0.0;
        for (size_t i = 0; i < runMs.size(); i++)
        {
            total += runMs[i];
        }
        printf("\n--- summary: %zu run(s) x batch %ld", runMs.size(), (long)batch);
        if (rows > 1)
        {
            printf(" x %ld rows", (long)rows);
        }
        if (highThroughput)
        {
            printf(", high throughput on C7x_1..C7x_%d", info.numCores);
        }
        printf(" ---\n");
        printf("  runInfer       : avg %.2f ms, min %.2f ms, max %.2f ms\n",
               total / (double)runMs.size(),
               *std::min_element(runMs.begin(), runMs.end()),
               *std::max_element(runMs.begin(), runMs.end()));
        printf("  throughput     : %.1f items/s\n",
               (double)items * (double)runMs.size() * 1000.0 / total);

        status = writeOutput(outputPtrs[0], args.outputFile);
    }

    for (size_t i = 0; i < inputPtrs.size(); i++)
    {
        if (inputPtrs[i]) freeMem(inputPtrs[i]->data, args.offload);
    }
    for (size_t i = 0; i < outputPtrs.size(); i++)
    {
        if (outputPtrs[i]) freeMem(outputPtrs[i]->data, args.offload);
    }

    printf("\n%s\n", (status == 0) ? "Done." : "Failed.");
    return status;
}
