// SPDX-License-Identifier: BSD-3-Clause
//
// Standalone raw temporal-patch runner for the Qualcomm GenieX v0.3.16
// Qwen3-VL QAIRT backend. The implementation intentionally bypasses image
// decoding: pixel_values is already packed by prepare_video_npu_inputs.py.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "llm/llm_utils.h"
#include "qwen3_vl/qwen3_vl.h"
#include "types.h"
#include "vlm/vlm_types.h"

namespace {

constexpr int32_t kVisionStartTokenId = 151652;
constexpr int32_t kVisionEndTokenId   = 151653;
constexpr int32_t kImagePadTokenId    = 151655;
constexpr size_t  kVisualTokenCount   = 256;
constexpr size_t  kPixelRows          = 1024;
constexpr size_t  kPixelColumns       = 1536;
constexpr size_t  kPixelFloatCount    = kPixelRows * kPixelColumns;
constexpr size_t  kPixelByteCount     = kPixelFloatCount * sizeof(float);

struct Args {
    std::filesystem::path model_dir;
    std::filesystem::path pixel_values;
    std::filesystem::path prefix_file;
    std::filesystem::path suffix_file;
    size_t                expected_prompt_tokens = 0;
    int32_t               max_tokens             = 64;
    bool                  verbose                = false;
};

[[noreturn]] void usageError(const std::string& message) {
    throw std::runtime_error(
        message +
        "\nUsage: cosmos_geniex_raw_video"
        " --model-dir DIR"
        " --pixel-values pair.raw"
        " --prefix-file prefix.txt"
        " --suffix-file suffix.txt"
        " --expected-prompt-tokens N"
        " [--max-tokens N] [--verbose]");
}

size_t parsePositiveSize(const std::string& value, const std::string& flag) {
    if (value.empty() || value[0] == '-') usageError(flag + " must be positive");
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed, 10);
    } catch (...) {
        usageError(flag + " must be an integer");
    }
    if (consumed != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max()) {
        usageError(flag + " must be a positive integer");
    }
    return static_cast<size_t>(parsed);
}

Args parseArgs(int argc, char** argv) {
    Args args;
    auto next = [&](int& index, const std::string& flag) -> std::string {
        if (index + 1 >= argc) usageError("Missing value for " + flag);
        return argv[++index];
    };
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        if (flag == "--model-dir") {
            args.model_dir = next(index, flag);
        } else if (flag == "--pixel-values") {
            args.pixel_values = next(index, flag);
        } else if (flag == "--prefix-file") {
            args.prefix_file = next(index, flag);
        } else if (flag == "--suffix-file") {
            args.suffix_file = next(index, flag);
        } else if (flag == "--expected-prompt-tokens") {
            args.expected_prompt_tokens =
                parsePositiveSize(next(index, flag), flag);
        } else if (flag == "--max-tokens") {
            const size_t value = parsePositiveSize(next(index, flag), flag);
            if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                usageError("--max-tokens is too large");
            }
            args.max_tokens = static_cast<int32_t>(value);
        } else if (flag == "--verbose") {
            args.verbose = true;
        } else if (flag == "--help" || flag == "-h") {
            usageError("Raw two-frame Qwen3-VL temporal-patch runner.");
        } else {
            usageError("Unknown argument: " + flag);
        }
    }
    if (args.model_dir.empty() || args.pixel_values.empty() ||
        args.prefix_file.empty() || args.suffix_file.empty() ||
        args.expected_prompt_tokens == 0) {
        usageError("All required arguments must be supplied");
    }
    return args;
}

void requireFile(const std::filesystem::path& path, const std::string& label) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(label + " is not a regular file: " +
                                 path.string());
    }
}

std::string readText(const std::filesystem::path& path,
                     const std::string& label) {
    requireFile(path, label);
    const auto bytes = std::filesystem::file_size(path);
    if (bytes == 0 || bytes > 1024 * 1024) {
        throw std::runtime_error(label + " has an unsafe byte count: " +
                                 std::to_string(bytes));
    }
    std::ifstream stream(path, std::ios::binary);
    std::string text(static_cast<size_t>(bytes), '\0');
    stream.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream || text.find('\0') != std::string::npos) {
        throw std::runtime_error("Cannot read a NUL-free " + label);
    }
    return text;
}

std::vector<float> readPixels(const std::filesystem::path& path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "Raw pixel tensor requires a little-endian target");
    }
    requireFile(path, "pixel tensor");
    const auto bytes = std::filesystem::file_size(path);
    if (bytes != kPixelByteCount) {
        throw std::runtime_error(
            "pixel tensor has " + std::to_string(bytes) +
            " bytes; expected " + std::to_string(kPixelByteCount) +
            " for float32[1024,1536]");
    }
    std::vector<float> values(kPixelFloatCount);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(kPixelByteCount));
    if (!stream) throw std::runtime_error("Cannot read the pixel tensor");
    const auto invalid =
        std::find_if(values.begin(), values.end(),
                     [](float value) { return !std::isfinite(value); });
    if (invalid != values.end()) {
        throw std::runtime_error("pixel tensor contains NaN or infinity");
    }
    return values;
}

void requireBundleFiles(const std::filesystem::path& model_dir) {
    if (!std::filesystem::is_directory(model_dir)) {
        throw std::runtime_error("model directory does not exist: " +
                                 model_dir.string());
    }
    for (const char* name : {
             "metadata.json",
             "genie_config.json",
             "htp_backend_ext_config.json",
             "tokenizer.json",
             "embedding_weights.raw",
             "vision_encoder.bin",
             "part1_of_4.bin",
             "part2_of_4.bin",
             "part3_of_4.bin",
             "part4_of_4.bin",
         }) {
        requireFile(model_dir / name, std::string("bundle file ") + name);
    }
}

size_t countToken(const std::vector<int32_t>& tokens, int32_t target) {
    return static_cast<size_t>(
        std::count(tokens.begin(), tokens.end(), target));
}

std::vector<int32_t> buildPrompt(
    const geniex::Tokenizer& tokenizer,
    const std::string& prefix,
    const std::string& suffix,
    size_t expected_prompt_tokens) {
    auto prefix_tokens = tokenizer.encode(prefix, false);
    auto suffix_tokens = tokenizer.encode(suffix, false);
    const auto pad_probe = tokenizer.encode("<|image_pad|>", false);

    if (prefix_tokens.empty() ||
        prefix_tokens.back() != kVisionStartTokenId ||
        countToken(prefix_tokens, kVisionStartTokenId) != 1) {
        throw std::runtime_error(
            "prefix must tokenize to exactly one trailing <|vision_start|>");
    }
    if (suffix_tokens.empty() ||
        suffix_tokens.front() != kVisionEndTokenId ||
        countToken(suffix_tokens, kVisionEndTokenId) != 1) {
        throw std::runtime_error(
            "suffix must tokenize to exactly one leading <|vision_end|>");
    }
    if (countToken(prefix_tokens, kImagePadTokenId) != 0 ||
        countToken(suffix_tokens, kImagePadTokenId) != 0) {
        throw std::runtime_error(
            "text chunks must not contain embedded image-pad tokens");
    }
    if (pad_probe.size() != 1 || pad_probe.front() != kImagePadTokenId) {
        throw std::runtime_error(
            "tokenizer does not map <|image_pad|> to Qwen3-VL token 151655");
    }

    std::vector<int32_t> prompt;
    prompt.reserve(prefix_tokens.size() + kVisualTokenCount +
                   suffix_tokens.size());
    prompt.insert(prompt.end(), prefix_tokens.begin(), prefix_tokens.end());
    prompt.insert(prompt.end(), kVisualTokenCount, kImagePadTokenId);
    prompt.insert(prompt.end(), suffix_tokens.begin(), suffix_tokens.end());

    if (prompt.size() != expected_prompt_tokens) {
        throw std::runtime_error(
            "tokenized prompt has " + std::to_string(prompt.size()) +
            " tokens; the hash-bound run manifest expects " +
            std::to_string(expected_prompt_tokens));
    }
    if (countToken(prompt, kImagePadTokenId) != kVisualTokenCount) {
        throw std::runtime_error(
            "prompt must contain exactly 256 image-pad tokens");
    }
    return prompt;
}

geniex::VLMConfig modelConfig(const std::filesystem::path& model_dir) {
    geniex::VLMConfig config;
    config.llm_config.model_paths = {
        (model_dir / "part1_of_4.bin").string(),
        (model_dir / "part2_of_4.bin").string(),
        (model_dir / "part3_of_4.bin").string(),
        (model_dir / "part4_of_4.bin").string(),
    };
    config.llm_config.tokenizer_path =
        (model_dir / "tokenizer.json").string();
    config.llm_config.htp_config_path =
        (model_dir / "htp_backend_ext_config.json").string();
    config.llm_config.embedding_path =
        (model_dir / "embedding_weights.raw").string();

    config.vision_config.model_paths = {
        (model_dir / "vision_encoder.bin").string(),
    };
    config.vision_config.htp_config_path =
        (model_dir / "htp_backend_ext_config.json").string();
    return config;
}

void requirePatchedTensorClassifier() {
    if (!geniex::isSpecialTensor("visual_pos_masks") ||
        !geniex::isSpecialTensor("deepstack_visual_embeds_0") ||
        !geniex::isSpecialTensor("deepstack_visual_embeds_17")) {
        throw std::runtime_error(
            "linked GenieX lacks the full-DeepStack isSpecialTensor patch; "
            "refusing a potentially miswired first decoder shard");
    }
    if (geniex::isSpecialTensor("inputs_embeds")) {
        throw std::runtime_error(
            "linked GenieX incorrectly classifies inputs_embeds as auxiliary");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        requirePatchedTensorClassifier();
        requireBundleFiles(args.model_dir);

        const std::string prefix = readText(args.prefix_file, "prefix file");
        const std::string suffix = readText(args.suffix_file, "suffix file");
        auto tokenizer =
            geniex::Tokenizer::from_file((args.model_dir / "tokenizer.json").string());
        if (!tokenizer) throw std::runtime_error("Failed to load tokenizer");
        const auto prompt = buildPrompt(
            *tokenizer, prefix, suffix, args.expected_prompt_tokens);
        auto pixels = readPixels(args.pixel_values);

        geniex::PixelData pixel_data;
        pixel_data.pixel_values = std::move(pixels);
        // temporal_patch_size=2 fuses the two source frames into T=1.
        pixel_data.image_grid_thw = {{1, 32, 32}};
        geniex::VLMInput input;
        input.pixel_data = std::move(pixel_data);

        std::cout
            << "Validated raw-video contract: float32[1024,1536], grid "
               "[1,32,32], 256 visual tokens, patched DeepStack classifier.\n"
            << "Loading full-DeepStack QAIRT contexts on QnnHtp...\n";

        geniex::QnnRuntimeConfig runtime;
        auto model = geniex::qwen3_vl::makeModel(runtime, modelConfig(args.model_dir));
        if (!model) throw std::runtime_error("Failed to initialize Qwen3-VL model");

        geniex::GenerationConfig generation;
        generation.max_tokens     = args.max_tokens;
        generation.enable_sampling = false;

        const auto start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point first_token;
        bool got_first = false;
        std::string answer;
        std::cout << "COSMOS_GENIEX_OUTPUT_BEGIN\n";
        auto output_tokens = model->generate(
            prompt, input, generation, [&](int32_t token) {
                if (!got_first) {
                    first_token = std::chrono::steady_clock::now();
                    got_first = true;
                }
                const std::string piece = tokenizer->decode_token(token);
                answer += piece;
                std::cout << piece << std::flush;
                return true;
            });
        const auto end = std::chrono::steady_clock::now();
        std::cout << "\nCOSMOS_GENIEX_OUTPUT_END\n";

        if (!got_first) {
            throw std::runtime_error("Generation returned no output token");
        }
        const double ttft_ms =
            std::chrono::duration<double, std::milli>(first_token - start).count();
        const double decode_ms =
            std::chrono::duration<double, std::milli>(end - first_token).count();
        const size_t decode_tokens =
            output_tokens.size() > 1 ? output_tokens.size() - 1 : 0;
        const double tokens_per_second =
            decode_ms > 0.0 ? decode_tokens / (decode_ms / 1000.0) : 0.0;
        if (args.verbose) {
            std::cout << "Prompt tokens: " << prompt.size() << "\n"
                      << "Generated tokens: " << output_tokens.size() << "\n"
                      << "TTFT (vision + prefill): " << std::fixed
                      << std::setprecision(1) << ttft_ms << " ms\n"
                      << "Decode: " << std::setprecision(2)
                      << tokens_per_second << " tokens/s\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
