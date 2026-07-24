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
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "geniex-proc/tokenizer.h"
#include "llm/llm_spec_loader.h"
#include "llm/llm_utils.h"
#include "qwen3_vl/qwen3_vl.h"
#include "types.h"
#include "utils/detail/json.hpp"
#include "vlm/vlm_types.h"

namespace {

constexpr int32_t kVisionStartTokenId = 151652;
constexpr int32_t kVisionEndTokenId   = 151653;
constexpr int32_t kImagePadTokenId    = 151655;
constexpr int32_t kVideoPadTokenId    = 151656;
// The deployed QAIRT text contexts expose AR128 prefill and AR1 decode
// graphs. Pinned GenieX v0.3.16 cannot safely merge a prompt longer than the
// CL-AR prefill KV input into the differently-strided AR1 cache.
constexpr size_t kTextPrefillAr = 128;
constexpr size_t kTextDecodeAr  = 1;

struct Args {
    std::filesystem::path model_dir;
    std::vector<std::filesystem::path> pixel_values;
    std::vector<std::filesystem::path> text_chunks;
    size_t                expected_prompt_tokens = 0;
    int32_t               max_tokens             = 64;
    bool                  verbose                = false;
};

[[noreturn]] void usageError(const std::string& message) {
    throw std::runtime_error(
        message +
        "\nUsage: cosmos_geniex_raw_video"
        " --model-dir DIR"
        " --pixel-values pair_000.raw [--pixel-values pair_001.raw ...]"
        " --text-chunk prefix.txt [--text-chunk bridge.txt ...]"
        " --text-chunk suffix.txt"
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
            args.pixel_values.emplace_back(next(index, flag));
        } else if (flag == "--text-chunk") {
            args.text_chunks.emplace_back(next(index, flag));
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
        args.text_chunks.size() != args.pixel_values.size() + 1 ||
        args.expected_prompt_tokens == 0) {
        usageError(
            "Supply a model, one or more pixel tensors, exactly one more "
            "ordered text chunk than pixel tensors, and the prompt size");
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

struct VisionContract {
    size_t image_width;
    size_t image_height;
    size_t patch_size;
    size_t temporal_patch_size;
    size_t spatial_merge_size;
    size_t grid_height;
    size_t grid_width;
    size_t pixel_rows;
    size_t pixel_columns;
    size_t pixel_float_count;
    size_t pixel_byte_count;
    size_t visual_tokens;
};

struct TextRuntimeContract {
    size_t context_size;
    size_t prefill_ar;
    size_t decode_ar;
    size_t prefill_kv_capacity;
};

TextRuntimeContract textRuntimeContract(
    const std::filesystem::path& model_dir) {
    const auto path = model_dir / "genie_config.json";
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "cannot open text runtime config: " + path.string());
    }
    size_t context_size = 0;
    try {
        const auto config = qualla::json::parse(stream);
        const auto value =
            config.at("dialog").at("context").at("size").get<int64_t>();
        if (value <= 0) {
            throw std::runtime_error("context size is not positive");
        }
        context_size = static_cast<size_t>(value);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            "cannot read dialog.context.size from " + path.string() +
            ": " + error.what());
    }
    if (context_size <= kTextPrefillAr) {
        throw std::runtime_error(
            "text context CL" + std::to_string(context_size) +
            " cannot host the fixed AR" +
            std::to_string(kTextPrefillAr) + " prefill graph");
    }
    return {
        context_size,
        kTextPrefillAr,
        kTextDecodeAr,
        context_size - kTextPrefillAr,
    };
}

VisionContract visionContract(const std::filesystem::path& model_dir) {
    const auto metadata = geniex::parseQAIRTMetadata(model_dir);
    if (!metadata.vision_preprocessing) {
        throw std::runtime_error(
            "bundle metadata has no vision_preprocessing block");
    }
    const auto& profile = *metadata.vision_preprocessing;
    const auto positive = [](int value, const std::string& name) -> size_t {
        if (value <= 0) {
            throw std::runtime_error(name + " must be positive");
        }
        return static_cast<size_t>(value);
    };
    VisionContract result{};
    result.image_width =
        positive(profile.image_width, "vision image_width");
    result.image_height =
        positive(profile.image_height, "vision image_height");
    result.patch_size = positive(profile.patch_size, "vision patch_size");
    result.temporal_patch_size = positive(
        profile.temporal_patch_size, "vision temporal_patch_size");
    result.spatial_merge_size = positive(
        profile.spatial_merge_size, "vision spatial_merge_size");
    if (result.temporal_patch_size != 2) {
        throw std::runtime_error(
            "raw paired-frame runner requires temporal_patch_size=2");
    }
    if (result.image_width % result.patch_size != 0 ||
        result.image_height % result.patch_size != 0) {
        throw std::runtime_error(
            "vision image dimensions are not divisible by patch_size");
    }
    result.grid_height = result.image_height / result.patch_size;
    result.grid_width = result.image_width / result.patch_size;
    if (result.grid_height % result.spatial_merge_size != 0 ||
        result.grid_width % result.spatial_merge_size != 0) {
        throw std::runtime_error(
            "vision patch grid is not divisible by spatial_merge_size");
    }
    result.pixel_rows = result.grid_height * result.grid_width;
    result.pixel_columns =
        3 * result.temporal_patch_size * result.patch_size *
        result.patch_size;
    result.pixel_float_count = result.pixel_rows * result.pixel_columns;
    result.pixel_byte_count = result.pixel_float_count * sizeof(float);
    result.visual_tokens =
        result.pixel_rows /
        (result.spatial_merge_size * result.spatial_merge_size);
    return result;
}

std::vector<float> readPixels(
    const std::filesystem::path& path,
    const VisionContract& profile) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "Raw pixel tensor requires a little-endian target");
    }
    requireFile(path, "pixel tensor");
    const auto bytes = std::filesystem::file_size(path);
    if (bytes != profile.pixel_byte_count) {
        throw std::runtime_error(
            "pixel tensor has " + std::to_string(bytes) +
            " bytes; expected " +
            std::to_string(profile.pixel_byte_count) + " for float32[" +
            std::to_string(profile.pixel_rows) + "," +
            std::to_string(profile.pixel_columns) + "]");
    }
    std::vector<float> values(profile.pixel_float_count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(profile.pixel_byte_count));
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
    const std::vector<std::string>& chunks,
    size_t pair_count,
    size_t visual_tokens_per_pair,
    size_t expected_prompt_tokens) {
    const auto pad_probe = tokenizer.encode("<|video_pad|>", false);
    if (pad_probe.size() != 1 || pad_probe.front() != kVideoPadTokenId) {
        throw std::runtime_error(
            "tokenizer does not map <|video_pad|> to Qwen3-VL token 151656");
    }
    if (pair_count == 0 || chunks.size() != pair_count + 1 ||
        visual_tokens_per_pair == 0) {
        throw std::runtime_error("invalid pair/text-chunk prompt contract");
    }

    std::vector<int32_t> prompt;
    prompt.reserve(expected_prompt_tokens);
    for (size_t pair = 0; pair < pair_count; ++pair) {
        auto tokens = tokenizer.encode(chunks[pair], false);
        if (tokens.empty() ||
            tokens.back() != kVisionStartTokenId ||
            countToken(tokens, kVisionStartTokenId) != 1 ||
            countToken(tokens, kImagePadTokenId) != 0 ||
            countToken(tokens, kVideoPadTokenId) != 0) {
            throw std::runtime_error(
                "prefix/bridge must tokenize to exactly one trailing "
                "<|vision_start|> and contain no image-pad token");
        }
        if (pair == 0) {
            if (countToken(tokens, kVisionEndTokenId) != 0) {
                throw std::runtime_error(
                    "prefix must not contain <|vision_end|>");
            }
        } else if (
            tokens.front() != kVisionEndTokenId ||
            countToken(tokens, kVisionEndTokenId) != 1) {
            throw std::runtime_error(
                "bridge must tokenize to exactly one leading "
                "<|vision_end|>");
        }
        prompt.insert(prompt.end(), tokens.begin(), tokens.end());
        prompt.insert(
            prompt.end(), visual_tokens_per_pair, kVideoPadTokenId);
    }

    auto suffix_tokens = tokenizer.encode(chunks.back(), false);
    if (suffix_tokens.empty() ||
        suffix_tokens.front() != kVisionEndTokenId ||
        countToken(suffix_tokens, kVisionEndTokenId) != 1 ||
        countToken(suffix_tokens, kVisionStartTokenId) != 0 ||
        countToken(suffix_tokens, kImagePadTokenId) != 0 ||
        countToken(suffix_tokens, kVideoPadTokenId) != 0) {
        throw std::runtime_error(
            "suffix must tokenize to exactly one leading <|vision_end|> "
            "and contain no vision-start/image-pad token");
    }
    prompt.insert(
        prompt.end(), suffix_tokens.begin(), suffix_tokens.end());

    if (prompt.size() != expected_prompt_tokens) {
        throw std::runtime_error(
            "tokenized prompt has " + std::to_string(prompt.size()) +
            " tokens; the hash-bound run manifest expects " +
            std::to_string(expected_prompt_tokens));
    }
    const size_t total_visual_tokens =
        pair_count * visual_tokens_per_pair;
    if (countToken(prompt, kVideoPadTokenId) != total_visual_tokens) {
        throw std::runtime_error(
            "prompt video-pad token count does not match the profile/pairs");
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

        const auto profile = visionContract(args.model_dir);
        const auto text_runtime = textRuntimeContract(args.model_dir);
        std::vector<std::string> chunks;
        chunks.reserve(args.text_chunks.size());
        for (size_t index = 0; index < args.text_chunks.size(); ++index) {
            chunks.push_back(
                readText(
                    args.text_chunks[index],
                    "text chunk " + std::to_string(index)));
        }
        auto tokenizer =
            geniex::Tokenizer::from_file((args.model_dir / "tokenizer.json").string());
        if (!tokenizer) throw std::runtime_error("Failed to load tokenizer");
        const auto prompt = buildPrompt(
            *tokenizer,
            chunks,
            args.pixel_values.size(),
            profile.visual_tokens,
            args.expected_prompt_tokens);
        if (prompt.size() > text_runtime.prefill_kv_capacity) {
            throw std::runtime_error(
                "prompt has " + std::to_string(prompt.size()) +
                " tokens, exceeding the safe AR" +
                std::to_string(text_runtime.prefill_ar) + "/CL" +
                std::to_string(text_runtime.context_size) +
                " prefill cache limit of " +
                std::to_string(text_runtime.prefill_kv_capacity) +
                "; reduce the temporal-pair count");
        }
        if (prompt.size() + static_cast<size_t>(args.max_tokens) >
            text_runtime.context_size) {
            throw std::runtime_error(
                "prompt plus generation exceeds CL" +
                std::to_string(text_runtime.context_size));
        }
        std::vector<float> pixels;
        pixels.reserve(
            args.pixel_values.size() * profile.pixel_float_count);
        for (const auto& path : args.pixel_values) {
            auto pair = readPixels(path, profile);
            pixels.insert(
                pixels.end(),
                std::make_move_iterator(pair.begin()),
                std::make_move_iterator(pair.end()));
        }

        geniex::PixelData pixel_data;
        pixel_data.pixel_values = std::move(pixels);
        // temporal_patch_size=2 fuses the two source frames into T=1.
        pixel_data.image_grid_thw.assign(
            args.pixel_values.size(),
            {
                1,
                static_cast<int>(profile.grid_height),
                static_cast<int>(profile.grid_width),
            });
        geniex::VLMInput input;
        input.pixel_data = std::move(pixel_data);

        std::cout
            << "Validated raw-video contract: "
            << args.pixel_values.size() << " pair(s), float32["
            << profile.pixel_rows << "," << profile.pixel_columns
            << "], grid [1," << profile.grid_height << ","
            << profile.grid_width << "], " << profile.visual_tokens
            << " visual tokens/pair, AR" << text_runtime.prefill_ar
            << "/AR" << text_runtime.decode_ar
            << " text runtime, prefill-safe prompt limit "
            << text_runtime.prefill_kv_capacity
            << ", patched DeepStack classifier.\n"
            << "Loading full-DeepStack QAIRT contexts on QnnHtp...\n";

        geniex::QnnRuntimeConfig runtime;
        auto model = geniex::qwen3_vl::makeModel(runtime, modelConfig(args.model_dir));
        if (!model) throw std::runtime_error("Failed to initialize Qwen3-VL model");
        // Qualcomm's Qwen3-VL factory defaults to the still-image pad token.
        // These inputs are temporal video patches, and the upstream
        // Transformers prompt uses <|video_pad|>. Select the same token for
        // visual-feature scatter, DeepStack masking, and MRoPE construction.
        model->setVisionTokenIds(kVisionStartTokenId, kVideoPadTokenId);

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
