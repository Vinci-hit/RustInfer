// Standalone reference executable; never linked into RustInfer.
// Build against the pinned llama.cpp checkout documented in GGUF_COMPARISON.md.
#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

static json read_json(const std::string & path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path);
    return json::parse(file);
}

static void write_json(const std::string & path, const json & value) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot write " + path);
    file << value.dump();
    if (!file) throw std::runtime_error("write failed: " + path);
}

static json step(llama_context * ctx, const std::vector<llama_token> & ids, int vocab) {
    auto input = ids;
    auto start = Clock::now();
    if (llama_decode(ctx, llama_batch_get_one(input.data(), input.size())) != 0)
        throw std::runtime_error("llama_decode failed");
    const auto * data = llama_get_logits_ith(ctx, -1); // synchronizes readback
    if (!data) throw std::runtime_error("missing logits");
    std::vector<float> logits(data, data + vocab);
    double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    for (float value : logits)
        if (!std::isfinite(value)) throw std::runtime_error("non-finite logit");
    int top = std::max_element(logits.begin(), logits.end()) - logits.begin();
    return {{"elapsed_seconds", elapsed}, {"top_id", top}, {"logits", logits}};
}

int main(int argc, char ** argv) try {
    if (argc != 2) throw std::runtime_error("usage: gguf-llama-reference manifest.json");
    auto manifest = read_json(argv[1]);
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp = false;
    auto start = Clock::now();
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(manifest.at("model").get<std::string>().c_str(), mp),
        llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    double load_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    auto cp = llama_context_default_params();
    cp.n_ctx = manifest.at("context");
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_BF16;
    cp.type_v = GGML_TYPE_BF16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context creation failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (const auto & entry : manifest.at("cases")) {
        auto reference = read_json(entry.at("rustinfer_report"));
        auto input = reference.at("input_ids").get<std::vector<llama_token>>();
        auto continuation = reference.at("generated_ids").get<std::vector<llama_token>>();
        json report = {{"input_ids", input}, {"model_load_seconds", load_seconds},
                       {"teacher_forced_steps", json::array()}};
        if (reference.contains("rendered_prompt") && reference["rendered_prompt"].is_string()) {
            auto prompt = reference["rendered_prompt"].get<std::string>();
            std::vector<llama_token> encoded(prompt.size() + 32);
            int n = llama_tokenize(vocab, prompt.data(), prompt.size(), encoded.data(), encoded.size(), false, true);
            if (n < 0) throw std::runtime_error("tokenization buffer too small");
            encoded.resize(n);
            report["tokenized_ids"] = encoded;
            report["tokenizer_matches"] = encoded == input;
        }
        // Replay RustInfer's continuation so every compared logit has the same prefix.
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        auto ids = input;
        for (size_t i = 0; i < continuation.size(); ++i) {
            report["teacher_forced_steps"].push_back(step(ctx.get(), ids, n_vocab));
            ids = {continuation[i]};
        }
        // Independent greedy run, from empty KV and recurrent state.
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        ids = input;
        std::vector<llama_token> generated;
        std::vector<double> timings;
        bool eos = false;
        for (int i = 0; i < entry.at("steps").get<int>(); ++i) {
            auto output = step(ctx.get(), ids, n_vocab);
            auto token = output.at("top_id").get<llama_token>();
            generated.push_back(token);
            timings.push_back(output.at("elapsed_seconds"));
            if (llama_vocab_is_eog(vocab, token)) { eos = true; break; }
            ids = {token};
        }
        std::vector<char> decoded(generated.size() * 256 + 1024);
        int n = llama_detokenize(vocab, generated.data(), generated.size(), decoded.data(), decoded.size(), true, false);
        if (n < 0) throw std::runtime_error("detokenization buffer too small");
        report["generated_ids"] = generated;
        report["generated_text"] = std::string(decoded.data(), n);
        report["finish_reason"] = eos ? "eos" : "length";
        report["greedy_step_seconds"] = timings;
        write_json(entry.at("llama_report"), report);
        std::cout << entry.at("name") << ": " << report["generated_text"] << std::endl;
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
