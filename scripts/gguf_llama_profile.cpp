// Teacher-forced profiling through the public llama.cpp API.
#include "llama.h"
#include "json.hpp"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
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
struct Range {
    explicit Range(const std::string & name) { nvtxRangePushA(name.c_str()); }
    ~Range() { nvtxRangePop(); }
};
int main(int argc, char ** argv) try {
    if (argc != 3) throw std::runtime_error("usage: gguf-llama-profile workload.json output.json");
    std::ifstream input(argv[1]);
    auto work = json::parse(input);
    const auto prompt = work.at("input_ids").get<std::vector<llama_token>>();
    const auto continuation = work.at("continuation_ids").get<std::vector<llama_token>>();
    int warmups = work.at("warmups"), repeats = work.at("repeats");
    if (prompt.empty() || continuation.empty() || warmups < 1 || repeats < 1)
        throw std::runtime_error("invalid workload");
    auto start = Clock::now();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(work.at("model").get<std::string>().c_str(), mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = work.at("context");
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_BF16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) throw std::runtime_error("context creation failed");
    int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    llama_synchronize(ctx.get());
    double load_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    json records = json::array();
    for (int run = 0; run < warmups + repeats; ++run) {
        bool measured = run >= warmups;
        if (run == warmups && cudaProfilerStart() != cudaSuccess)
            throw std::runtime_error("cudaProfilerStart failed");
        Range iteration((measured ? "iteration/" : "warmup/") + std::to_string(measured ? run - warmups : run));
        {
            Range reset("reset");
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            llama_synchronize(ctx.get());
        }
        for (size_t step = 0; step < continuation.size(); ++step) {
            auto ids = step == 0 ? prompt : std::vector<llama_token>{continuation[step - 1]};
            std::string phase = step == 0 ? "prefill" : "decode/" + std::to_string(step);
            std::vector<float> logits;
            double seconds;
            {
                Range range(phase);
                auto begin = Clock::now();
                if (llama_decode(ctx.get(), llama_batch_get_one(ids.data(), ids.size())) != 0)
                    throw std::runtime_error("decode failed");
                const float * data = llama_get_logits_ith(ctx.get(), -1);
                if (!data) throw std::runtime_error("missing logits");
                logits.assign(data, data + vocab);
                seconds = std::chrono::duration<double>(Clock::now() - begin).count();
            }
            int top;
            {
                Range sample("host_argmax");
                if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
                    throw std::runtime_error("non-finite logits");
                top = std::max_element(logits.begin(), logits.end()) - logits.begin();
            }
            if (measured) records.push_back({{"iteration", run - warmups}, {"phase", phase},
                {"position", prompt.size() + step}, {"wall_seconds", seconds}, {"top_id", top},
                {"expected_id", continuation[step]}});
        }
    }
    llama_synchronize(ctx.get());
    if (cudaProfilerStop() != cudaSuccess) throw std::runtime_error("cudaProfilerStop failed");
    std::ofstream out(argv[2]);
    out << json({{"engine", "llama.cpp"}, {"load_seconds", load_seconds}, {"records", records}}).dump(2);
    if (!out) throw std::runtime_error("report write failed");
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
