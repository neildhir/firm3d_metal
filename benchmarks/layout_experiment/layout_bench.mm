// Layout experiment host: traces one particle ensemble with one trace-kernel variant,
// using the setup kernel, buffers, TraceParams and host loop of firm3d_metal's
// metal_wrapper.mm (fixed steps per dispatch instead of time-sized dispatches).
//
// usage: layout_bench <metal_kernel.metal> <variants.metal> <data_dir> <variant>
//                     <nparticles> <tmax_s> <tol> <steps_per_dispatch> <threads_per_tg_1t>
//                     <out_prefix>
// variant: orig | coop | coop_active | 1t
// Prints one CSV row; writes <out_prefix>.final.f32 (n x 6: t, x1, x2, zeta, vpar, dt)
// and <out_prefix>.steps.u32 (DP5 steps per particle; zeros for orig).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int THREADS_PER_BLOCK = 32;
constexpr int PARTICLES_PER_BLOCK = 8;
constexpr int N_DERIV_OUTPUTS = 4;

// Must match TraceParams in metal_kernel.metal.
struct TraceParams {
    float grid_ranges[12];
    float mass;
    float charge;
    float atol;
    float rtol;
    float v_total;
    float psi0;
    float inv_psi0_charge;
    int n_x2;
    int n_x3;
    int nparticles;
    int rhs_id;
    int max_steps;
    int rescale_abstol;
};

std::string read_text(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

template <typename T>
std::vector<T> read_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto bytes = static_cast<size_t>(in.tellg());
    std::vector<T> out(bytes / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return out;
}

// meta.txt: one "key v1 v2 ..." line per quantity
std::map<std::string, std::vector<double>> read_meta(const std::string& path) {
    std::map<std::string, std::vector<double>> meta;
    std::istringstream lines(read_text(path));
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        std::string key;
        if (!(fields >> key)) {
            continue;
        }
        double v;
        while (fields >> v) {
            meta[key].push_back(v);
        }
    }
    return meta;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          NSString* name, MTLFunctionConstantValues* constants) {
    NSError* error = nil;
    id<MTLFunction> fn = constants == nil
        ? [library newFunctionWithName:name]
        : [library newFunctionWithName:name constantValues:constants error:&error];
    if (fn == nil) {
        throw std::runtime_error(std::string("no kernel ") + name.UTF8String);
    }
    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&error];
    if (ps == nil) {
        throw std::runtime_error(std::string("pipeline failed: ") + error.localizedDescription.UTF8String);
    }
    return ps;
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes, const void* data) {
    id<MTLBuffer> b = data != nullptr
        ? [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
        : [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (b == nil) {
        throw std::runtime_error("buffer allocation failed");
    }
    if (data == nullptr) {
        memset(b.contents, 0, bytes);
    }
    return b;
}

// One dispatch to completion; returns its GPU-busy time [s].
double run_command(id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps,
                   const std::vector<std::pair<int, id<MTLBuffer>>>& buffers,
                   const TraceParams& params, int params_index, MTLSize groups, MTLSize threads) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    for (const auto& [index, buffer] : buffers) {
        [enc setBuffer:buffer offset:0 atIndex:index];
    }
    [enc setBytes:&params length:sizeof(TraceParams) atIndex:params_index];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) {
        throw std::runtime_error(std::string("command buffer failed: ") +
                                 cb.error.localizedDescription.UTF8String);
    }
    return cb.GPUEndTime - cb.GPUStartTime;
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc != 11) {
            fprintf(stderr, "usage: %s kernel.metal variants.metal data_dir variant n tmax tol "
                            "steps_per_dispatch tg_1t out_prefix\n", argv[0]);
            return 2;
        }
        const std::string kernel_path = argv[1], variants_path = argv[2], data_dir = argv[3];
        const std::string variant = argv[4];
        const int n = std::stoi(argv[5]);
        const double tmax = std::stod(argv[6]);
        const double tol = std::stod(argv[7]);
        const int steps_per_dispatch = std::stoi(argv[8]);
        const int tg_1t = std::stoi(argv[9]);
        const std::string out_prefix = argv[10];

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe; // as metal_wrapper.mm
        NSError* error = nil;
        const std::string src = read_text(kernel_path) + "\n" + read_text(variants_path);
        id<MTLLibrary> library = [device newLibraryWithSource:@(src.c_str()) options:options error:&error];
        if (library == nil) {
            fprintf(stderr, "compile failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        id<MTLComputePipelineState> setup = make_pipeline(device, library, @"setup_boozer", nil);
        id<MTLComputePipelineState> trace;
        const bool one_thread = variant == "1t";
        if (variant == "orig") {
            trace = make_pipeline(device, library, @"trace_boozer", nil);
        } else if (variant == "coop" || variant == "coop_active") {
            MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
            bool gate = variant == "coop_active";
            [cv setConstantValue:&gate type:MTLDataTypeBool atIndex:0];
            trace = make_pipeline(device, library, @"trace_boozer_x", cv);
        } else if (one_thread) {
            trace = make_pipeline(device, library, @"trace_boozer_1t", nil);
        } else {
            fprintf(stderr, "unknown variant %s\n", variant.c_str());
            return 2;
        }
        if (setup.threadExecutionWidth != THREADS_PER_BLOCK) {
            fprintf(stderr, "SIMD width %lu, need 32\n", (unsigned long)setup.threadExecutionWidth);
            return 1;
        }

        auto meta = read_meta(data_dir + "/meta.txt");
        const std::vector<float> table = read_raw<float>(data_dir + "/table.f32");
        const std::vector<float> all_particles = read_raw<float>(data_dir + "/particles.f32");
        if ((int)(all_particles.size() / 4) < n) {
            throw std::runtime_error("not enough particles in particles.f32");
        }

        TraceParams params = {};
        const char* keys[3] = {"srange", "trange", "zrange"};
        for (int c = 0; c < 3; ++c) {
            const auto& r = meta.at(keys[c]);
            params.grid_ranges[4*c + 0] = (float)r[0];
            params.grid_ranges[4*c + 1] = (float)r[1];
            params.grid_ranges[4*c + 2] = (float)r[2];
            params.grid_ranges[4*c + 3] = (float)((r[2] - 1) / (r[1] - r[0]));
        }
        params.n_x2 = ((int)meta.at("trange")[2] - 1) / 3;
        params.n_x3 = ((int)meta.at("zrange")[2] - 1) / 3;
        const double mass = meta.at("mass")[0], charge = meta.at("charge")[0];
        const double psi0 = meta.at("psi0")[0], vtotal = meta.at("vtotal")[0];
        params.mass = (float)mass;
        params.charge = (float)charge;
        params.atol = (float)tol;
        params.rtol = (float)tol;
        params.v_total = (float)vtotal;
        params.psi0 = (float)psi0;
        params.inv_psi0_charge = (float)(1.0 / (psi0 * charge));
        params.nparticles = n;
        params.rhs_id = meta.count("rhs_id") ? (int)meta.at("rhs_id")[0] : 0; // 0 vacuum, 1 general
        params.max_steps = steps_per_dispatch;
        params.rescale_abstol = 1;

        const size_t nn = (size_t)n;
        const int nblocks = n / PARTICLES_PER_BLOCK + 1;
        std::vector<float> minus_one(nn, -1.0f), tmax_f(nn, (float)tmax);
        id<MTLBuffer> state_b = make_buffer(device, 4*nn*sizeof(float), all_particles.data());
        id<MTLBuffer> quad_b = make_buffer(device, table.size()*sizeof(float), table.data());
        id<MTLBuffer> derivs_b = make_buffer(device, (size_t)nblocks*PARTICLES_PER_BLOCK*7*N_DERIV_OUTPUTS*sizeof(float), nullptr);
        id<MTLBuffer> fsal_b = make_buffer(device, 4*nn*sizeof(float), nullptr);
        id<MTLBuffer> mu_b = make_buffer(device, nn*sizeof(float), minus_one.data());
        id<MTLBuffer> dt_b = make_buffer(device, nn*sizeof(float), minus_one.data());
        id<MTLBuffer> tmax_b = make_buffer(device, nn*sizeof(float), tmax_f.data());
        id<MTLBuffer> t_b = make_buffer(device, nn*sizeof(float), nullptr);
        id<MTLBuffer> dtmax_b = make_buffer(device, nn*sizeof(float), nullptr);
        id<MTLBuffer> left_b = make_buffer(device, nn, nullptr);
        id<MTLBuffer> steps_b = make_buffer(device, nn*sizeof(uint32_t), nullptr);

        run_command(queue, setup,
                    {{0, state_b}, {1, quad_b}, {2, mu_b}, {3, dt_b}, {4, dtmax_b}, {5, t_b}, {6, derivs_b}},
                    params, 7, MTLSizeMake(nblocks, 1, 1), MTLSizeMake(THREADS_PER_BLOCK, 1, 1));

        const std::vector<std::pair<int, id<MTLBuffer>>> trace_buffers = {
            {0, state_b}, {1, quad_b}, {2, one_thread ? fsal_b : derivs_b}, {3, mu_b}, {4, tmax_b},
            {5, t_b}, {6, dt_b}, {7, dtmax_b}, {8, left_b}, {10, steps_b}};
        const MTLSize groups = one_thread ? MTLSizeMake((nn + tg_1t - 1) / tg_1t, 1, 1)
                                          : MTLSizeMake(nblocks, 1, 1);
        const MTLSize threads = one_thread ? MTLSizeMake(tg_1t, 1, 1)
                                           : MTLSizeMake(THREADS_PER_BLOCK, 1, 1);

        const float* t_host = (const float*)t_b.contents;
        const uint8_t* left_host = (const uint8_t*)left_b.contents;
        double gpu_s = 0.0, longest_s = 0.0;
        long dispatches = 0;
        const auto wall0 = std::chrono::steady_clock::now();
        while (true) {
            long active = 0;
            for (size_t i = 0; i < nn; ++i) {
                active += (!left_host[i] && t_host[i] < tmax_f[i]) ? 1 : 0;
            }
            if (active == 0) {
                break;
            }
            const double s = run_command(queue, trace, trace_buffers, params, 9, groups, threads);
            gpu_s += s;
            longest_s = std::max(longest_s, s);
            ++dispatches;
        }
        const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

        const float* state_host = (const float*)state_b.contents;
        const float* dt_host = (const float*)dt_b.contents;
        const uint32_t* steps_host = (const uint32_t*)steps_b.contents;
        std::vector<float> final(6*nn);
        uint64_t total_steps = 0;
        long lost = 0;
        for (size_t i = 0; i < nn; ++i) {
            final[6*i + 0] = t_host[i];
            for (int j = 0; j < 4; ++j) {
                final[6*i + 1 + j] = state_host[4*i + j];
            }
            final[6*i + 5] = dt_host[i];
            total_steps += steps_host[i];
            lost += left_host[i] ? 1 : 0;
        }
        std::ofstream(out_prefix + ".final.f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(final.data()), (std::streamsize)(final.size()*sizeof(float)));
        std::ofstream(out_prefix + ".steps.u32", std::ios::binary)
            .write(reinterpret_cast<const char*>(steps_host), (std::streamsize)(nn*sizeof(uint32_t)));

        printf("%s,%d,%d,%d,%g,%g,%.6f,%.6f,%ld,%.6f,%llu,%ld\n", variant.c_str(), n,
               one_thread ? tg_1t : THREADS_PER_BLOCK, steps_per_dispatch, tol, tmax, gpu_s, wall_s,
               dispatches, longest_s, (unsigned long long)total_steps, lost);
    }
    return 0;
}
