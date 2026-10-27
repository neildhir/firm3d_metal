// Metal host code for the CATAPULT kernels on Apple GPUs.
//
// Implements the same entry points as the CUDA host code in cuda_kernel.cu
// (boozer_gpu_tracing, test_gpu_interpolation, test_derivatives_boozer) for
// float only: Apple GPUs have no FP64, so the double specializations throw.
// The kernels are compiled at runtime from the embedded MSL source, which
// needs no Xcode Metal toolchain on the build machine. Buffers use the
// shared storage mode, so on Apple Silicon host and GPU read the same
// memory and there are no explicit transfers.
//
// The trace kernel is dispatched in bounded chunks of max_steps DP5 steps
// (macOS limits how long a command buffer may run) with all per-particle
// state in device buffers; the host redispatches until every particle has
// reached its tmax or left the plasma.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using std::vector;

#include "metal_kernel_embedded.h"

namespace {

constexpr int THREADS_PER_BLOCK = 32;
constexpr int PARTICLES_PER_BLOCK = 8;
constexpr int N_DERIV_OUTPUTS = 4;

// rhs_id values; must match metal_kernel.metal
constexpr int RHS_BOOZER_VACUUM = 0;
constexpr int RHS_BOOZER = 1;

// Must match the struct of the same name in metal_kernel.metal.
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

[[noreturn]] void throw_no_fp64(const char* name) {
    throw std::runtime_error(
        std::string(name) +
        ": the Metal GPU backend is single precision only, because Apple GPUs "
        "have no float64. Build the CATAPULT field with precision=\"single\".");
}

// DP5 steps per trace dispatch. Bounds command-buffer runtime below the macOS
// GPU watchdog while keeping dispatch overhead negligible; overridable for
// experiments via FIRM3D_METAL_MAX_STEPS.
int max_steps_per_dispatch() {
    static int value = [] {
        const char* env = std::getenv("FIRM3D_METAL_MAX_STEPS");
        int steps = env ? std::atoi(env) : 1000;
        return steps > 0 ? steps : 1000;
    }();
    return value;
}

// Safety cap on trace dispatches, so a step size collapsing to zero surfaces
// as an error instead of an endless host loop.
constexpr long MAX_TRACE_DISPATCHES = 1000000;

// Dispatch profiling, enabled with FIRM3D_METAL_PROFILE=1: sums each command
// buffer's GPU-busy interval (GPUStartTime..GPUEndTime) so kernel time can be
// separated from host-loop overhead (encode, commit, completion wait, the
// active-particle scan). Costs two property reads per dispatch when enabled.
struct ProfileStats {
    bool enabled = std::getenv("FIRM3D_METAL_PROFILE") != nullptr;
    long dispatches = 0;
    double gpu_s = 0.0;
};

ProfileStats& profile_stats() {
    static ProfileStats stats;
    return stats;
}

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> setup = nil;
    id<MTLComputePipelineState> trace = nil;
    id<MTLComputePipelineState> test_interp = nil;
    id<MTLComputePipelineState> test_derivs = nil;
};

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          NSString* name) {
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (fn == nil) {
        throw std::runtime_error(std::string("Metal kernel not found: ") +
                                 [name UTF8String]);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:fn error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string("Failed to build Metal pipeline for ") +
                                 [name UTF8String] + ": " +
                                 [[[error localizedDescription] description] UTF8String]);
    }
    // The kernels assume a 32-lane SIMD-group (one CUDA warp) for their
    // shuffles and ballots; every Apple-designed GPU satisfies this.
    if (pipeline.threadExecutionWidth != THREADS_PER_BLOCK) {
        throw std::runtime_error(
            "The Metal backend requires a GPU with a SIMD-group width of 32 "
            "(all Apple Silicon GPUs); this device reports " +
            std::to_string(pipeline.threadExecutionWidth));
    }
    if (profile_stats().enabled) {
        // maxTotalThreadsPerThreadgroup below the device maximum signals
        // register pressure limiting occupancy for this kernel
        fprintf(stderr,
                "[metal-profile] pipeline %s: maxThreadsPerThreadgroup %lu, "
                "staticThreadgroupMemory %lu B\n",
                [name UTF8String],
                (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
                (unsigned long)pipeline.staticThreadgroupMemoryLength);
    }
    return pipeline;
}

MetalContext& context() {
    static MetalContext ctx = [] {
        MetalContext c;
        c.device = MTLCreateSystemDefaultDevice();
        if (c.device == nil) {
            throw std::runtime_error("No Metal GPU device is available");
        }
        c.queue = [c.device newCommandQueue];

        MTLCompileOptions* options = [MTLCompileOptions new];
        // IEEE-conformant float math by default, matching the CUDA kernels.
        // FIRM3D_METAL_FAST_MATH=1 compiles with relaxed math (FMA
        // contraction, reassociation, faster transcendentals) for
        // experiments; trajectories then differ from the safe build at the
        // roundoff level, so validate against the CPU tracer before trusting
        // results from it.
        options.mathMode = std::getenv("FIRM3D_METAL_FAST_MATH") != nullptr
                         ? MTLMathModeRelaxed : MTLMathModeSafe;
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:FIRM3D_METAL_KERNEL_SOURCE];
        id<MTLLibrary> library = [c.device newLibraryWithSource:src
                                                        options:options
                                                          error:&error];
        if (library == nil) {
            throw std::runtime_error(
                std::string("Failed to compile the Metal kernels: ") +
                [[[error localizedDescription] description] UTF8String]);
        }
        c.setup = make_pipeline(c.device, library, @"setup_boozer");
        c.trace = make_pipeline(c.device, library, @"trace_boozer");
        c.test_interp = make_pipeline(c.device, library, @"test_interp_boozer");
        c.test_derivs = make_pipeline(c.device, library, @"test_derivs_boozer");
        return c;
    }();
    return ctx;
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes, const void* data,
                          const char* what) {
    if (bytes > device.maxBufferLength) {
        throw std::runtime_error(
            std::string(what) + " needs " + std::to_string(bytes) +
            " bytes, more than this GPU's maximum buffer size of " +
            std::to_string(device.maxBufferLength) +
            " bytes; reduce the interpolation resolution");
    }
    id<MTLBuffer> buffer =
        data ? [device newBufferWithBytes:data length:bytes
                                  options:MTLResourceStorageModeShared]
             : [device newBufferWithLength:bytes
                                   options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        throw std::runtime_error(std::string("Failed to allocate ") +
                                 std::to_string(bytes) + " bytes for " + what +
                                 "; reduce the interpolation resolution or the "
                                 "number of particles");
    }
    return buffer;
}

void run_command(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                 NSArray<id<MTLBuffer>>* buffers, const TraceParams& params,
                 int params_index, int nblocks) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < buffers.count; ++i) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }
    [enc setBytes:&params length:sizeof(TraceParams) atIndex:params_index];
    [enc dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS_PER_BLOCK, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error != nil) {
        throw std::runtime_error(
            std::string("Metal command buffer failed: ") +
            [[[cb.error localizedDescription] description] UTF8String]);
    }
    ProfileStats& prof = profile_stats();
    if (prof.enabled) {
        prof.dispatches += 1;
        prof.gpu_s += cb.GPUEndTime - cb.GPUStartTime;
    }
}

// Capture one trace dispatch into an Xcode .gputrace document (for the Metal
// debugger's occupancy and limiter counters), enabled by pointing
// FIRM3D_METAL_CAPTURE at the output path. Programmatic capture outside
// Xcode additionally requires MTL_CAPTURE_ENABLED=1 in the environment.
void capture_dispatch(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                      NSArray<id<MTLBuffer>>* buffers, const TraceParams& params,
                      int params_index, int nblocks, const char* path) {
    MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];
    MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
    desc.captureObject = queue;
    desc.destination = MTLCaptureDestinationGPUTraceDocument;
    desc.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError* error = nil;
    if (![mgr startCaptureWithDescriptor:desc error:&error]) {
        fprintf(stderr,
                "[metal-profile] GPU capture failed (set MTL_CAPTURE_ENABLED=1): "
                "%s\n",
                [[[error localizedDescription] description] UTF8String]);
        run_command(queue, pipeline, buffers, params, params_index, nblocks);
        return;
    }
    run_command(queue, pipeline, buffers, params, params_index, nblocks);
    [mgr stopCapture];
    fprintf(stderr, "[metal-profile] wrote GPU capture to %s\n", path);
}

// Fill grid_ranges (start, end, n_pts, 1/grid_size per coordinate) and the
// cell counts from the double-precision ranges, as the CUDA host does.
void fill_grid(TraceParams& params, const double* x1, const double* x2,
               const double* x3) {
    const double* ranges[3] = {x1, x2, x3};
    for (int c = 0; c < 3; ++c) {
        double inv = (ranges[c][2] - 1) / (ranges[c][1] - ranges[c][0]);
        params.grid_ranges[4*c + 0] = (float)ranges[c][0];
        params.grid_ranges[4*c + 1] = (float)ranges[c][1];
        params.grid_ranges[4*c + 2] = (float)ranges[c][2];
        params.grid_ranges[4*c + 3] = (float)inv;
    }
    params.n_x2 = ((int)x2[2] - 1) / 3;
    params.n_x3 = ((int)x3[2] - 1) / 3;
}

template <typename T>
T* array_ptr(py::array_t<T>& x) {
    return static_cast<T*>(x.request().ptr);
}

} // namespace

// The declarations these specializations satisfy live in tracing.h under
// USE_METAL; signatures must match the CUDA primary template exactly.
#include "tracing.h"

template <>
vector<float> boozer_gpu_tracing<float>(py::array_t<float> quad_pts,
                                        py::array_t<double> srange,
                                        py::array_t<double> trange,
                                        py::array_t<double> zrange,
                                        py::array_t<float> stz_init, double m, double q,
                                        double vtotal, py::array_t<float> vtang,
                                        py::array_t<double> tmax, double tol,
                                        py::array_t<float> dt_in,
                                        py::array_t<float> mu_in, double psi0,
                                        int nparticles, bool vacuum) {
    @autoreleasepool {
        MetalContext& ctx = context();

        TraceParams params = {};
        fill_grid(params, array_ptr(srange), array_ptr(trange), array_ptr(zrange));
        params.mass = (float)m;
        params.charge = (float)q;
        params.atol = (float)tol;
        params.rtol = (float)tol;
        params.v_total = (float)vtotal;
        params.psi0 = (float)psi0;
        params.inv_psi0_charge = (float)(1.0 / (psi0 * q));
        params.nparticles = nparticles;
        params.rhs_id = vacuum ? RHS_BOOZER_VACUUM : RHS_BOOZER;
        params.max_steps = max_steps_per_dispatch();
        params.rescale_abstol = 1;

        // interleave (x1, x2, zeta) with vpar into the state buffer
        float* stz_arr = array_ptr(stz_init);
        float* vtang_arr = array_ptr(vtang);
        vector<float> state(4 * (size_t)nparticles);
        for (int i = 0; i < nparticles; ++i) {
            for (int j = 0; j < 3; ++j) {
                state[4*i + j] = stz_arr[3*i + j];
            }
            state[4*i + 3] = vtang_arr[i];
        }

        double* tmax_arr = array_ptr(tmax);
        vector<float> tmax_f(nparticles);
        for (int i = 0; i < nparticles; ++i) {
            tmax_f[i] = (float)tmax_arr[i];
        }

        int nblocks = nparticles / PARTICLES_PER_BLOCK + 1;
        size_t n = (size_t)nparticles;

        id<MTLBuffer> state_b = make_buffer(ctx.device, 4*n*sizeof(float),
                                            state.data(), "the particle state");
        id<MTLBuffer> quad_b = make_buffer(ctx.device, quad_pts.size()*sizeof(float),
                                           array_ptr(quad_pts), "the field table");
        id<MTLBuffer> derivs_b = make_buffer(
            ctx.device,
            (size_t)nblocks*PARTICLES_PER_BLOCK*7*N_DERIV_OUTPUTS*sizeof(float),
            nullptr, "the derivative scratch space");
        id<MTLBuffer> mu_b = make_buffer(ctx.device, n*sizeof(float),
                                         array_ptr(mu_in), "mu");
        id<MTLBuffer> tmax_b = make_buffer(ctx.device, n*sizeof(float), tmax_f.data(),
                                           "tmax");
        id<MTLBuffer> t_b = make_buffer(ctx.device, n*sizeof(float), nullptr, "t");
        id<MTLBuffer> dt_b = make_buffer(ctx.device, n*sizeof(float),
                                         array_ptr(dt_in), "dt");
        id<MTLBuffer> dtmax_b = make_buffer(ctx.device, n*sizeof(float), nullptr,
                                            "dtmax");
        id<MTLBuffer> left_b = make_buffer(ctx.device, n*sizeof(uint8_t), nullptr,
                                           "the loss flags");
        memset(t_b.contents, 0, n*sizeof(float));
        memset(left_b.contents, 0, n*sizeof(uint8_t));

        run_command(ctx.queue, ctx.setup,
                    @[state_b, quad_b, mu_b, dt_b, dtmax_b, t_b, derivs_b], params, 7,
                    nblocks);

        NSArray<id<MTLBuffer>>* trace_buffers =
            @[state_b, quad_b, derivs_b, mu_b, tmax_b, t_b, dt_b, dtmax_b, left_b];
        const float* t_host = (const float*)t_b.contents;
        const uint8_t* left_host = (const uint8_t*)left_b.contents;
        ProfileStats& prof = profile_stats();
        const long prof_dispatches0 = prof.dispatches;
        const double prof_gpu0 = prof.gpu_s;
        const double wall0 =
            prof.enabled ? [[NSDate date] timeIntervalSince1970] : 0.0;
        long dispatches = 0;
        while (true) {
            long active = 0;
            for (int i = 0; i < nparticles; ++i) {
                active += (!left_host[i] && t_host[i] < tmax_f[i]) ? 1 : 0;
            }
            if (active == 0) {
                break;
            }
            if (++dispatches > MAX_TRACE_DISPATCHES) {
                throw std::runtime_error(
                    "Metal tracing did not finish after " +
                    std::to_string(MAX_TRACE_DISPATCHES) +
                    " dispatches; the step size has likely collapsed");
            }
            // capture the second dispatch: past start-up transients, still
            // early enough that nearly all particles are active
            static const char* capture_path = std::getenv("FIRM3D_METAL_CAPTURE");
            if (capture_path != nullptr && dispatches == 2) {
                capture_dispatch(ctx.queue, ctx.trace, trace_buffers, params, 9,
                                 nblocks, capture_path);
                capture_path = nullptr;
            } else {
                run_command(ctx.queue, ctx.trace, trace_buffers, params, 9, nblocks);
            }
        }
        if (prof.enabled) {
            double wall = [[NSDate date] timeIntervalSince1970] - wall0;
            double gpu = prof.gpu_s - prof_gpu0;
            fprintf(stderr,
                    "[metal-profile] trace: %d particles, %ld dispatches x %d "
                    "steps, GPU busy %.3f s, loop wall %.3f s (GPU %.1f%%)\n",
                    nparticles, prof.dispatches - prof_dispatches0,
                    params.max_steps, gpu, wall, 100.0*gpu/wall);
        }

        const float* state_host = (const float*)state_b.contents;
        const float* dt_host = (const float*)dt_b.contents;
        const float* mu_host = (const float*)mu_b.contents;
        vector<float> out(7*n);
        for (int i = 0; i < nparticles; ++i) {
            out[7*i + 0] = t_host[i];
            for (int j = 0; j < 4; ++j) {
                out[7*i + 1 + j] = state_host[4*i + j];
            }
            out[7*i + 5] = dt_host[i];
            out[7*i + 6] = mu_host[i];
        }
        return out;
    }
}

template <>
vector<double> boozer_gpu_tracing<double>(py::array_t<double>, py::array_t<double>,
                                          py::array_t<double>, py::array_t<double>,
                                          py::array_t<double>, double, double, double,
                                          py::array_t<double>, py::array_t<double>,
                                          double, py::array_t<double>,
                                          py::array_t<double>, double, int, bool) {
    throw_no_fp64("boozer_gpu_tracing");
}

template <>
py::array_t<float> test_gpu_interpolation<float>(py::array_t<float> quad_pts,
                                                 py::array_t<double> x1_range,
                                                 py::array_t<double> x2_range,
                                                 py::array_t<double> x3_range,
                                                 py::array_t<float> loc,
                                                 std::string coordinates,
                                                 int n_points) {
    @autoreleasepool {
        int rhs_id;
        if (coordinates == "boozer_vacuum") {
            rhs_id = RHS_BOOZER_VACUUM;
        } else if (coordinates == "boozer") {
            rhs_id = RHS_BOOZER;
        } else {
            throw std::runtime_error("test_gpu_interpolation: coordinates '" +
                                     coordinates +
                                     "' are not implemented in the Metal backend");
        }
        int n = rhs_id == RHS_BOOZER_VACUUM ? 6 : 12;

        MetalContext& ctx = context();

        // convert (s, theta) to pseudo-Cartesian in place, as the CUDA host does
        float* loc_arr = array_ptr(loc);
        for (int i = 0; i < n_points; ++i) {
            float x1 = loc_arr[3*i] * cosf(loc_arr[3*i + 1]);
            float x2 = loc_arr[3*i] * sinf(loc_arr[3*i + 1]);
            loc_arr[3*i] = x1;
            loc_arr[3*i + 1] = x2;
        }

        TraceParams params = {};
        fill_grid(params, array_ptr(x1_range), array_ptr(x2_range),
                  array_ptr(x3_range));
        params.nparticles = n_points;
        params.rhs_id = rhs_id;

        id<MTLBuffer> quad_b = make_buffer(ctx.device, quad_pts.size()*sizeof(float),
                                           array_ptr(quad_pts), "the field table");
        id<MTLBuffer> loc_b = make_buffer(ctx.device, 3*(size_t)n_points*sizeof(float),
                                          loc_arr, "the evaluation points");
        id<MTLBuffer> out_b = make_buffer(ctx.device,
                                          (size_t)n*n_points*sizeof(float), nullptr,
                                          "the interpolation output");

        int nblocks = n_points / PARTICLES_PER_BLOCK + 1;
        run_command(ctx.queue, ctx.test_interp, @[quad_b, loc_b, out_b], params, 3,
                    nblocks);

        return py::array_t<float>((size_t)n*n_points, (const float*)out_b.contents);
    }
}

template <>
py::array_t<double> test_gpu_interpolation<double>(py::array_t<double>,
                                                   py::array_t<double>,
                                                   py::array_t<double>,
                                                   py::array_t<double>,
                                                   py::array_t<double>, std::string,
                                                   int) {
    throw_no_fp64("test_gpu_interpolation");
}

template <>
py::array_t<float> test_derivatives_boozer<float>(py::array_t<float> quad_pts,
                                                  py::array_t<double> x1_range,
                                                  py::array_t<double> x2_range,
                                                  py::array_t<double> x3_range,
                                                  py::array_t<float> loc,
                                                  py::array_t<float> vpar,
                                                  double v_total, double m, double q,
                                                  double psi0, int n_points,
                                                  bool vacuum) {
    @autoreleasepool {
        MetalContext& ctx = context();

        TraceParams params = {};
        fill_grid(params, array_ptr(x1_range), array_ptr(x2_range),
                  array_ptr(x3_range));
        params.mass = (float)m;
        params.charge = (float)q;
        params.v_total = (float)v_total;
        params.psi0 = (float)psi0;
        params.inv_psi0_charge = (float)(1.0 / (psi0 * q));
        params.nparticles = n_points;
        params.rhs_id = vacuum ? RHS_BOOZER_VACUUM : RHS_BOOZER;

        float* loc_arr = array_ptr(loc);
        float* vpar_arr = array_ptr(vpar);
        vector<float> init_pos(4 * (size_t)n_points);
        for (int i = 0; i < n_points; ++i) {
            init_pos[4*i + 0] = loc_arr[3*i + 0] * cosf(loc_arr[3*i + 1]);
            init_pos[4*i + 1] = loc_arr[3*i + 0] * sinf(loc_arr[3*i + 1]);
            init_pos[4*i + 2] = loc_arr[3*i + 2];
            init_pos[4*i + 3] = vpar_arr[i];
        }

        size_t n = (size_t)n_points;
        int nblocks = n_points / PARTICLES_PER_BLOCK + 1;
        vector<float> minus_one(n, -1.0f);

        id<MTLBuffer> quad_b = make_buffer(ctx.device, quad_pts.size()*sizeof(float),
                                           array_ptr(quad_pts), "the field table");
        id<MTLBuffer> state_b = make_buffer(ctx.device, 4*n*sizeof(float),
                                            init_pos.data(), "the particle state");
        id<MTLBuffer> t_b = make_buffer(ctx.device, n*sizeof(float), nullptr, "t");
        id<MTLBuffer> out_b = make_buffer(ctx.device, 4*n*sizeof(float), nullptr,
                                          "the derivative output");
        id<MTLBuffer> derivs_b = make_buffer(
            ctx.device,
            (size_t)nblocks*PARTICLES_PER_BLOCK*7*N_DERIV_OUTPUTS*sizeof(float),
            nullptr, "the derivative scratch space");
        id<MTLBuffer> mu_b = make_buffer(ctx.device, n*sizeof(float),
                                         minus_one.data(), "mu");
        id<MTLBuffer> dt_b = make_buffer(ctx.device, n*sizeof(float),
                                         minus_one.data(), "dt");
        id<MTLBuffer> dtmax_b = make_buffer(ctx.device, n*sizeof(float), nullptr,
                                            "dtmax");
        memset(t_b.contents, 0, n*sizeof(float));

        run_command(ctx.queue, ctx.setup,
                    @[state_b, quad_b, mu_b, dt_b, dtmax_b, t_b, derivs_b], params, 7,
                    nblocks);
        run_command(ctx.queue, ctx.test_derivs,
                    @[quad_b, state_b, t_b, out_b, derivs_b, mu_b, dt_b], params, 7,
                    nblocks);

        return py::array_t<float>(4*n, (const float*)out_b.contents);
    }
}

template <>
py::array_t<double> test_derivatives_boozer<double>(py::array_t<double>,
                                                    py::array_t<double>,
                                                    py::array_t<double>,
                                                    py::array_t<double>,
                                                    py::array_t<double>,
                                                    py::array_t<double>, double,
                                                    double, double, double, int,
                                                    bool) {
    throw_no_fp64("test_derivatives_boozer");
}
