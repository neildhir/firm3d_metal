// Metal compute kernels for CATAPULT guiding-center tracing on Apple GPUs.
//
// This is a single-precision port of the Boozer-coordinate kernels in
// cuda_kernel.cu. The numerics (tricubic Lagrange interpolation, adaptive
// Dormand-Prince 5 stepping, stellarator-symmetry handling) follow the CUDA
// implementation line for line; differences are structural:
//
//  - A CUDA block of 32 threads / 8 particles is one Metal threadgroup, which
//    on Apple GPUs is exactly one 32-lane SIMD-group, so warp shuffles and
//    ballots map onto simd_shuffle_down / simd_ballot.
//  - macOS bounds how long a command buffer may run, so the persistent CUDA
//    trace loop becomes a bounded loop (params.max_steps per dispatch) with
//    the particle state held in device buffers between dispatches. Particles
//    keep a fixed threadgroup slot instead of workstealing, which also keeps
//    each particle's first-same-as-last derivative slot valid across
//    dispatches.
//  - CUDA __constant__ symbols become the TraceParams argument buffer, and
//    the vacuum/finite-beta RHS choice is a uniform runtime switch
//    (params.rhs_id) instead of a template parameter.
//
// Only float arithmetic is used: Apple GPUs have no FP64. The host refuses
// double-precision calls with an informative error.

#include <metal_stdlib>

using namespace metal;

constant int THREADS_PER_BLOCK = 32;
constant int PARTICLES_PER_BLOCK = 8;

// rhs_id values; must match RhsId in metal_wrapper.mm
constant int RHS_BOOZER_VACUUM = 0; // 6 interpolants
constant int RHS_BOOZER = 1;        // 12 interpolants
constant int MAX_INTERPOLANTS = 12;
constant int N_DERIV_OUTPUTS = 4; // Boozer guiding-center state is (x1, x2, zeta, vpar)

// Must match the struct of the same name in metal_wrapper.mm (all 4-byte members).
struct TraceParams {
    float grid_ranges[12]; // per coordinate: start, end, n_pts, 1/grid_size
    float mass;
    float charge;
    float atol;
    float rtol;
    float v_total;
    float psi0;
    float inv_psi0_charge;
    int n_x2;   // interpolation cells in x2
    int n_x3;   // interpolation cells in x3
    int nparticles;
    int rhs_id;
    int max_steps;       // DP5 steps per dispatch in the trace kernel
    int rescale_abstol;  // bool: rescale atol for the vpar component
};

// position weights for Dormand-Prince 5 timesteps
constant float dp5_wgts[7][7] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0},
    {19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0},
    {9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0},
    {35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0}
};
constant float dp5_t_wgts[7] = {
    0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
};
// error estimation weights for Dormand-Prince 5 timesteps
constant float bhat_wgts[7] = {
    71.0/57600.0, 0.0, -71.0/16695.0, 71.0/1920.0,
    -17253.0/339200.0, 22.0/525.0, -1.0/40.0
};

inline int n_interpolants(int rhs_id) {
    return rhs_id == RHS_BOOZER_VACUUM ? 6 : 12;
}

// Number of threads in this SIMD-group (== threadgroup) for which pred holds;
// the Metal equivalent of CUDA __syncthreads_count for a 32-thread block.
inline uint tg_count(bool pred) {
    simd_vote v = simd_ballot(pred);
    return popcount((simd_vote::vote_t)v & 0xFFFFFFFFul);
}

// Tricubic interpolation of n components at each particle's precomputed cell
// and shape-function values; a port of cuda_kernel.cu::interpolate. Eight
// threads cooperate per particle and reduce with SIMD shuffles.
inline void interpolate(int n,
                        threadgroup float* out,
                        device const float* data,
                        threadgroup const int* cell_index_start,
                        threadgroup const float* shape_fun_vals,
                        threadgroup const bool* is_valid,
                        constant TraceParams& P,
                        uint tid) {
    for (int p = tid / 8; p < PARTICLES_PER_BLOCK; p += THREADS_PER_BLOCK / 8) {
        const int i = cell_index_start[3*p];
        const int j = cell_index_start[3*p + 1];
        const int k = cell_index_start[3*p + 2];

        // constant third dimension index for the thread
        const int kk = tid % 4;
        // second dimension index for the thread, 0 -> 2 and 1 -> 3
        const int jj = (tid / 4) % 2;

        const int base_offset = 64*n*(i*P.n_x2*P.n_x3 + j*P.n_x3 + k) + 4*jj + kk;

        const float shape_jk1 = shape_fun_vals[(8 + kk)*PARTICLES_PER_BLOCK + p]
                              * shape_fun_vals[(4 + jj)*PARTICLES_PER_BLOCK + p];
        const float shape_jk2 = shape_fun_vals[(8 + kk)*PARTICLES_PER_BLOCK + p]
                              * shape_fun_vals[(4 + jj + 2)*PARTICLES_PER_BLOCK + p];

        const bool p_valid = is_valid[p];

        for (int zz = 0; zz < n; ++zz) {
            float local_val = 0.0f;
            if (p_valid) {
                for (int ii = 0; ii < 4; ++ii) {
                    int base = base_offset + 16*ii;
                    const float shape_i = shape_fun_vals[ii*PARTICLES_PER_BLOCK + p];
                    local_val += shape_i * (shape_jk1 * data[base + 64*zz] +
                                            shape_jk2 * data[base + 8 + 64*zz]);
                }
            }
            // reduction across the 8 threads of this particle
            for (int offset = 4; offset > 0; offset /= 2) {
                local_val += simd_shuffle_down(local_val, offset);
            }
            if (tid % 8 == 0 && p_valid) {
                out[PARTICLES_PER_BLOCK*zz + p] = local_val;
            }
        }
    }
}

// Guiding-center RHS in a vacuum Boozer field; pointers are pre-offset by the
// particle index, as in cuda_kernel.cu::rhs_GC_BoozerVacuum.
inline void rhs_boozer_vacuum(int deriv_id,
                              device float* derivs,
                              threadgroup const float* x_temp,
                              threadgroup const float* block_interpolants,
                              threadgroup const bool* symmetry_exploited,
                              threadgroup const float* mu,
                              constant TraceParams& P) {
    float x1 = x_temp[1*PARTICLES_PER_BLOCK];
    float x2 = x_temp[2*PARTICLES_PER_BLOCK];

    float inv_s = 1.0f / sqrt(x1*x1 + x2*x2);
    float v_par = x_temp[4*PARTICLES_PER_BLOCK];

    float modB = block_interpolants[0*PARTICLES_PER_BLOCK];
    float dmodBds = block_interpolants[1*PARTICLES_PER_BLOCK];
    float dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK];
    float dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK];
    float G = block_interpolants[4*PARTICLES_PER_BLOCK];
    float iota = block_interpolants[5*PARTICLES_PER_BLOCK];
    float mu_val = mu[0];

    float modB_inv_G = modB / G;

    float sign = symmetry_exploited[0] ? -1.0f : 1.0f;
    dmodBdtheta *= sign;
    dmodBdzeta *= sign;

    float fak1 = P.mass*v_par*v_par/modB + P.mass*mu_val;
    float sdot = -dmodBdtheta*(fak1 * P.inv_psi0_charge);
    float tdot = dmodBds*(fak1 * P.inv_psi0_charge) + iota*(v_par*modB_inv_G);

    derivs[(4*deriv_id + 0)*PARTICLES_PER_BLOCK] = sdot*x1*inv_s - x2*tdot;
    derivs[(4*deriv_id + 1)*PARTICLES_PER_BLOCK] = sdot*x2*inv_s + x1*tdot;
    derivs[(4*deriv_id + 2)*PARTICLES_PER_BLOCK] = v_par*modB_inv_G;
    derivs[(4*deriv_id + 3)*PARTICLES_PER_BLOCK] = -(iota*dmodBdtheta + dmodBdzeta)*mu_val*modB_inv_G;
}

// General guiding-center RHS (K != 0); a port of cuda_kernel.cu::rhs_GC_Boozer.
inline void rhs_boozer_general(int deriv_id,
                               device float* derivs,
                               threadgroup const float* x_temp,
                               threadgroup const float* block_interpolants,
                               threadgroup const bool* symmetry_exploited,
                               threadgroup const float* mu,
                               constant TraceParams& P) {
    float x1 = x_temp[1*PARTICLES_PER_BLOCK];
    float x2 = x_temp[2*PARTICLES_PER_BLOCK];

    float s = sqrt(x1*x1 + x2*x2);
    float theta = atan2(x2, x1);
    float v_par = x_temp[4*PARTICLES_PER_BLOCK];

    float modB = block_interpolants[0*PARTICLES_PER_BLOCK];
    float dmodBdpsi = block_interpolants[1*PARTICLES_PER_BLOCK] / P.psi0;
    float dmodBdtheta = block_interpolants[2*PARTICLES_PER_BLOCK];
    float dmodBdzeta = block_interpolants[3*PARTICLES_PER_BLOCK];
    float G = block_interpolants[4*PARTICLES_PER_BLOCK];
    float dGdpsi = block_interpolants[5*PARTICLES_PER_BLOCK] / P.psi0;
    float I = block_interpolants[6*PARTICLES_PER_BLOCK];
    float dIdpsi = block_interpolants[7*PARTICLES_PER_BLOCK] / P.psi0;
    float iota = block_interpolants[8*PARTICLES_PER_BLOCK];
    float K = block_interpolants[9*PARTICLES_PER_BLOCK];
    float dKdtheta = block_interpolants[10*PARTICLES_PER_BLOCK];
    float dKdzeta = block_interpolants[11*PARTICLES_PER_BLOCK];

    float mu_val = mu[0];

    if (symmetry_exploited[0]) {
        dmodBdtheta *= -1.0f;
        dmodBdzeta *= -1.0f;
        K *= -1.0f;
    }

    // General guiding center equations (mode='gc'); see rhs_GC_Boozer.
    float C = -P.mass * v_par * dKdzeta / modB - P.charge * iota + P.mass * v_par * dGdpsi / modB;
    float F = -P.mass * v_par * dKdtheta / modB + P.charge + P.mass * v_par * dIdpsi / modB;
    float D = (F * G - C * I) / iota;

    float fak1 = P.mass * v_par * v_par / modB + P.mass * mu_val;

    float sdot = (I * dmodBdzeta - G * dmodBdtheta) * fak1 / (iota * D * P.psi0);
    float tdot = ((G * dmodBdpsi - K * dmodBdzeta) * fak1 - C * v_par * modB) / (iota * D);
    float zetadot = (F * v_par * modB - (dmodBdpsi * I - dmodBdtheta * K) * fak1) / (iota * D);
    float vpardot = (C * dmodBdtheta - F * dmodBdzeta) * mu_val * modB / (iota * D);

    derivs[(4*deriv_id + 0)*PARTICLES_PER_BLOCK] = sdot*cos(theta) - s*sin(theta)*tdot;
    derivs[(4*deriv_id + 1)*PARTICLES_PER_BLOCK] = sdot*sin(theta) + s*cos(theta)*tdot;
    derivs[(4*deriv_id + 2)*PARTICLES_PER_BLOCK] = zetadot;
    derivs[(4*deriv_id + 3)*PARTICLES_PER_BLOCK] = vpardot;
}

// Interpolate the field at the precomputed cells and evaluate the RHS; a port
// of cuda_kernel.cu::calc_derivs with block_interpolants hoisted to the kernel.
inline void calc_derivs(int deriv_id,
                        device float* derivs,
                        device const float* quadpts_arr,
                        threadgroup const float* x_temp,
                        threadgroup const bool* symmetry_exploited,
                        threadgroup const int* cell_index_start,
                        threadgroup const float* shape_fun_vals,
                        threadgroup const float* mu,
                        threadgroup const bool* is_valid,
                        threadgroup float* block_interpolants,
                        constant TraceParams& P,
                        uint tid) {
    int n = n_interpolants(P.rhs_id);
    interpolate(n, block_interpolants, quadpts_arr, cell_index_start, shape_fun_vals,
                is_valid, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < (uint)PARTICLES_PER_BLOCK && is_valid[tid]) {
        if (P.rhs_id == RHS_BOOZER_VACUUM) {
            rhs_boozer_vacuum(deriv_id, derivs + tid, x_temp + tid,
                              block_interpolants + tid, symmetry_exploited + tid,
                              mu + tid, P);
        } else {
            rhs_boozer_general(deriv_id, derivs + tid, x_temp + tid,
                               block_interpolants + tid, symmetry_exploited + tid,
                               mu + tid, P);
        }
    }
}

// Map a particle position onto the interpolation grid, exploiting field
// periodicity and stellarator symmetry; port of map_to_grid_boozer.
inline void map_to_grid_boozer(threadgroup float* interp_pt,
                               threadgroup float* x_temp,
                               threadgroup bool* symmetry_exploited,
                               constant TraceParams& P,
                               uint tid) {
    float x1 = x_temp[1*PARTICLES_PER_BLOCK + tid];
    float x2 = x_temp[2*PARTICLES_PER_BLOCK + tid];
    float s = sqrt(x1*x1 + x2*x2);
    interp_pt[tid] = s;
    symmetry_exploited[tid] = x2 < 0.0f;

    // periodicity in the B-field with sin(theta) unchanged: atan2 in [-pi, pi]
    float t = atan2(fabs(x2), x1);
    interp_pt[PARTICLES_PER_BLOCK + tid] = t;

    float z = x_temp[3*PARTICLES_PER_BLOCK + tid]; // zeta
    float period = P.grid_ranges[9];
    z = fmod(z, period);
    z += period*(z < 0.0f);

    if (symmetry_exploited[tid]) {
        z = period - z;
    }
    interp_pt[2*PARTICLES_PER_BLOCK + tid] = z;
}

// Build the DP5 stage state, map it to the grid, and precompute per-coordinate
// shape functions and cell indices; a port of cuda_kernel.cu::build_state with
// the interp_pt scratch hoisted to the kernel.
inline void build_state(int deriv_id,
                        threadgroup float* x_temp,
                        threadgroup bool* symmetry_exploited,
                        threadgroup int* cell_index_start,
                        threadgroup float* shape_fun_vals,
                        threadgroup const float* state,
                        device const float* derivs,
                        threadgroup const float* t,
                        threadgroup const float* dt,
                        threadgroup const bool* is_valid,
                        threadgroup float* interp_pt,
                        constant TraceParams& P,
                        uint tid) {
    if (tid < (uint)PARTICLES_PER_BLOCK && is_valid[tid]) {
        x_temp[tid] = t[tid] + dp5_t_wgts[deriv_id]*dt[tid];
    }

    for (uint idx = tid; idx < 4*PARTICLES_PER_BLOCK; idx += PARTICLES_PER_BLOCK) {
        int particle_id = idx % PARTICLES_PER_BLOCK;
        int state_var = idx / PARTICLES_PER_BLOCK;

        float val = state[state_var*PARTICLES_PER_BLOCK + particle_id];
        float dt_particle = dt[particle_id];
        for (int j = 0; j < deriv_id; ++j) {
            val += dt_particle * dp5_wgts[deriv_id][j]
                 * derivs[(N_DERIV_OUTPUTS*j + state_var)*PARTICLES_PER_BLOCK + particle_id];
        }
        x_temp[(state_var + 1)*PARTICLES_PER_BLOCK + particle_id] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < (uint)PARTICLES_PER_BLOCK && is_valid[tid]) {
        map_to_grid_boozer(interp_pt, x_temp, symmetry_exploited, P, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int particle_id = tid % PARTICLES_PER_BLOCK;
    if (tid < 3*PARTICLES_PER_BLOCK && is_valid[particle_id]) {
        int coord_id = tid / PARTICLES_PER_BLOCK;

        float value = interp_pt[tid];
        float inv_grid_size = P.grid_ranges[coord_id*4 + 3];
        float min_bound = P.grid_ranges[coord_id*4 + 0];

        float raw_offset = (value - min_bound) * inv_grid_size;
        int index = 3*((int)raw_offset / 3);
        index = min(index, (int)P.grid_ranges[coord_id*4 + 2] - 4);
        index = max(index, 0);

        float value_rel = raw_offset - (float)index;

        float x_minus_1 = value_rel - 1.0f;
        float x_minus_2 = value_rel - 2.0f;
        float x_minus_3 = value_rel - 3.0f;

        const float one_sixth = 1.0f/6.0f;
        const float one_half = 0.5f;

        float prod23 = x_minus_2 * x_minus_3;
        float prodx1 = value_rel * x_minus_1;
        int base_idx = (coord_id*4)*PARTICLES_PER_BLOCK + particle_id;

        shape_fun_vals[base_idx + 0] = -one_sixth * x_minus_1 * prod23;
        shape_fun_vals[base_idx + 1*PARTICLES_PER_BLOCK] = one_half * value_rel * prod23;
        shape_fun_vals[base_idx + 2*PARTICLES_PER_BLOCK] = -one_half * prodx1 * x_minus_3;
        shape_fun_vals[base_idx + 3*PARTICLES_PER_BLOCK] = one_sixth * prodx1 * x_minus_2;

        cell_index_start[3*particle_id + coord_id] = index/3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One DP5 stage: build the stage state and evaluate the RHS there.
inline void dp5_one_step(int deriv_id,
                         threadgroup float* x_temp,
                         device float* derivs,
                         device const float* quadpts_arr,
                         threadgroup int* cell_index_start,
                         threadgroup float* shape_fun_vals,
                         threadgroup const float* t,
                         threadgroup const float* dt,
                         threadgroup bool* symmetry_exploited,
                         threadgroup const float* state,
                         threadgroup const float* mu,
                         threadgroup const bool* is_valid,
                         threadgroup float* interp_pt,
                         threadgroup float* block_interpolants,
                         constant TraceParams& P,
                         uint tid) {
    build_state(deriv_id, x_temp, symmetry_exploited, cell_index_start, shape_fun_vals,
                state, derivs, t, dt, is_valid, interp_pt, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    calc_derivs(deriv_id, derivs, quadpts_arr, x_temp, symmetry_exploited,
                cell_index_start, shape_fun_vals, mu, is_valid, block_interpolants, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Maximum allowable timestep: at most a quarter revolution per step.
inline void max_stepsize_boozer(threadgroup float* dtmax,
                                threadgroup const float* interpolants,
                                constant TraceParams& P,
                                uint tid) {
    float modB = interpolants[PARTICLES_PER_BLOCK*0 + tid];
    float G = interpolants[PARTICLES_PER_BLOCK*4 + tid];
    dtmax[tid] = (G / modB)*0.5f*M_PI_F / P.v_total;
}

inline void check_has_left_boozer(threadgroup bool* has_left,
                                  threadgroup const float* state,
                                  uint tid) {
    float x1 = state[0*PARTICLES_PER_BLOCK + tid];
    float x2 = state[1*PARTICLES_PER_BLOCK + tid];
    float s = sqrt(x1*x1 + x2*x2);
    has_left[tid] = s >= 1.0f;
}

// Error estimation, step acceptance and step-size control; a port of
// cuda_kernel.cu::adjust_time.
inline void adjust_time(threadgroup float* t,
                        threadgroup float* dt,
                        threadgroup const float* tmax,
                        threadgroup float* state,
                        device float* derivs,
                        threadgroup const float* x_temp,
                        threadgroup bool* has_left,
                        threadgroup const float* dtmax,
                        threadgroup const bool* is_valid,
                        constant TraceParams& P,
                        uint tid) {
    const int p = tid % PARTICLES_PER_BLOCK;        // particle
    const int state_id = tid / PARTICLES_PER_BLOCK; // state variable

    const bool active = is_valid[p] && !(has_left[p] || t[p] >= tmax[p]);
    const float dt_p = dt[p];

    float error_elt = 0.0f;
    if (active) {
        const float state_i = state[state_id*PARTICLES_PER_BLOCK + p];
        const float deriv_i = derivs[(N_DERIV_OUTPUTS*0 + state_id)*PARTICLES_PER_BLOCK + p];
        error_elt = bhat_wgts[0]*deriv_i;
        for (int j = 2; j < 7; ++j) {
            error_elt += bhat_wgts[j]*derivs[(N_DERIV_OUTPUTS*j + state_id)*PARTICLES_PER_BLOCK + p];
        }
        error_elt *= dt_p;
        const float atol_i = (P.rescale_abstol != 0) && (state_id == 3)
                           ? P.atol * P.v_total : P.atol;
        error_elt = fabs(error_elt) / (atol_i + P.rtol*(fabs(state_i) + dt_p*fabs(deriv_i)));
    }

    // maximum error across the 4 state variables of each particle
    error_elt = max(error_elt, simd_shuffle_down(error_elt, 16));
    error_elt = max(error_elt, simd_shuffle_down(error_elt, 8));

    const float max_err = simd_shuffle(error_elt, (ushort)p);
    const bool accept = active && (max_err <= 1.0f);
    if (accept) {
        float new_val = x_temp[(state_id + 1)*PARTICLES_PER_BLOCK + p];
        if (state_id == 2) { // wrap zeta to [0, 2pi] to avoid large values
            new_val = fmod(new_val, 2.0f*M_PI_F);
            new_val += 2.0f*M_PI_F*(new_val < 0.0f);
        }
        state[state_id*PARTICLES_PER_BLOCK + p] = new_val;

        // first-same-as-last: stage 6 derivative becomes stage 0 of the next step
        derivs[(N_DERIV_OUTPUTS*0 + state_id)*PARTICLES_PER_BLOCK + p] =
            derivs[(N_DERIV_OUTPUTS*6 + state_id)*PARTICLES_PER_BLOCK + p];
    }
    if (active && state_id == 0) { // one thread per particle
        float dt_new = dt_p*0.9f;
        float exponent = 0.0f;
        if (max_err > 1.0f) {
            exponent = -1.0f/3.0f;
        }
        if (max_err < 0.5f) {
            exponent = -1.0f/5.0f;
        }
        dt_new *= pow(max_err, exponent);
        dt_new = max(dt_new, 0.2f * dt_p);
        dt_new = min(dt_new, 5.0f * dt_p);

        if (accept) {
            if (0.5f < max_err) {
                dt_new = dt_p;
            }
            t[p] += dt_p;
        }
        dt_new = min(dt_new, dtmax[p]);
        dt[p] = dt_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (accept && state_id == 0) {
        check_has_left_boozer(has_left, state, tid);
    }
}

// Compute mu, dtmax, and the initial dt where the host passed the -1 sentinel;
// a port of setup_particle/setup_kernel for Boozer coordinates.
kernel void setup_boozer(device const float* state_g [[buffer(0)]],
                         device const float* quadpts_arr [[buffer(1)]],
                         device float* mu_g [[buffer(2)]],
                         device float* dt_g [[buffer(3)]],
                         device float* dtmax_g [[buffer(4)]],
                         device const float* t_g [[buffer(5)]],
                         device float* derivs_g [[buffer(6)]],
                         constant TraceParams& P [[buffer(7)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint bid [[threadgroup_position_in_grid]]) {
    uint idx = tid + bid*PARTICLES_PER_BLOCK;

    threadgroup float x_temp[5*PARTICLES_PER_BLOCK];
    threadgroup bool symmetry_exploited[PARTICLES_PER_BLOCK];
    threadgroup int cell_index_start[3*PARTICLES_PER_BLOCK];
    threadgroup float shape_fun_vals[12*PARTICLES_PER_BLOCK];
    threadgroup float state[4*PARTICLES_PER_BLOCK];
    threadgroup float interp_pt[3*PARTICLES_PER_BLOCK];
    threadgroup float block_interpolants[MAX_INTERPOLANTS*PARTICLES_PER_BLOCK];
    threadgroup bool is_valid_arr[PARTICLES_PER_BLOCK];
    threadgroup float block_t[PARTICLES_PER_BLOCK];
    threadgroup float block_dt[PARTICLES_PER_BLOCK];
    threadgroup float block_dtmax[PARTICLES_PER_BLOCK];
    threadgroup float block_mu[PARTICLES_PER_BLOCK];

    device float* block_derivs = derivs_g + bid*PARTICLES_PER_BLOCK*7*N_DERIV_OUTPUTS;

    bool is_valid = idx < (uint)P.nparticles && tid < (uint)PARTICLES_PER_BLOCK;
    if (tid < (uint)PARTICLES_PER_BLOCK) {
        is_valid_arr[tid] = is_valid;
    }
    if (is_valid) {
        for (int i = 0; i < 4; ++i) {
            state[i*PARTICLES_PER_BLOCK + tid] = state_g[4*idx + i];
        }
        block_t[tid] = t_g[idx];
        block_dt[tid] = dt_g[idx];
        block_mu[tid] = mu_g[idx];
        symmetry_exploited[tid] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    build_state(0, x_temp, symmetry_exploited, cell_index_start, shape_fun_vals,
                state, block_derivs, block_t, block_dt, is_valid_arr, interp_pt, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int n = n_interpolants(P.rhs_id);
    interpolate(n, block_interpolants, quadpts_arr, cell_index_start, shape_fun_vals,
                is_valid_arr, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_valid) {
        float v_par = state[3*PARTICLES_PER_BLOCK + tid];
        float v_perp2 = P.v_total*P.v_total - v_par*v_par;
        float modB = block_interpolants[0*PARTICLES_PER_BLOCK + tid];

        if (block_mu[tid] == -1.0f) { // sentinel: mu needs to be computed
            block_mu[tid] = v_perp2 / (2.0f*modB);
        }

        max_stepsize_boozer(block_dtmax, block_interpolants, P, tid);

        if (block_dt[tid] == -1.0f) { // sentinel: dt needs to be computed
            block_dt[tid] = 1e-3f*block_dtmax[tid];
        }

        mu_g[idx] = block_mu[tid];
        dt_g[idx] = block_dt[tid];
        dtmax_g[idx] = block_dtmax[tid];
    }
}

// Advance particles by up to P.max_steps accepted-or-rejected DP5 steps; the
// bounded, fixed-assignment port of cuda_kernel.cu::particle_trace_kernel.
// All per-particle state lives in device buffers so the host can redispatch
// until every particle reaches tmax or leaves the plasma.
kernel void trace_boozer(device float* state_g [[buffer(0)]],
                         device const float* quadpts_arr [[buffer(1)]],
                         device float* derivs_g [[buffer(2)]],
                         device const float* mu_g [[buffer(3)]],
                         device const float* tmax_g [[buffer(4)]],
                         device float* t_g [[buffer(5)]],
                         device float* dt_g [[buffer(6)]],
                         device const float* dtmax_g [[buffer(7)]],
                         device uchar* has_left_g [[buffer(8)]],
                         constant TraceParams& P [[buffer(9)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint bid [[threadgroup_position_in_grid]]) {
    uint idx = tid + bid*PARTICLES_PER_BLOCK;

    threadgroup float x_temp[5*PARTICLES_PER_BLOCK];
    threadgroup bool symmetry_exploited[PARTICLES_PER_BLOCK];
    threadgroup int cell_index_start[3*PARTICLES_PER_BLOCK];
    threadgroup float shape_fun_vals[12*PARTICLES_PER_BLOCK];
    threadgroup float state[4*PARTICLES_PER_BLOCK];
    threadgroup float interp_pt[3*PARTICLES_PER_BLOCK];
    threadgroup float block_interpolants[MAX_INTERPOLANTS*PARTICLES_PER_BLOCK];
    threadgroup bool is_valid_arr[PARTICLES_PER_BLOCK];
    threadgroup bool has_left[PARTICLES_PER_BLOCK];
    threadgroup bool needs_stage0[PARTICLES_PER_BLOCK];
    threadgroup float block_t[PARTICLES_PER_BLOCK];
    threadgroup float block_dt[PARTICLES_PER_BLOCK];
    threadgroup float block_tmax[PARTICLES_PER_BLOCK];
    threadgroup float block_dtmax[PARTICLES_PER_BLOCK];
    threadgroup float block_mu[PARTICLES_PER_BLOCK];

    device float* block_derivs = derivs_g + bid*PARTICLES_PER_BLOCK*7*N_DERIV_OUTPUTS;

    bool is_valid = idx < (uint)P.nparticles && tid < (uint)PARTICLES_PER_BLOCK;
    if (tid < (uint)PARTICLES_PER_BLOCK) {
        is_valid_arr[tid] = is_valid;
    }
    if (is_valid) {
        for (int i = 0; i < 4; ++i) {
            state[i*PARTICLES_PER_BLOCK + tid] = state_g[4*idx + i];
        }
        block_t[tid] = t_g[idx];
        block_dt[tid] = dt_g[idx];
        block_tmax[tid] = tmax_g[idx];
        block_dtmax[tid] = dtmax_g[idx];
        block_mu[tid] = mu_g[idx];
        has_left[tid] = has_left_g[idx] != 0;
        symmetry_exploited[tid] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int step = 0;
    while (step < P.max_steps &&
           tg_count(is_valid_arr[tid % PARTICLES_PER_BLOCK] &&
                    !(block_t[tid % PARTICLES_PER_BLOCK] >= block_tmax[tid % PARTICLES_PER_BLOCK] ||
                      has_left[tid % PARTICLES_PER_BLOCK])) > 0) {
        ++step;

        // particles at t = 0 have no stage-0 derivative yet; afterwards the
        // first-same-as-last slot written by adjust_time persists, including
        // across dispatches, because derivs_g is indexed by the fixed slot
        if (tid < (uint)PARTICLES_PER_BLOCK) {
            needs_stage0[tid] = is_valid_arr[tid] && (block_t[tid] == 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tg_count(needs_stage0[tid % PARTICLES_PER_BLOCK]) > 0) {
            dp5_one_step(0, x_temp, block_derivs, quadpts_arr, cell_index_start,
                         shape_fun_vals, block_t, block_dt, symmetry_exploited, state,
                         block_mu, needs_stage0, interp_pt, block_interpolants, P, tid);
        }
        for (int stage = 1; stage < 7; ++stage) {
            dp5_one_step(stage, x_temp, block_derivs, quadpts_arr, cell_index_start,
                         shape_fun_vals, block_t, block_dt, symmetry_exploited, state,
                         block_mu, is_valid_arr, interp_pt, block_interpolants, P, tid);
        }
        adjust_time(block_t, block_dt, block_tmax, state, block_derivs, x_temp,
                    has_left, block_dtmax, is_valid_arr, P, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (is_valid) {
        for (int i = 0; i < 4; ++i) {
            state_g[4*idx + i] = state[i*PARTICLES_PER_BLOCK + tid];
        }
        t_g[idx] = block_t[tid];
        dt_g[idx] = block_dt[tid];
        has_left_g[idx] = has_left[tid] ? 1 : 0;
    }
}

// Evaluate the interpolated field components at given points; a port of
// test_gpu_interpolation_kernel, with the stellarator-symmetry correction of
// account_for_symmetry_rhs applied to the output.
kernel void test_interp_boozer(device const float* quadpts_arr [[buffer(0)]],
                               device const float* loc [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant TraceParams& P [[buffer(3)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint bid [[threadgroup_position_in_grid]]) {
    uint idx = tid + bid*PARTICLES_PER_BLOCK;

    threadgroup float x_temp[5*PARTICLES_PER_BLOCK];
    threadgroup bool symmetry_exploited[PARTICLES_PER_BLOCK];
    threadgroup int cell_index_start[3*PARTICLES_PER_BLOCK];
    threadgroup float shape_fun_vals[12*PARTICLES_PER_BLOCK];
    threadgroup float state[4*PARTICLES_PER_BLOCK];
    threadgroup float interp_pt[3*PARTICLES_PER_BLOCK];
    threadgroup float block_interpolants[MAX_INTERPOLANTS*PARTICLES_PER_BLOCK];
    threadgroup bool is_valid_arr[PARTICLES_PER_BLOCK];
    threadgroup float block_t[PARTICLES_PER_BLOCK];
    threadgroup float block_dt[PARTICLES_PER_BLOCK];

    int n = n_interpolants(P.rhs_id);

    bool is_valid = idx < (uint)P.nparticles && tid < (uint)PARTICLES_PER_BLOCK;
    if (tid < (uint)PARTICLES_PER_BLOCK) {
        is_valid_arr[tid] = is_valid;
    }
    if (is_valid) {
        block_dt[tid] = 1e-3f; // needed for build_state
        block_t[tid] = 0.0f;
        symmetry_exploited[tid] = false;
        for (int i = 0; i < 3; ++i) {
            state[i*PARTICLES_PER_BLOCK + tid] = loc[3*idx + i];
        }
        state[3*PARTICLES_PER_BLOCK + tid] = 0.0f; // dummy vpar
        for (int i = 0; i < n; ++i) {
            block_interpolants[i*PARTICLES_PER_BLOCK + tid] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // stage 0 never reads the derivative buffer, so quadpts_arr is a safe dummy
    build_state(0, x_temp, symmetry_exploited, cell_index_start, shape_fun_vals,
                state, quadpts_arr, block_t, block_dt, is_valid_arr, interp_pt, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    interpolate(n, block_interpolants, quadpts_arr, cell_index_start, shape_fun_vals,
                is_valid_arr, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_valid) {
        device float* out_arr = out + idx*n;
        for (int i = 0; i < n; ++i) {
            out_arr[i] = block_interpolants[i*PARTICLES_PER_BLOCK + tid];
        }
        if (symmetry_exploited[tid]) {
            out_arr[2] *= -1.0f; // d|B|/dtheta
            out_arr[3] *= -1.0f; // d|B|/dzeta
            if (P.rhs_id == RHS_BOOZER && n >= 12) {
                out_arr[9] *= -1.0f; // K
            }
        }
    }
}

// Evaluate the guiding-center RHS at given phase-space points (after
// setup_boozer has computed mu); a port of test_gpu_derivs_kernel.
kernel void test_derivs_boozer(device const float* quadpts_arr [[buffer(0)]],
                               device const float* state_g [[buffer(1)]],
                               device const float* t_g [[buffer(2)]],
                               device float* out [[buffer(3)]],
                               device float* derivs_g [[buffer(4)]],
                               device const float* mu_g [[buffer(5)]],
                               device const float* dt_g [[buffer(6)]],
                               constant TraceParams& P [[buffer(7)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint bid [[threadgroup_position_in_grid]]) {
    uint idx = tid + bid*PARTICLES_PER_BLOCK;

    threadgroup float x_temp[5*PARTICLES_PER_BLOCK];
    threadgroup bool symmetry_exploited[PARTICLES_PER_BLOCK];
    threadgroup int cell_index_start[3*PARTICLES_PER_BLOCK];
    threadgroup float shape_fun_vals[12*PARTICLES_PER_BLOCK];
    threadgroup float state[4*PARTICLES_PER_BLOCK];
    threadgroup float interp_pt[3*PARTICLES_PER_BLOCK];
    threadgroup float block_interpolants[MAX_INTERPOLANTS*PARTICLES_PER_BLOCK];
    threadgroup bool is_valid_arr[PARTICLES_PER_BLOCK];
    threadgroup float block_t[PARTICLES_PER_BLOCK];
    threadgroup float block_dt[PARTICLES_PER_BLOCK];
    threadgroup float block_mu[PARTICLES_PER_BLOCK];

    device float* block_derivs = derivs_g + bid*PARTICLES_PER_BLOCK*7*N_DERIV_OUTPUTS;

    bool is_valid = idx < (uint)P.nparticles && tid < (uint)PARTICLES_PER_BLOCK;
    if (tid < (uint)PARTICLES_PER_BLOCK) {
        is_valid_arr[tid] = is_valid;
    }
    if (is_valid) {
        for (int i = 0; i < 4; ++i) {
            state[i*PARTICLES_PER_BLOCK + tid] = state_g[4*idx + i];
        }
        block_t[tid] = t_g[idx];
        block_dt[tid] = dt_g[idx];
        block_mu[tid] = mu_g[idx];
        symmetry_exploited[tid] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    build_state(0, x_temp, symmetry_exploited, cell_index_start, shape_fun_vals,
                state, block_derivs, block_t, block_dt, is_valid_arr, interp_pt, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    calc_derivs(0, block_derivs, quadpts_arr, x_temp, symmetry_exploited,
                cell_index_start, shape_fun_vals, block_mu, is_valid_arr,
                block_interpolants, P, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_valid) {
        for (int i = 0; i < 4; ++i) {
            out[4*idx + i] = block_derivs[i*PARTICLES_PER_BLOCK + tid];
        }
    }
}
