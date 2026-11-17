// Layout experiment: appended to firm3d_metal's metal_kernel.metal at compile time,
// so it reuses that file's TraceParams, DP5 tables, dp5_one_step and adjust_time.
//
// trace_boozer_x : trace_boozer plus a per-particle step counter (buffer 10). With
//                  function constant GATE_ACTIVE = true, the DP5 stages are gated on
//                  "still active" instead of "valid", so finished particles in a
//                  threadgroup stop issuing table loads.
// trace_boozer_1t: one thread per particle, the same numerics (vacuum and general RHS),
//                  with the per-lane partial sums and the 8-lane shuffle_down
//                  reduction order of interpolate() reproduced, so that results can
//                  match the cooperative kernel bit for bit.

constant bool GATE_ACTIVE [[function_constant(0)]];

kernel void trace_boozer_x(device float* state_g [[buffer(0)]],
                           device const float* quadpts_arr [[buffer(1)]],
                           device float* derivs_g [[buffer(2)]],
                           device const float* mu_g [[buffer(3)]],
                           device const float* tmax_g [[buffer(4)]],
                           device float* t_g [[buffer(5)]],
                           device float* dt_g [[buffer(6)]],
                           device const float* dtmax_g [[buffer(7)]],
                           device uchar* has_left_g [[buffer(8)]],
                           constant TraceParams& P [[buffer(9)]],
                           device uint* steps_g [[buffer(10)]],
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
    threadgroup bool active_arr[PARTICLES_PER_BLOCK];
    threadgroup bool has_left[PARTICLES_PER_BLOCK];
    threadgroup bool needs_stage0[PARTICLES_PER_BLOCK];
    threadgroup float block_t[PARTICLES_PER_BLOCK];
    threadgroup float block_dt[PARTICLES_PER_BLOCK];
    threadgroup float block_tmax[PARTICLES_PER_BLOCK];
    threadgroup float block_dtmax[PARTICLES_PER_BLOCK];
    threadgroup float block_mu[PARTICLES_PER_BLOCK];
    threadgroup uint block_steps[PARTICLES_PER_BLOCK];

    device float* block_derivs = derivs_g + bid*PARTICLES_PER_BLOCK*7*N_DERIV_OUTPUTS;

    bool is_valid = idx < (uint)P.nparticles && tid < (uint)PARTICLES_PER_BLOCK;
    if (tid < (uint)PARTICLES_PER_BLOCK) {
        is_valid_arr[tid] = is_valid;
        active_arr[tid] = false;
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
        block_steps[tid] = steps_g[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const bool* gate = GATE_ACTIVE ? active_arr : is_valid_arr;

    int step = 0;
    while (step < P.max_steps &&
           tg_count(is_valid_arr[tid % PARTICLES_PER_BLOCK] &&
                    !(block_t[tid % PARTICLES_PER_BLOCK] >= block_tmax[tid % PARTICLES_PER_BLOCK] ||
                      has_left[tid % PARTICLES_PER_BLOCK])) > 0) {
        ++step;

        if (tid < (uint)PARTICLES_PER_BLOCK) {
            const bool act = is_valid_arr[tid] && !(block_t[tid] >= block_tmax[tid] || has_left[tid]);
            active_arr[tid] = act;
            needs_stage0[tid] = (GATE_ACTIVE ? act : is_valid_arr[tid]) && (block_t[tid] == 0.0f);
            if (act) {
                block_steps[tid] += 1;
            }
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
                         block_mu, gate, interp_pt, block_interpolants, P, tid);
        }
        adjust_time(block_t, block_dt, block_tmax, state, block_derivs, x_temp,
                    has_left, block_dtmax, is_valid_arr, P, tid);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    if (is_valid) {
        for (int i = 0; i < 4; ++i) {
            state_g[4*idx + i] = state[i*PARTICLES_PER_BLOCK + tid];
        }
        t_g[idx] = block_t[tid];
        dt_g[idx] = block_dt[tid];
        has_left_g[idx] = has_left[tid] ? 1 : 0;
        steps_g[idx] = block_steps[tid];
    }
}

// Interpolate N tabulated quantities at one point, one thread: the per-lane partial
// sums of interpolate() and its 8-lane shuffle_down reduction order. shp holds the
// shape functions as build_state stores them per particle: s at 0-3, theta 4-7, zeta 8-11.
template <int N>
inline void interpolate_1t(thread float* interp,
                           device const float* quadpts_arr,
                           int cell_linear,
                           thread const float* shp) {
    const int cell_base = 64*N*cell_linear;
    for (int zz = 0; zz < N; ++zz) {
        float lane[8];
        for (int L = 0; L < 8; ++L) {
            const int kk = L % 4;
            const int jj = L / 4;
            const int base_offset = cell_base + 4*jj + kk;
            const float shape_jk1 = shp[8 + kk] * shp[4 + jj];
            const float shape_jk2 = shp[8 + kk] * shp[4 + jj + 2];
            float local_val = 0.0f;
            for (int ii = 0; ii < 4; ++ii) {
                const int base = base_offset + 16*ii;
                const float shape_i = shp[ii];
                local_val += shape_i * (shape_jk1 * quadpts_arr[base + 64*zz] +
                                        shape_jk2 * quadpts_arr[base + 8 + 64*zz]);
            }
            lane[L] = local_val;
        }
        // simd_shuffle_down tree over 8 lanes: offsets 4, 2, 1, result in lane 0
        const float v0 = lane[0] + lane[4];
        const float v1 = lane[1] + lane[5];
        const float v2 = lane[2] + lane[6];
        const float v3 = lane[3] + lane[7];
        const float w0 = v0 + v2;
        const float w1 = v1 + v3;
        interp[zz] = w0 + w1;
    }
}

// Guiding-center RHS at stage state x = (x1, x2, zeta, vpar), one thread:
// map_to_grid_boozer, the shape functions of build_state, interpolate() and
// rhs_boozer_vacuum / rhs_boozer_general, with the same operation order.
inline void eval_rhs_1t(thread const float* x,
                        thread float* kout,
                        device const float* quadpts_arr,
                        float mu_val,
                        constant TraceParams& P) {
    const float x1 = x[0];
    const float x2 = x[1];
    const float s = sqrt(x1*x1 + x2*x2);
    const bool sym = x2 < 0.0f;
    const float th = atan2(fabs(x2), x1);
    float z = x[2];
    const float period = P.grid_ranges[9];
    z = fmod(z, period);
    z += period*(z < 0.0f);
    if (sym) {
        z = period - z;
    }
    const float pt[3] = {s, th, z};

    float shp[12];
    int cell[3];
    for (int c = 0; c < 3; ++c) {
        const float inv_grid_size = P.grid_ranges[c*4 + 3];
        const float min_bound = P.grid_ranges[c*4 + 0];
        const float raw_offset = (pt[c] - min_bound) * inv_grid_size;
        int index = 3*((int)raw_offset / 3);
        index = min(index, (int)P.grid_ranges[c*4 + 2] - 4);
        index = max(index, 0);
        const float value_rel = raw_offset - (float)index;
        const float x_minus_1 = value_rel - 1.0f;
        const float x_minus_2 = value_rel - 2.0f;
        const float x_minus_3 = value_rel - 3.0f;
        const float one_sixth = 1.0f/6.0f;
        const float one_half = 0.5f;
        const float prod23 = x_minus_2 * x_minus_3;
        const float prodx1 = value_rel * x_minus_1;
        shp[4*c + 0] = -one_sixth * x_minus_1 * prod23;
        shp[4*c + 1] = one_half * value_rel * prod23;
        shp[4*c + 2] = -one_half * prodx1 * x_minus_3;
        shp[4*c + 3] = one_sixth * prodx1 * x_minus_2;
        cell[c] = index/3;
    }
    const int cell_linear = cell[0]*P.n_x2*P.n_x3 + cell[1]*P.n_x3 + cell[2];
    const float v_par = x[3];
    float interp[MAX_INTERPOLANTS];

    if (P.rhs_id == RHS_BOOZER_VACUUM) {
        interpolate_1t<6>(interp, quadpts_arr, cell_linear, shp);
        const float inv_s = 1.0f / sqrt(x1*x1 + x2*x2);
        const float modB = interp[0];
        const float dmodBds = interp[1];
        float dmodBdtheta = interp[2];
        float dmodBdzeta = interp[3];
        const float G = interp[4];
        const float iota = interp[5];
        const float modB_inv_G = modB / G;
        const float sign = sym ? -1.0f : 1.0f;
        dmodBdtheta *= sign;
        dmodBdzeta *= sign;
        const float fak1 = P.mass*v_par*v_par/modB + P.mass*mu_val;
        const float sdot = -dmodBdtheta*(fak1 * P.inv_psi0_charge);
        const float tdot = dmodBds*(fak1 * P.inv_psi0_charge) + iota*(v_par*modB_inv_G);
        kout[0] = sdot*x1*inv_s - x2*tdot;
        kout[1] = sdot*x2*inv_s + x1*tdot;
        kout[2] = v_par*modB_inv_G;
        kout[3] = -(iota*dmodBdtheta + dmodBdzeta)*mu_val*modB_inv_G;
    } else {
        interpolate_1t<12>(interp, quadpts_arr, cell_linear, shp);
        const float theta = atan2(x2, x1);
        const float modB = interp[0];
        const float dmodBdpsi = interp[1] / P.psi0;
        float dmodBdtheta = interp[2];
        float dmodBdzeta = interp[3];
        const float G = interp[4];
        const float dGdpsi = interp[5] / P.psi0;
        const float I = interp[6];
        const float dIdpsi = interp[7] / P.psi0;
        const float iota = interp[8];
        float K = interp[9];
        const float dKdtheta = interp[10];
        const float dKdzeta = interp[11];
        if (sym) {
            dmodBdtheta *= -1.0f;
            dmodBdzeta *= -1.0f;
            K *= -1.0f;
        }
        const float C = -P.mass * v_par * dKdzeta / modB - P.charge * iota + P.mass * v_par * dGdpsi / modB;
        const float F = -P.mass * v_par * dKdtheta / modB + P.charge + P.mass * v_par * dIdpsi / modB;
        const float D = (F * G - C * I) / iota;
        const float fak1 = P.mass * v_par * v_par / modB + P.mass * mu_val;
        const float sdot = (I * dmodBdzeta - G * dmodBdtheta) * fak1 / (iota * D * P.psi0);
        const float tdot = ((G * dmodBdpsi - K * dmodBdzeta) * fak1 - C * v_par * modB) / (iota * D);
        const float zetadot = (F * v_par * modB - (dmodBdpsi * I - dmodBdtheta * K) * fak1) / (iota * D);
        const float vpardot = (C * dmodBdtheta - F * dmodBdzeta) * mu_val * modB / (iota * D);
        kout[0] = sdot*cos(theta) - s*sin(theta)*tdot;
        kout[1] = sdot*sin(theta) + s*cos(theta)*tdot;
        kout[2] = zetadot;
        kout[3] = vpardot;
    }
}

// One thread per particle; same buffers as trace_boozer except buffer 2, which
// holds only each particle's first-same-as-last derivative (4 floats).
kernel void trace_boozer_1t(device float* state_g [[buffer(0)]],
                            device const float* quadpts_arr [[buffer(1)]],
                            device float* fsal_g [[buffer(2)]],
                            device const float* mu_g [[buffer(3)]],
                            device const float* tmax_g [[buffer(4)]],
                            device float* t_g [[buffer(5)]],
                            device float* dt_g [[buffer(6)]],
                            device const float* dtmax_g [[buffer(7)]],
                            device uchar* has_left_g [[buffer(8)]],
                            constant TraceParams& P [[buffer(9)]],
                            device uint* steps_g [[buffer(10)]],
                            uint idx [[thread_position_in_grid]]) {
    if (idx >= (uint)P.nparticles) {
        return;
    }
    float y[4];
    for (int v = 0; v < 4; ++v) {
        y[v] = state_g[4*idx + v];
    }
    float t = t_g[idx];
    float dt = dt_g[idx];
    const float tmax = tmax_g[idx];
    const float dtmax = dtmax_g[idx];
    const float mu = mu_g[idx];
    bool left = has_left_g[idx] != 0;
    uint nsteps = steps_g[idx];

    float k[7][4];
    if (t != 0.0f) {
        for (int v = 0; v < 4; ++v) {
            k[0][v] = fsal_g[4*idx + v];
        }
    }

    int step = 0;
    while (step < P.max_steps && !(t >= tmax || left)) {
        ++step;
        ++nsteps;
        if (t == 0.0f) {
            eval_rhs_1t(y, k[0], quadpts_arr, mu, P);
        }
        float x[4];
        for (int stage = 1; stage < 7; ++stage) {
            for (int v = 0; v < 4; ++v) {
                float val = y[v];
                for (int j = 0; j < stage; ++j) {
                    val += dt * dp5_wgts[stage][j] * k[j][v];
                }
                x[v] = val;
            }
            eval_rhs_1t(x, k[stage], quadpts_arr, mu, P);
        }

        float err[4];
        for (int v = 0; v < 4; ++v) {
            const float state_i = y[v];
            const float deriv_i = k[0][v];
            float e = bhat_wgts[0]*deriv_i;
            for (int j = 2; j < 7; ++j) {
                e += bhat_wgts[j]*k[j][v];
            }
            e *= dt;
            const float atol_i = (P.rescale_abstol != 0) && (v == 3) ? P.atol * P.v_total : P.atol;
            err[v] = fabs(e) / (atol_i + P.rtol*(fabs(state_i) + dt*fabs(deriv_i)));
        }
        // adjust_time's shuffle order: offset 16 pairs (0,2), (1,3); offset 8 pairs those
        const float max_err = max(max(err[0], err[2]), max(err[1], err[3]));
        const bool accept = max_err <= 1.0f;
        if (accept) {
            for (int v = 0; v < 4; ++v) {
                float new_val = x[v];
                if (v == 2) {
                    new_val = fmod(new_val, 2.0f*M_PI_F);
                    new_val += 2.0f*M_PI_F*(new_val < 0.0f);
                }
                y[v] = new_val;
                k[0][v] = k[6][v];
            }
        }
        float dt_new = dt*0.9f;
        float exponent = 0.0f;
        if (max_err > 1.0f) {
            exponent = -1.0f/3.0f;
        }
        if (max_err < 0.5f) {
            exponent = -1.0f/5.0f;
        }
        dt_new *= pow(max_err, exponent);
        dt_new = max(dt_new, 0.2f * dt);
        dt_new = min(dt_new, 5.0f * dt);
        if (accept) {
            if (0.5f < max_err) {
                dt_new = dt;
            }
            t += dt;
        }
        dt_new = min(dt_new, dtmax);
        dt = dt_new;
        if (accept) {
            left = sqrt(y[0]*y[0] + y[1]*y[1]) >= 1.0f;
        }
    }

    for (int v = 0; v < 4; ++v) {
        state_g[4*idx + v] = y[v];
        fsal_g[4*idx + v] = k[0][v];
    }
    t_g[idx] = t;
    dt_g[idx] = dt;
    has_left_g[idx] = left ? 1 : 0;
    steps_g[idx] = nsteps;
}
