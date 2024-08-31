# Copyright (c) 2024, Sukjun Hwang
# This is an implementation of Flash Attention: https://tridao.me/publications/flash2/flash2.pdf
# Based on the Triton attention tutorial: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K, V, Bias,
    start_N, num_steps, qk_scale,
    stride_k_N, stride_k_D,
    stride_v_N, stride_v_D,
    stride_bias_N,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    offs_M: tl.constexpr, mask_M: tl.constexpr, N_CTX: tl.constexpr, 
    IS_CAUSAL: tl.constexpr, IS_BIAS: tl.constexpr, IS_BIAS_BOOL: tl.constexpr
):
    # q: (M, Dqk)
    K += start_N * stride_k_N # (Dqk, N)
    V += start_N * stride_v_N # (N, Dv)
    if IS_BIAS:
        Bias = Bias + start_N * stride_bias_N # (M, N)
    # loop over k, v and update accumulator
    curr_N = start_N
    for blk_idx in range(num_steps):
        curr_N = tl.multiple_of(curr_N, BLOCK_N)
        offs_N = curr_N + tl.arange(0, BLOCK_N)
        mask_N = offs_N < N_CTX

        # -- compute qk ----
        k = tl.load(K, mask_N[None, :], other=0.0) # (Dqk, N)
        qk = tl.dot(q, k) # (M, N)
        qk = qk * qk_scale

        mask_MN = mask_M[:, None] & mask_N[None, :]
        bias = tl.where(mask_MN, 0.0, -1.0e6)
        if IS_CAUSAL:
            # Autoregressive masking.
            causal_mask = offs_M[:, None] >= offs_N[None, :]
            bias += tl.where(causal_mask, 0.0, -1.0e6)
        if IS_BIAS:
            if IS_BIAS_BOOL: # attention mask
                attn_mask = tl.load(Bias, mask=mask_MN, other=False)
                bias += tl.where(attn_mask, 0.0, -1.0e6)
            else: # attention bias
                bias += tl.load(Bias, mask=mask_MN, other=-1.0e6).to(tl.float32)
        qk += bias # (M, N)
        m_ij = tl.maximum(m_i, tl.max(qk, 1)) # (M,)
        qk -= m_ij[:, None] # (M, N)

        p = tl.math.exp2(qk) # (M, N)
        l_ij = tl.sum(p, 1) # (M,)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij) # (M,)
        l_i = l_i * alpha + l_ij # (M,)
        # -- update output accumulator --
        acc = acc * alpha[:, None] # (M, Dv)
        # update acc
        v = tl.load(V, mask=mask_N[:, None], other=0.0) # (N, Dv)
        p = p.to(q.dtype)
        acc = tl.dot(p, v, acc) # pv + acc (fp16 + fp32) (M, Dv) 
        # update m_i and l_i
        m_i = m_ij # (M,)
        K += BLOCK_N * stride_k_N
        V += BLOCK_N * stride_v_N
        if IS_BIAS:
            Bias += BLOCK_N * stride_bias_N
        curr_N += BLOCK_N

    # (M, Dv), (M,), (M,)
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
    ],
    key=["M_CTX", "N_CTX", "HEAD_DIM_QK", "HEAD_DIM_V"],
)
@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, O, Bias,
    stride_q_B, stride_q_M, stride_q_H, stride_q_D,
    stride_k_B, stride_k_N, stride_k_H, stride_k_D,
    stride_v_B, stride_v_N, stride_v_H, stride_v_D,
    stride_o_B, stride_o_M, stride_o_H, stride_o_D,
    stride_m_B, stride_m_M, stride_m_H,
    stride_bias_B, stride_bias_H, stride_bias_M, stride_bias_N,
    B, H, M_CTX, N_CTX,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BIAS: tl.constexpr,
    IS_BIAS_BOOL: tl.constexpr,
):
    pid_M = tl.program_id(0)
    pid_BH = tl.program_id(1)
    start_M = pid_M * BLOCK_M
    offs_B = (pid_BH // H).to(tl.int64)
    offs_H = (pid_BH % H).to(tl.int64)
    q_offset = offs_B * stride_q_B + offs_H * stride_q_H
    k_offset = offs_B * stride_k_B + offs_H * stride_k_H
    v_offset = offs_B * stride_v_B + offs_H * stride_v_H
    o_offset = offs_B * stride_o_B + offs_H * stride_o_H

    Q += q_offset + (
        (start_M + tl.arange(0, BLOCK_M))[:, None] * stride_q_M
        + tl.arange(0, HEAD_DIM_QK)[None, :] * stride_q_D
    )
    K += k_offset + (
        tl.arange(0, BLOCK_N)[None, :] * stride_k_N
        + tl.arange(0, HEAD_DIM_QK)[:, None] * stride_k_D
    )
    V += v_offset + (
        tl.arange(0, BLOCK_N)[:, None] * stride_v_N
        + tl.arange(0, HEAD_DIM_V)[None, :] * stride_v_D
    )
    O += o_offset + (
        (start_M + tl.arange(0, BLOCK_M))[:, None] * stride_o_M
        + tl.arange(0, HEAD_DIM_V)[None, :] * stride_o_D
    )
    if IS_BIAS:
        Bias += (
            offs_B * stride_bias_B
            + offs_H * stride_bias_H
            + (start_M + tl.arange(0, BLOCK_M))[:, None] * stride_bias_M
            + tl.arange(0, BLOCK_N)[None, :] * stride_bias_N
        )

    # initialize offsets
    offs_M = start_M + tl.arange(0, BLOCK_M)
    mask_M = offs_M < M_CTX

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634 # = 1.0 / ln(2)

    # load q: it will stay in SRAM throughout
    q = tl.load(Q, mask=mask_M[:, None], other=0.0) # (M, Dqk)

    # on-band
    if IS_CAUSAL:
        num_steps = BLOCK_M // BLOCK_N
        start_N = start_M
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K, V, Bias,
            start_N, num_steps, qk_scale,
            stride_k_N, stride_k_D,
            stride_v_N, stride_v_D,
            stride_bias_N,
            BLOCK_M, HEAD_DIM_V, BLOCK_N,
            offs_M, mask_M, N_CTX,
            True, IS_BIAS, IS_BIAS_BOOL,
        )
        num_steps = start_N // BLOCK_N
        start_N = 0
    else:
        num_steps = (N_CTX + BLOCK_N - 1) // BLOCK_N
        start_N = 0

    # off-band
    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q, K, V, Bias,
        start_N, num_steps, qk_scale,
        stride_k_N, stride_k_D,
        stride_v_N, stride_v_D,
        stride_bias_N,
        BLOCK_M, HEAD_DIM_V, BLOCK_N,
        offs_M, mask_M, N_CTX,
        False, IS_BIAS, IS_BIAS_BOOL,
    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # M: (B, H, M)
    M += offs_B * stride_m_B + offs_H * stride_m_H + offs_M * stride_m_M
    tl.store(M, m_i, mask=mask_M) # (M,)
    tl.store(O, acc.to(q.dtype), mask=mask_M[:, None]) # (M, Dv)


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    stride_o_B, stride_o_M, stride_o_H, stride_o_D,
    stride_do_B, stride_do_M, stride_do_H, stride_do_D,
    stride_delta_B, stride_delta_M, stride_delta_H,
    B, H, M_CTX,
    BLOCK_M: tl.constexpr, HEAD_DIM_QK: tl.constexpr,
):
    pid_M = tl.program_id(0)
    pid_BH = tl.program_id(1)
    offs_M = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_B = pid_BH // H
    offs_H = pid_BH % H
    offs_N = tl.arange(0, HEAD_DIM_QK)

    # load
    mask_M = offs_M < M_CTX
    o = tl.load( # (M, D)
        O + offs_B * stride_o_B + offs_H * stride_o_H
        + offs_M[:, None] * stride_o_M + offs_N[None, :] * stride_o_D,
        mask=mask_M[:, None], other=0.0,
    )

    do = tl.load( # (M, D)
        DO + offs_B * stride_do_B + offs_H * stride_do_H
        + offs_M[:, None] * stride_do_M + offs_N[None, :] * stride_do_D,
        mask=mask_M[:, None], other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1) # (M, )

    # write-back
    tl.store(
        Delta + offs_B * stride_delta_B + offs_H * stride_delta_H
        + offs_M * stride_delta_M, delta, mask=mask_M
    )


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv_inner(
    dk, dv, DBias,
    Q, k, v, sm_scale, Bias,
    DO,
    M, Delta,
    stride_q_M, stride_q_D,
    stride_do_M, stride_do_D,
    stride_dbias_M, stride_dbias_N,
    stride_bias_M, stride_bias_N,
    stride_m_M, stride_delta_M,
    H, M_CTX, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    # Filled in by the wrapper.
    start_N, start_M, num_steps,
    mask_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BIAS: tl.constexpr,
    IS_BIAS_BOOL: tl.constexpr,
    IS_BIAS_TRAINABLE: tl.constexpr,
):
    offs_M = start_M + tl.arange(0, BLOCK_M) # (M,)
    offs_N = start_N + tl.arange(0, BLOCK_N) # (N,)
    offs_qk_D = tl.arange(0, HEAD_DIM_QK) # (D,)
    QT = Q + offs_M[None, :] * stride_q_M + offs_qk_D[:, None] * stride_q_D # (D, M)
    DO = DO + offs_M[:, None] * stride_do_M + offs_qk_D[None, :] * stride_do_D # (M, D)
    if IS_BIAS:
        Bias = Bias + offs_M[None, :] * stride_bias_M + offs_N[:, None] * stride_bias_N # (N, M)
        if IS_BIAS_TRAINABLE:
            DBias = DBias + offs_M[None, :] * stride_dbias_M + offs_N[:, None] * stride_dbias_N # (N, M)
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_M = start_M
    for blk_idx in range(num_steps):
        # k, v : N, D
        offs_M = curr_M + tl.arange(0, BLOCK_M) # (M,)
        mask_M = offs_M < M_CTX # (M,)

        qT = tl.load(QT, mask=mask_M[None, :], other=0.0) # (D, M)
        qkT = tl.dot(k, qT) # (N, M)
        m = tl.load(M + offs_M * stride_m_M, mask=mask_M, other=0.0) # (M,)

        mask_NM = mask_N[:, None] & mask_M[None, :] # (N, M)
        bias = tl.where(mask_NM, 0.0, -1.0e6) # (N, M)
        if IS_CAUSAL:
            # Autoregressive masking.
            causal_mask = offs_M[None, :] >= offs_N[:, None] # (N, M)
            bias += tl.where(causal_mask, 0.0, -1.0e6)
        if IS_BIAS:
            if IS_BIAS_BOOL: # attention mask
                attn_mask = tl.load(Bias, mask=mask_NM, other=False) # (N, M)
                bias += tl.where(attn_mask, 0.0, -1.0e6)
            else: # attention bias
                bias += tl.load(Bias, mask=mask_NM, other=-1.0e6).to(tl.float32)
        qkT += bias # (N, M)
        pT = tl.math.exp2(qkT - m[None, :]) # (N, M)

        do = tl.load(DO, mask_M[:, None], other=0.0) # (M, D)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(v.dtype)
        dv = tl.dot(ppT, do, dv) # (N, D)

        # D (= delta) is pre-divided by ds_scale.
        delta = tl.load(Delta + offs_M * stride_delta_M, mask=mask_M, other=0.0) # (M,)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32) # (N, M)
        dsT = pT * (dpT - delta[None, :]) # (N, M)
        if IS_BIAS_TRAINABLE:
            tl.store(DBias, dsT, mask=mask_NM)
        dsT = dsT.to(v.dtype)
        dk = tl.dot(dsT, tl.trans(qT), dk) # (N, D)
        # Increment pointers.
        curr_M += BLOCK_M
        QT += BLOCK_M * stride_q_M
        DO += BLOCK_M * stride_do_M
        if IS_BIAS:
            Bias += BLOCK_M * stride_bias_M
            if IS_BIAS_TRAINABLE:
                DBias += BLOCK_M * stride_dbias_M
    return dk, dv


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
    ],
    key=["M_CTX", "N_CTX", "HEAD_DIM_QK", "HEAD_DIM_V"],
)
@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, sm_scale,
    DO, DK, DV, DBias,
    M, Delta, Bias,
    stride_q_B, stride_q_M, stride_q_H, stride_q_D,
    stride_k_B, stride_k_N, stride_k_H, stride_k_D,
    stride_v_B, stride_v_N, stride_v_H, stride_v_D,
    stride_dk_B, stride_dk_N, stride_dk_H, stride_dk_D,
    stride_dv_B, stride_dv_N, stride_dv_H, stride_dv_D,
    stride_do_B, stride_do_M, stride_do_H, stride_do_D,
    stride_dbias_B, stride_dbias_H, stride_dbias_M, stride_dbias_N,
    stride_m_B, stride_m_M, stride_m_H,
    stride_delta_B, stride_delta_M, stride_delta_H,
    stride_bias_B, stride_bias_H, stride_bias_M, stride_bias_N,
    H, M_CTX, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BIAS: tl.constexpr,
    IS_BIAS_BOOL: tl.constexpr,
    IS_BIAS_TRAINABLE: tl.constexpr,
):
    pid_N = tl.program_id(0)
    pid_BH = tl.program_id(1)
    offs_B, offs_H = (pid_BH // H).to(tl.int64), (pid_BH % H).to(tl.int64)

    # offset pointers for batch/head
    Q += (offs_B * stride_q_B + offs_H * stride_q_H)
    K += (offs_B * stride_k_B + offs_H * stride_k_H)
    V += (offs_B * stride_v_B + offs_H * stride_v_H)
    DO += (offs_B * stride_do_B + offs_H * stride_do_H)
    DK += (offs_B * stride_dk_B + offs_H * stride_dk_H)
    DV += (offs_B * stride_dv_B + offs_H * stride_dv_H)
    M += (offs_B * stride_m_B + offs_H * stride_m_H)
    Delta += (offs_B * stride_delta_B + offs_H * stride_delta_H)
    if IS_BIAS:
        Bias += (offs_B * stride_bias_B + offs_H * stride_bias_H)
        if IS_BIAS_TRAINABLE:
            DBias += (offs_B * stride_dbias_B + offs_H * stride_dbias_H)

    # load scales
    offs_qk_D = tl.arange(0, HEAD_DIM_QK)
    offs_v_D = tl.arange(0, HEAD_DIM_V)

    start_N = pid_N * BLOCK_N
    offs_N = start_N + tl.arange(0, BLOCK_N)
    mask_N = offs_N < N_CTX

    dk = tl.zeros([BLOCK_N, HEAD_DIM_QK], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM_V], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load( # (N, Dqk)
        K + offs_N[:, None] * stride_k_N + offs_qk_D[None, :] * stride_k_D,
        mask=mask_N[:, None], other=0.0,
    )
    v = tl.load( # (N, Dv)
        V + offs_N[:, None] * stride_v_N + offs_v_D[None, :] * stride_v_D,
        mask=mask_N[:, None], other=0.0,
    )

    if IS_CAUSAL:
        start_M = start_N

        MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
        num_steps = BLOCK_N // MASK_BLOCK_M

        dk, dv = _attn_bwd_dkdv_inner(
            dk, dv, DBias,
            Q, k, v, sm_scale, Bias,
            DO,
            M, Delta,
            stride_q_M, stride_q_D,
            stride_do_M, stride_do_D,
            stride_dbias_M, stride_dbias_N,
            stride_bias_M, stride_bias_N,
            stride_m_M, stride_delta_M,
            H, M_CTX,
            MASK_BLOCK_M, BLOCK_N, HEAD_DIM_QK, HEAD_DIM_V,
            start_N, start_M, num_steps,
            mask_N=mask_N,
            IS_CAUSAL=True,
            IS_BIAS=IS_BIAS,
            IS_BIAS_BOOL=IS_BIAS_BOOL,
            IS_BIAS_TRAINABLE=IS_BIAS_TRAINABLE,
        )

        start_M += num_steps * MASK_BLOCK_M
        num_steps = (M_CTX - start_M + BLOCK_M - 1) // BLOCK_M
    else:
        start_M = 0
        num_steps = (M_CTX + BLOCK_M - 1) // BLOCK_M

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv_inner(
        dk, dv, DBias,
        Q, k, v, sm_scale, Bias,
        DO,
        M, Delta,
        stride_q_M, stride_q_D,
        stride_do_M, stride_do_D,
        stride_dbias_M, stride_dbias_N,
        stride_bias_M, stride_bias_N,
        stride_m_M, stride_delta_M,
        H, M_CTX,
        BLOCK_M, BLOCK_N, HEAD_DIM_QK, HEAD_DIM_V,
        start_N, start_M, num_steps,
        mask_N=mask_N,
        IS_CAUSAL=False,
        IS_BIAS=IS_BIAS,
        IS_BIAS_BOOL=IS_BIAS_BOOL,
        IS_BIAS_TRAINABLE=IS_BIAS_TRAINABLE,
    )

    # Write back dK and dV
    dk *= sm_scale
    DK += offs_N[:, None] * stride_dk_N + offs_qk_D[None, :] * stride_dk_D
    tl.store(DK, dk, mask=mask_N[:, None])

    DV += offs_N[:, None] * stride_dv_N + offs_v_D[None, :] * stride_dv_D
    tl.store(DV, dv, mask=mask_N[:, None])


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq_inner(
    dq, q, K, V, Bias,
    do, m, Delta,
    stride_k_N, stride_v_N, stride_k_D, stride_v_D,
    stride_bias_M, stride_bias_N,
    stride_delta_M,
    H, M_CTX, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    # Filled in by the wrapper.
    start_M, start_N, num_steps,
    mask_M: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BIAS: tl.constexpr,
    IS_BIAS_BOOL: tl.constexpr,
):
    offs_M = start_M + tl.arange(0, BLOCK_M) # (M,)
    offs_N = start_N + tl.arange(0, BLOCK_N) # (N,)
    offs_qk_D = tl.arange(0, HEAD_DIM_QK) # (Dqk,)
    offs_v_D = tl.arange(0, HEAD_DIM_V) # (Dv,)
    KT = K + offs_N[None, :] * stride_k_N + offs_qk_D[:, None] * stride_k_D # (Dqk, N)
    VT = V + offs_N[None, :] * stride_v_N + offs_v_D[:, None] * stride_v_D # (Dv, N)
    if IS_BIAS:
        Bias = Bias + offs_M[:, None] * stride_bias_M + offs_N[None, :] * stride_bias_N # (M, N)
    # D (= delta) is pre-divided by ds_scale.
    delta = tl.load(Delta + offs_M * stride_delta_M, mask=offs_M < M_CTX, other=0.0) # (M,)
    # BLOCK_M must be a multiple of BLOCK_N, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    curr_N = start_N
    for blk_idx in range(num_steps):
        # q: M, D / m: M, 1 / do: M, D
        offs_N = curr_N + tl.arange(0, BLOCK_N)
        mask_N = offs_N < N_CTX # (1, N)

        kT = tl.load(KT, mask=mask_N[None, :], other=0.0) # (D, N)
        vT = tl.load(VT, mask=mask_N[None, :], other=0.0) # (D, N)
        qk = tl.dot(q, kT) # (M, N)

        mask_MN = mask_M[:, None] & mask_N[None, :]
        bias = tl.where(mask_MN, 0.0, -1.0e6)
        if IS_CAUSAL:
            # Autoregressive masking.
            causal_mask = offs_M[:, None] >= offs_N[None, :]
            bias += tl.where(causal_mask, 0.0, -1.0e6)
        if IS_BIAS:
            if IS_BIAS_BOOL: # attention mask
                attn_mask = tl.load(Bias, mask=mask_MN, other=False)
                bias += tl.where(attn_mask, 0.0, -1.0e6)
            else: # attention bias
                bias += tl.load(Bias, mask=mask_MN, other=-1.0e6).to(tl.float32)
        qk += bias # (M, N)
        p = tl.math.exp2(qk - m) # (M, N)

        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32) # (M, N)
        ds = p * (dp - delta[:, None]) # (M, N)
        ds = ds.to(q.dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT)) # (M, D)
        # Increment pointers.
        curr_N += BLOCK_N
        KT += BLOCK_N * stride_k_N
        VT += BLOCK_N * stride_v_N
        if IS_BIAS:
            Bias += BLOCK_N * stride_bias_N
    return dq


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
    ],
    key=["M_CTX", "N_CTX", "HEAD_DIM_QK", "HEAD_DIM_V"],
)
@triton.jit
def _attn_bwd_dq(
    Q, K, V, sm_scale, Bias,
    DO, DQ, M, Delta,
    # shared by Q/K/V/DO.
    stride_q_B, stride_q_M, stride_q_H, stride_q_D,
    stride_k_B, stride_k_N, stride_k_H, stride_k_D,
    stride_v_B, stride_v_N, stride_v_H, stride_v_D,
    stride_dq_B, stride_dq_M, stride_dq_H, stride_dq_D,
    stride_do_B, stride_do_M, stride_do_H, stride_do_D,
    stride_m_B, stride_m_M, stride_m_H,
    stride_delta_B, stride_delta_M, stride_delta_H,
    stride_bias_B, stride_bias_H, stride_bias_M, stride_bias_N,
    H, M_CTX, N_CTX,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BIAS: tl.constexpr,
    IS_BIAS_BOOL: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996 # = ln(2)

    pid_M = tl.program_id(0)
    pid_BH = tl.program_id(1)
    offs_B, offs_H = (pid_BH // H).to(tl.int64), (pid_BH % H).to(tl.int64)

    # offset pointers for batch/head
    Q += (offs_B * stride_q_B + offs_H * stride_q_H)
    K += (offs_B * stride_k_B + offs_H * stride_k_H)
    V += (offs_B * stride_v_B + offs_H * stride_v_H)
    DO += (offs_B * stride_do_B + offs_H * stride_do_H)
    DQ += (offs_B * stride_dq_B + offs_H * stride_dq_H)
    M += (offs_B * stride_m_B + offs_H * stride_m_H)
    Delta += (offs_B * stride_delta_B + offs_H * stride_delta_H)
    if IS_BIAS:
        Bias += (offs_B * stride_bias_B + offs_H * stride_bias_H)

    # load scales
    offs_qk_D = tl.arange(0, HEAD_DIM_QK)

    # THIS BLOCK DOES DQ:
    start_M = pid_M * BLOCK_M
    offs_M = start_M + tl.arange(0, BLOCK_M)

    # load Q, DO, M
    mask_M = offs_M < M_CTX
    q = tl.load( # (M, D)
        Q + offs_M[:, None] * stride_q_M + offs_qk_D[None, :] * stride_q_D,
        mask=mask_M[:, None], other=0.0,
    )

    do = tl.load( # (M, D)
        DO + offs_M[:, None] * stride_do_M + offs_qk_D[None, :] * stride_do_D,
        mask=mask_M[:, None], other=0.0,
    )

    m = tl.load(M + offs_M * stride_m_M, mask=mask_M, other=0.0)
    m = m[:, None]

    # Initialize DQ
    dq = tl.zeros([BLOCK_M, HEAD_DIM_QK], dtype=tl.float32)

    if IS_CAUSAL:
        # Compute dQ for diagonal blocks.
        end_N = start_M + BLOCK_M
        MASK_BLOCK_N: tl.constexpr = BLOCK_N // BLK_SLICE_FACTOR
        num_steps = BLOCK_M // MASK_BLOCK_N
        dq = _attn_bwd_dq_inner(
            dq, q, K, V, Bias,
            do, m, Delta,
            stride_k_N, stride_v_N, stride_k_D, stride_v_D,
            stride_bias_M, stride_bias_N,
            stride_delta_M,
            H, M_CTX, N_CTX,
            BLOCK_M, MASK_BLOCK_N, HEAD_DIM_QK, HEAD_DIM_V,
            start_M, end_N - num_steps * MASK_BLOCK_N, num_steps,
            mask_M,
            IS_CAUSAL=True,
            IS_BIAS=IS_BIAS,
            IS_BIAS_BOOL=IS_BIAS_BOOL,
        )
        end_N -= num_steps * MASK_BLOCK_N
        num_steps = end_N // BLOCK_N
    else:
        num_steps = (N_CTX + BLOCK_N - 1) // BLOCK_N

    # Compute dQ for non-diagonal blocks.
    dq = _attn_bwd_dq_inner(
        dq, q, K, V, Bias,
        do, m, Delta,
        stride_k_N, stride_v_N, stride_k_D, stride_v_D,
        stride_bias_M, stride_bias_N,
        stride_delta_M,
        H, M_CTX, N_CTX,
        BLOCK_M, BLOCK_N, HEAD_DIM_QK, HEAD_DIM_V,
        start_M, 0, num_steps,
        mask_M,
        IS_CAUSAL=False,
        IS_BIAS=IS_BIAS,
        IS_BIAS_BOOL=IS_BIAS_BOOL,
    )

    # Write back dQ.
    dq *= LN2
    DQ += offs_M[:, None] * stride_dq_M + offs_qk_D[None, :] * stride_dq_D
    tl.store(DQ, dq, mask=mask_M[:, None])


def _flash_attn_fwd(q, k, v, bias, causal, sm_scale):
    BATCH, M_CTX, N_HEAD, HEAD_DIM_QK = q.shape
    _, N_CTX, _, HEAD_DIM_V = v.shape

    o = q.new_empty(BATCH, M_CTX, N_HEAD, HEAD_DIM_V)
    extra_kern_args = {}

    is_bias = bias is not None
    is_bias_bool = True if (is_bias and (bias.dtype == torch.bool)) else False

    arg_bias = bias
    if is_bias and not is_bias_bool:
        RCP_LN2 = 1.4426950408889634 # = 1.0 / ln(2)
        arg_bias = bias
        arg_bias = arg_bias * RCP_LN2
    bias_stride = [arg_bias.stride(i) for i in range(4)] if is_bias else [None] * 4

    grid = lambda args: (triton.cdiv(M_CTX, args["BLOCK_M"]), BATCH * N_HEAD) # (M // BLOCK_M, BH)
    M = torch.empty((BATCH, M_CTX, N_HEAD), device=q.device, dtype=torch.float32)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, o, arg_bias,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        bias_stride[0], bias_stride[1], bias_stride[2], bias_stride[3],
        BATCH, N_HEAD,
        M_CTX=M_CTX,
        N_CTX=N_CTX,
        HEAD_DIM_QK=HEAD_DIM_QK,
        HEAD_DIM_V=HEAD_DIM_V,
        IS_CAUSAL=causal,
        IS_BIAS=is_bias,
        IS_BIAS_BOOL=is_bias_bool,
        **extra_kern_args)
    return o, M


def _flash_attn_bwd(do, q, k, v, o, bias, causal, sm_scale, M, dq, dk, dv, dbias):
    delta = torch.empty_like(M) # B, M, H

    is_bias = bias is not None
    is_bias_bool = True if (is_bias and (bias.dtype == torch.bool)) else False
    is_bias_trainable = True if (is_bias and bias.requires_grad) else False

    BATCH, M_CTX, N_HEAD, HEAD_DIM_QK = q.shape
    _, N_CTX, _, HEAD_DIM_V = v.shape
    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.4426950408889634 # = 1.0 / ln(2)
    arg_k = k
    arg_k = arg_k * (sm_scale * RCP_LN2)
    arg_bias = bias
    if is_bias and not is_bias_bool:
        arg_bias = bias
        arg_bias = arg_bias * RCP_LN2
    bias_stride = [arg_bias.stride(i) for i in range(4)] if is_bias else [None] * 4
    dbias_stride = [dbias.stride(i) for i in range(4)] if is_bias_trainable else [None] * 4

    PRE_BLOCK = 128
    pre_grid = (triton.cdiv(M_CTX, PRE_BLOCK), BATCH * N_HEAD)
    _attn_bwd_preprocess[pre_grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        BATCH, N_HEAD, M_CTX,
        BLOCK_M=PRE_BLOCK, HEAD_DIM_QK=HEAD_DIM_QK
    )

    grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_N"]), BATCH * N_HEAD)
    _attn_bwd_dkdv[grid](
        q, arg_k, v, sm_scale,
        do, dk, dv, dbias,
        M, delta, arg_bias,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        arg_k.stride(0), arg_k.stride(1), arg_k.stride(2), arg_k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dbias_stride[0], dbias_stride[1], dbias_stride[2], dbias_stride[3],
        M.stride(0), M.stride(1), M.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        bias_stride[0], bias_stride[1], bias_stride[2], bias_stride[3],
        N_HEAD, M_CTX, N_CTX,
        HEAD_DIM_QK=HEAD_DIM_QK,
        HEAD_DIM_V=HEAD_DIM_V,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        IS_CAUSAL=causal,
        IS_BIAS=is_bias,
        IS_BIAS_BOOL=is_bias_bool,
        IS_BIAS_TRAINABLE=is_bias_trainable,
    )

    grid = lambda args: (triton.cdiv(M_CTX, args["BLOCK_M"]), BATCH * N_HEAD)
    _attn_bwd_dq[grid](
        q, arg_k, v, sm_scale, arg_bias,
        do, dq, M, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        arg_k.stride(0), arg_k.stride(1), arg_k.stride(2), arg_k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        bias_stride[0], bias_stride[1], bias_stride[2], bias_stride[3],
        N_HEAD, M_CTX, N_CTX,
        HEAD_DIM_QK=HEAD_DIM_QK,
        HEAD_DIM_V=HEAD_DIM_V,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        IS_CAUSAL=causal,
        IS_BIAS=is_bias,
        IS_BIAS_BOOL=is_bias_bool,
    )


class FlashAttentionFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, sm_scale=None):
        # shape constraints
        BATCH, M_CTX, N_HEAD, HEAD_DIM_QK = q.shape
        sm_scale = (HEAD_DIM_QK ** -0.5) if sm_scale is None else sm_scale
        # when v is in float8_e5m2 it is transposed.
        _, N_CTX, _, HEAD_DIM_V = v.shape
        assert HEAD_DIM_QK == k.shape[-1]
        assert HEAD_DIM_QK in {16, 32, 64, 128, 256}
        assert (causal and (M_CTX == N_CTX)) or (~causal), """
            If causal, length of Q and KV should be the same.
        """
        is_bias = bias is not None
        is_bias_trainable = True if (is_bias and bias.requires_grad) else False

        o, M = _flash_attn_fwd(q, k, v, bias, causal, sm_scale)

        ctx.save_for_backward(q, k, v, o, M, bias)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.is_bias = is_bias
        ctx.is_bias_trainable = is_bias_trainable
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, bias = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = torch.zeros_like(bias) if ctx.is_bias_trainable else None

        _flash_attn_bwd(do, q, k, v, o, bias, ctx.causal, ctx.sm_scale, M, dq, dk, dv, dbias)

        return dq, dk, dv, dbias, None, None


class FlashAttentionQKVPackedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, bias=None, causal=False, sm_scale=None):
        # shape constraints
        BATCH, M_CTX, _, N_HEAD, HEAD_DIM = qkv.shape
        sm_scale = (HEAD_DIM ** -0.5) if sm_scale is None else sm_scale
        # when v is in float8_e5m2 it is transposed.
        assert HEAD_DIM in {16, 32, 64, 128, 256}
        is_bias = bias is not None
        is_bias_trainable = True if (is_bias and bias.requires_grad) else False

        o, M = _flash_attn_fwd(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            bias, causal, sm_scale
        )

        ctx.save_for_backward(qkv, o, M, bias)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.is_bias = is_bias
        ctx.is_bias_trainable = is_bias_trainable
        return o

    @staticmethod
    def backward(ctx, do):
        qkv, o, M, bias = ctx.saved_tensors

        dqkv = torch.empty_like(qkv)
        dbias = torch.zeros_like(bias) if ctx.is_bias_trainable else None

        _flash_attn_bwd(
            do, qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2],
            o, bias, ctx.causal, ctx.sm_scale, M,
            dqkv[:, :, 0], dqkv[:, :, 1], dqkv[:, :, 2], dbias
        )

        return dqkv, dbias, None, None


def flash_attention_fn(q, k, v, bias=None, causal=False, sm_scale=None):
    return FlashAttentionFn.apply(q, k, v, bias, causal, sm_scale)


def flash_attention_qkvpacked_fn(qkv, bias=None, causal=False, sm_scale=None):
    return FlashAttentionQKVPackedFn.apply(qkv, bias, causal, sm_scale)
