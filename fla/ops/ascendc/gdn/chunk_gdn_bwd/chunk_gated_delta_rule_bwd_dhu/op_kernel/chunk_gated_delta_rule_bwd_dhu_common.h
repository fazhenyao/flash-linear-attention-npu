#ifndef FLA_CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H
#define FLA_CHUNK_GATED_DELTA_RULE_BWD_DHU_COMMON_H

#include "catlass/arch/cross_core_sync.hpp"
#include "kernel_operator.h"

namespace GDN {
constexpr uint64_t DHU_VEC_TO_CUBE_READY = 1;
constexpr uint64_t DHU_CUBE_TO_VEC_FREE = 2;
constexpr uint64_t DHU_CUBE_TO_VEC_READY = 3;
constexpr uint64_t DHU_VEC_TO_CUBE_FREE = 4;
constexpr uint32_t DHU_ALIGN_BYTES = 512;
constexpr uint32_t DHU_FP32_BLOCK = 8;

struct DhuChunkInfo {
    uint64_t seq;
    uint64_t globalChunk;
    uint64_t start;
    uint64_t len;
};

__aicore__ inline uint64_t DhuAlignUp(uint64_t value)
{
    return (value + DHU_ALIGN_BYTES - 1) / DHU_ALIGN_BYTES * DHU_ALIGN_BYTES;
}

__aicore__ inline uint64_t DhuSlotByteOffset(uint32_t core, const ChunkGatedDeltaRuleBwdDhuTilingData &p)
{
    return static_cast<uint64_t>(core) * p.slot_bytes;
}

__aicore__ inline uint64_t DhuStateOffset(uint32_t core, const ChunkGatedDeltaRuleBwdDhuTilingData &p)
{
    return DhuSlotByteOffset(core, p) / sizeof(float);
}

__aicore__ inline uint64_t DhuMatrixByteOffset(uint32_t core, uint32_t matrix,
                                               const ChunkGatedDeltaRuleBwdDhuTilingData &p)
{
    return DhuSlotByteOffset(core, p) + p.state_bytes + static_cast<uint64_t>(matrix) * p.matrix_bytes;
}

__aicore__ inline uint64_t DhuMatrixFloatOffset(uint32_t core, uint32_t matrix,
                                                const ChunkGatedDeltaRuleBwdDhuTilingData &p)
{
    return DhuMatrixByteOffset(core, matrix, p) / sizeof(float);
}

template <typename T>
__aicore__ inline uint64_t DhuMatrixTypedOffset(uint32_t core, uint32_t matrix,
                                                const ChunkGatedDeltaRuleBwdDhuTilingData &p)
{
    return DhuMatrixByteOffset(core, matrix, p) / sizeof(T);
}

template <int STRATEGY>
__aicore__ inline DhuChunkInfo DhuGetChunkInfo(uint64_t seq, uint64_t localChunk, GM_ADDR cu,
                                               const ChunkGatedDeltaRuleBwdDhuTilingData &p)
{
    DhuChunkInfo info{seq, localChunk, 0, p.chunk};
    if constexpr (STRATEGY == CHUNK_DHU_STRATEGY_VARLEN) {
        AscendC::GlobalTensor<int64_t> cuGm;
        cuGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu));
        const uint64_t bos = static_cast<uint64_t>(cuGm.GetValue(seq));
        const uint64_t eos = static_cast<uint64_t>(cuGm.GetValue(seq + 1));
        info.start = bos + localChunk * p.chunk;
        info.len = info.start + p.chunk <= eos ? p.chunk : eos - info.start;
        uint64_t offset = 0;
        for (uint64_t i = 0; i < seq; ++i) {
            const uint64_t begin = static_cast<uint64_t>(cuGm.GetValue(i));
            const uint64_t end = static_cast<uint64_t>(cuGm.GetValue(i + 1));
            offset += (end - begin + p.chunk - 1) / p.chunk;
        }
        info.globalChunk = offset + localChunk;
    } else {
        info.start = seq * p.t + localChunk * p.chunk;
        info.len = (localChunk + 1) * p.chunk <= p.t ? p.chunk : p.t - localChunk * p.chunk;
        info.globalChunk = seq * p.chunks + localChunk;
    }
    return info;
}

} // namespace GDN
#endif
