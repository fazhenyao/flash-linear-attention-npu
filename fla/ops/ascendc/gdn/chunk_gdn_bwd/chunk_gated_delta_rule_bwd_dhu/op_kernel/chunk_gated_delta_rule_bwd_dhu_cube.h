#ifndef FLA_CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H
#define FLA_CHUNK_GATED_DELTA_RULE_BWD_DHU_CUBE_H

#include <type_traits>
#define CATLASS_ARCH 2201
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "chunk_gated_delta_rule_bwd_dhu_common.h"

namespace GDN {
template <typename T, int V_DIM, int CHUNK_SIZE, int STRATEGY>
class DhuCube {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR dO, GM_ADDR cu, GM_ADDR workspace,
                                const ChunkGatedDeltaRuleBwdDhuTilingData *p)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        k_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(w));
        do_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dO));
        ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        wsT_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
        cu_ = cu;
        p_ = p;
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint64_t taskCount = (STRATEGY == CHUNK_DHU_STRATEGY_VARLEN ? p_->seqs : p_->b) * p_->hv;
        for (uint64_t task = core; task < taskCount; task += p_->used_core) {
            const uint64_t seq = task / p_->hv;
            const uint64_t hv = task % p_->hv;
            const uint64_t hq = hv / (p_->hv / p_->hk);
            const uint64_t nChunks = SequenceChunks(seq);
            for (int64_t chunk = static_cast<int64_t>(nChunks) - 1; chunk >= 0; --chunk) {
                const auto info = DhuGetChunkInfo<STRATEGY>(seq, static_cast<uint64_t>(chunk), cu_, *p_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(vecToCube_);
                GemmKState(info, hq, core);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(cubeToVec_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(vecToCube_);
                GemmUpdate(info, hq, hv, core);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(cubeToVec_);
            }
        }
    }

private:
    using Arch = Catlass::Arch::AtlasA2;
    using Dispatch = Catlass::Gemm::MmadPingpong<Arch, true, false>;
    using L1TileShape = typename std::conditional<
        V_DIM == 128,
        tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<128>>,
        tla::Shape<tla::Int<128>, tla::Int<256>, tla::Int<64>>>::type;
    using L0TileShape = L1TileShape;
    using LayoutR = Catlass::layout::RowMajor;
    using LayoutC = Catlass::layout::ColumnMajor;

    __aicore__ inline uint64_t SequenceChunks(uint64_t seq) const
    {
        if constexpr (STRATEGY == CHUNK_DHU_STRATEGY_FIXED) return p_->chunks;
        AscendC::GlobalTensor<int64_t> cuGm;
        cuGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_));
        const uint64_t len = static_cast<uint64_t>(cuGm.GetValue(seq + 1) - cuGm.GetValue(seq));
        return (len + p_->chunk - 1) / p_->chunk;
    }

    template <typename AType, typename BType, typename CType, typename ALayout, typename BLayout>
    __aicore__ inline void Gemm(AscendC::GlobalTensor<AType> &a, uint64_t aOff, uint32_t rowsA, uint32_t colsA,
                                AscendC::GlobalTensor<BType> &b, uint64_t bOff, uint32_t rowsB, uint32_t colsB,
                                AscendC::GlobalTensor<CType> &c, uint64_t cOff, uint32_t rowsC, uint32_t colsC,
                                uint32_t m, uint32_t n, uint32_t k)
    {
        using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<Arch, AType, ALayout, BType, BLayout, CType, LayoutR>;
        using Block = Catlass::Gemm::Block::BlockMmadTla<Dispatch, L1TileShape, L0TileShape,
                                                         AType, BType, CType, void, Copy>;
        Catlass::Arch::Resource<Arch> resource;
        auto ta = tla::MakeTensor(a[aOff], tla::MakeLayout<AType, ALayout>(rowsA, colsA), Catlass::Arch::PositionGM{});
        auto tb = tla::MakeTensor(b[bOff], tla::MakeLayout<BType, BLayout>(rowsB, colsB), Catlass::Arch::PositionGM{});
        auto tc = tla::MakeTensor(c[cOff], tla::MakeLayout<CType, LayoutR>(rowsC, colsC), Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto ba = GetTile(ta, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto bb = GetTile(tb, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto bc = GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        Block block(resource);
        block(ba, bb, bc, shape);
    }

    __aicore__ inline void GemmKState(const DhuChunkInfo &info, uint64_t hq, uint32_t core)
    {
        const uint64_t kOff = ((STRATEGY == CHUNK_DHU_STRATEGY_FIXED ? info.seq : 0) * p_->hk + hq) * p_->t * p_->k +
                              (info.start % p_->t) * p_->k;
        Gemm<T, T, float, LayoutR, LayoutR>(k_, kOff, CHUNK_SIZE, 128,
                                            wsT_, DhuMatrixTypedOffset<T>(core, 0, *p_), 128, V_DIM,
                                            ws_, DhuMatrixFloatOffset(core, 1, *p_), CHUNK_SIZE, V_DIM,
                                            info.len, V_DIM, 128);
    }

    __aicore__ inline void GemmUpdate(const DhuChunkInfo &info, uint64_t hq, uint64_t hv, uint32_t core)
    {
        const uint64_t wOff = ((STRATEGY == CHUNK_DHU_STRATEGY_FIXED ? info.seq : 0) * p_->hv + hv) * p_->t * p_->k +
                              (info.start % p_->t) * p_->k;
        const uint64_t doOff = ((STRATEGY == CHUNK_DHU_STRATEGY_FIXED ? info.seq : 0) * p_->hv + hv) * p_->t * p_->v +
                               (info.start % p_->t) * p_->v;
        Gemm<T, T, float, LayoutC, LayoutR>(wsT_, DhuMatrixTypedOffset<T>(core, 3, *p_), 128, CHUNK_SIZE,
                                            do_, doOff, CHUNK_SIZE, V_DIM,
                                            ws_, DhuMatrixFloatOffset(core, 4, *p_), 128, V_DIM,
                                            128, V_DIM, info.len);
        Gemm<T, T, float, LayoutC, LayoutR>(w_, wOff, 128, CHUNK_SIZE,
                                            wsT_, DhuMatrixTypedOffset<T>(core, 2, *p_), CHUNK_SIZE, V_DIM,
                                            ws_, DhuMatrixFloatOffset(core, 5, *p_), 128, V_DIM,
                                            128, V_DIM, info.len);
    }

    AscendC::GlobalTensor<T> q_, k_, w_, do_;
    AscendC::GlobalTensor<T> wsT_;
    AscendC::GlobalTensor<float> ws_;
    GM_ADDR cu_{nullptr};
    const ChunkGatedDeltaRuleBwdDhuTilingData *p_{nullptr};
    Catlass::Arch::CrossCoreFlagWithReverse<> vecToCube_{DHU_VEC_TO_CUBE_READY, DHU_CUBE_TO_VEC_FREE};
    Catlass::Arch::CrossCoreFlagWithReverse<> cubeToVec_{DHU_CUBE_TO_VEC_READY, DHU_VEC_TO_CUBE_FREE};
};
} // namespace GDN
#endif
