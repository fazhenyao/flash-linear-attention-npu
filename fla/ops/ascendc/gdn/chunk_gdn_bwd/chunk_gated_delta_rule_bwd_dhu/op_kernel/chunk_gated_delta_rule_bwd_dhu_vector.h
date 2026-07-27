#ifndef FLA_CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H
#define FLA_CHUNK_GATED_DELTA_RULE_BWD_DHU_VECTOR_H

#include <type_traits>
#include "chunk_gated_delta_rule_bwd_dhu_common.h"

namespace GDN {
template <typename T, typename GT, int V_DIM, int CHUNK_SIZE, int STRATEGY>
class DhuVector {
public:
    static constexpr uint32_t ROW_TILE = 16;
    static constexpr uint32_t ROW_FLOAT_ELEMS = ROW_TILE * 256;
    static constexpr uint32_t STATE_TILE_ELEMS = 4096;
    static constexpr uint32_t STATE_ELEMS = 128 * V_DIM;
    static constexpr bool STATE_RESIDENT = V_DIM == 128;

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR dht, GM_ADDR cu,
                                GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace,
                                const ChunkGatedDeltaRuleBwdDhuTilingData *p, AscendC::TPipe *pipe)
    {
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        dv_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dv));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(g));
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ GT *>(gk));
        dht_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dht));
        dh_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dh));
        dh0_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dh0));
        dv2_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dv2));
        ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        wsT_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
        cu_ = cu;
        p_ = p;
        pipe_ = pipe;
        pipe_->InitBuffer(stateBuf_,
            (STATE_RESIDENT ? STATE_ELEMS + 2 * STATE_TILE_ELEMS : 3 * STATE_TILE_ELEMS) * sizeof(float));
        pipe_->InitBuffer(matrixIn_, 2, ROW_FLOAT_ELEMS * sizeof(float));
        pipe_->InitBuffer(matrixOut_, 1, ROW_TILE * 256 * sizeof(T));
        pipe_->InitBuffer(calcBuf_, 2 * ROW_FLOAT_ELEMS * sizeof(float) + 3 * 128 * sizeof(float));
        pipe_->InitBuffer(gateIn_, 128 * sizeof(GT));
        mte2VEvent_ = pipe_->FetchEventID(AscendC::HardEvent::MTE2_V);
        vMte3Event_ = pipe_->FetchEventID(AscendC::HardEvent::V_MTE3);
        mte3Mte2Event_ = pipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        vMte2Event_ = pipe_->FetchEventID(AscendC::HardEvent::V_MTE2);
        mte2Mte3Event_ = pipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3);
        mte3VEvent_ = pipe_->FetchEventID(AscendC::HardEvent::MTE3_V);
        vSEvent_ = pipe_->FetchEventID(AscendC::HardEvent::V_S);
        sVEvent_ = pipe_->FetchEventID(AscendC::HardEvent::S_V);
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetSubBlockIdx() != 0) return;
        const uint32_t core = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const uint64_t taskCount = (STRATEGY == CHUNK_DHU_STRATEGY_VARLEN ? p_->seqs : p_->b) * p_->hv;
        for (uint64_t task = core; task < taskCount; task += p_->used_core) {
            const uint64_t seq = task / p_->hv;
            const uint64_t hv = task % p_->hv;
            const uint64_t hq = hv / (p_->hv / p_->hk);
            InitState(seq, hv, core);
            const uint64_t nChunks = SequenceChunks(seq);
            for (int64_t chunk = static_cast<int64_t>(nChunks) - 1; chunk >= 0; --chunk) {
                const auto info = DhuGetChunkInfo<STRATEGY>(seq, static_cast<uint64_t>(chunk), cu_, *p_);
                StoreDh(info, hv, core);
                PrepareStateOperand(core);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(vecToCube_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(cubeToVec_);
                BuildDv2AndQg(info, hq, hv, core);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(vecToCube_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(cubeToVec_);
                UpdateState(info, hv, core);
            }
            if (p_->has_h0) StoreDh0(seq, hv, core);
        }
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2VEvent_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vMte3Event_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3Mte2Event_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vMte2Event_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(mte2Mte3Event_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3VEvent_);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_S>(vSEvent_);
        pipe_->ReleaseEventID<AscendC::HardEvent::S_V>(sVEvent_);
    }

private:
    __aicore__ inline uint64_t SequenceChunks(uint64_t seq) const
    {
        if constexpr (STRATEGY == CHUNK_DHU_STRATEGY_FIXED) return p_->chunks;
        AscendC::GlobalTensor<int64_t> cuGm;
        cuGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cu_));
        const uint64_t len = static_cast<uint64_t>(cuGm.GetValue(seq + 1) - cuGm.GetValue(seq));
        return (len + p_->chunk - 1) / p_->chunk;
    }

    __aicore__ inline uint64_t TokenBase(uint64_t seq, uint64_t hv, uint64_t start, uint64_t width,
                                         uint64_t heads) const
    {
        const uint64_t b = STRATEGY == CHUNK_DHU_STRATEGY_FIXED ? seq : 0;
        return (b * heads + hv) * p_->t * width + (start % p_->t) * width;
    }

    __aicore__ inline uint64_t GateBase(uint64_t seq, uint64_t hv, uint64_t start) const
    {
        const uint64_t b = STRATEGY == CHUNK_DHU_STRATEGY_FIXED ? seq : 0;
        return (b * p_->hv + hv) * p_->t + (start % p_->t);
    }

    __aicore__ inline void SyncMte2V()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2VEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2VEvent_);
    }

    __aicore__ inline void SyncVMte3()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3Event_);
    }

    __aicore__ inline void SyncMte3Mte2()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3Mte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3Mte2Event_);
    }

    __aicore__ inline void SyncVMte2()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vMte2Event_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vMte2Event_);
    }

    __aicore__ inline void SyncMte2Mte3()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte2Mte3Event_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte2Mte3Event_);
    }

    __aicore__ inline void SyncMte3V()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3VEvent_);
    }

    __aicore__ inline void SyncVS()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vSEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vSEvent_);
    }

    __aicore__ inline void SyncSV()
    {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(sVEvent_);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(sVEvent_);
    }

    __aicore__ inline void ApplyRowScale(AscendC::LocalTensor<float> matrix,
                                         AscendC::LocalTensor<float> rowScale,
                                         AscendC::LocalTensor<float> broadcast,
                                         uint32_t rows, uint32_t cols)
    {
        constexpr uint32_t FP32_PER_BLOCK = 8;
        constexpr uint32_t FP32_PER_REPEAT = 64;
        const uint8_t rowStride = static_cast<uint8_t>(cols / FP32_PER_BLOCK);
        AscendC::BinaryRepeatParams params(1, 1, 0, rowStride, rowStride, 1);
        for (uint32_t row = 0; row < rows; row += FP32_PER_BLOCK) {
            const uint32_t rowsThisBlock = rows - row < FP32_PER_BLOCK ? rows - row : FP32_PER_BLOCK;
            AscendC::Brcb(broadcast, rowScale[row], 1, {1, FP32_PER_BLOCK});
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < cols; col += FP32_PER_REPEAT) {
                const uint32_t count = cols - col < FP32_PER_REPEAT ? cols - col : FP32_PER_REPEAT;
                const uint32_t offset = row * cols + col;
                AscendC::Mul(matrix[offset], matrix[offset], broadcast, count, rowsThisBlock, params);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ApplyGkRowScale(AscendC::LocalTensor<float> matrix,
                                           AscendC::LocalTensor<float> rowScale,
                                           uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            SyncVS();
            const float scale = rowScale.GetValue(row);
            SyncSV();
            AscendC::Muls(matrix[row * cols], matrix[row * cols], scale, cols);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitState(uint64_t seq, uint64_t hv, uint32_t core)
    {
        auto state = stateBuf_.Get<float>();
        const uint64_t elems = p_->k * p_->v;
        const uint64_t src = (seq * p_->hv + hv) * elems;
        const uint64_t dst = DhuStateOffset(core, *p_);
        for (uint64_t off = 0; off < elems; off += STATE_TILE_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(elems - off < STATE_TILE_ELEMS ? elems - off : STATE_TILE_ELEMS);
            if (p_->has_dht) {
                AscendC::DataCopy(state[STATE_RESIDENT ? off : 0], dht_[src + off], count);
                if constexpr (STATE_RESIDENT) {
                    SyncMte2V();
                } else {
                    SyncMte2Mte3();
                }
            } else {
                AscendC::Duplicate(state[STATE_RESIDENT ? off : 0], 0.0f, count);
                if constexpr (!STATE_RESIDENT) SyncVMte3();
            }
            if constexpr (!STATE_RESIDENT) {
                AscendC::DataCopy(ws_[dst + off], state, count);
                SyncMte3Mte2();
            }
        }
    }

    __aicore__ inline void StoreDh(const DhuChunkInfo &info, uint64_t hv, uint32_t core)
    {
        const uint64_t outHead = info.globalChunk * p_->hv + hv;
        const uint64_t elems = p_->k * p_->v;
        const uint64_t src = DhuStateOffset(core, *p_);
        const uint64_t dst = outHead * elems;
        auto state = stateBuf_.Get<float>();
        auto out = calcBuf_.Get<T>();
        for (uint64_t off = 0; off < elems; off += STATE_TILE_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(elems - off < STATE_TILE_ELEMS ? elems - off : STATE_TILE_ELEMS);
            if constexpr (!STATE_RESIDENT) {
                AscendC::DataCopy(state, ws_[src + off], count);
                SyncMte2V();
            }
            AscendC::Cast(out, state[STATE_RESIDENT ? off : 0], AscendC::RoundMode::CAST_RINT, count);
            SyncVMte3();
            AscendC::DataCopy(dh_[dst + off], out, count);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void PrepareStateOperand(uint32_t core)
    {
        auto state = stateBuf_.Get<float>();
        const uint64_t elems = p_->k * p_->v;
        const uint64_t src = DhuStateOffset(core, *p_);
        const uint64_t dst = DhuMatrixTypedOffset<T>(core, 0, *p_);
        for (uint64_t off = 0; off < elems; off += STATE_TILE_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(elems - off < STATE_TILE_ELEMS ? elems - off : STATE_TILE_ELEMS);
            auto out = matrixOut_.AllocTensor<T>();
            if constexpr (!STATE_RESIDENT) {
                AscendC::DataCopy(state, ws_[src + off], count);
                SyncMte2V();
            }
            AscendC::Cast(out, state[STATE_RESIDENT ? off : 0], AscendC::RoundMode::CAST_RINT, count);
            SyncVMte2();
            matrixOut_.EnQue(out);
            out = matrixOut_.DeQue<T>();
            AscendC::DataCopy(wsT_[dst + off], out, count);
            matrixOut_.FreeTensor(out);
        }
    }

    __aicore__ inline void LoadGate(const DhuChunkInfo &info, uint64_t hv,
                                    AscendC::LocalTensor<float> gate, float &lastExp)
    {
        if (!p_->has_g) {
            AscendC::Duplicate(gate, 1.0f, 8);
            SyncVS();
            lastExp = gate.GetValue(0);
            SyncSV();
            return;
        }
        auto in = gateIn_.Get<GT>();
        AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(info.len * sizeof(GT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<GT> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(in, g_[GateBase(info.seq, hv, info.start)], copy, pad);
        SyncMte2V();
        if constexpr (std::is_same_v<GT, float>) {
            AscendC::Adds(gate, in, 0.0f, info.len);
        } else {
            AscendC::Cast(gate, in, AscendC::RoundMode::CAST_NONE, info.len);
        }
        SyncVMte2();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(gate, gate, 0.6931471805599453f, info.len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gate, gate, info.len);
        SyncVS();
        lastExp = gate.GetValue(info.len - 1);
        SyncSV();
    }

    __aicore__ inline void BuildDv2AndQg(const DhuChunkInfo &info, uint64_t hq, uint64_t hv, uint32_t core)
    {
        auto arena = calcBuf_.Get<float>();
        auto dvCalc = arena;
        auto rawCalc = arena[ROW_FLOAT_ELEMS];
        auto gate = arena[2 * ROW_FLOAT_ELEMS];
        auto rawScale = arena[2 * ROW_FLOAT_ELEMS + 128];
        auto broadcast = arena[2 * ROW_FLOAT_ELEMS + 256];
        float lastExp = 1.0f;
        LoadGate(info, hv, gate, lastExp);
        if (p_->has_g) {
            AscendC::Duplicate(rawScale, lastExp, info.len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(rawScale, rawScale, gate, info.len);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint64_t rowBase = 0; rowBase < info.len; rowBase += ROW_TILE) {
            const uint32_t rows = static_cast<uint32_t>(info.len - rowBase < ROW_TILE ? info.len - rowBase : ROW_TILE);
            const uint32_t elems = rows * V_DIM;
            auto raw = matrixIn_.AllocTensor<float>();
            auto dvIn = matrixIn_.AllocTensor<T>();
            auto out = matrixOut_.AllocTensor<T>();
            AscendC::DataCopy(raw, ws_[DhuMatrixFloatOffset(core, 1, *p_) + rowBase * V_DIM], elems);
            AscendC::DataCopy(dvIn, dv_[TokenBase(info.seq, hv, info.start + rowBase, V_DIM, p_->hv)], elems);
            SyncMte2V();
            AscendC::DataCopy(rawCalc, raw, elems);
            AscendC::Cast(dvCalc, dvIn, AscendC::RoundMode::CAST_NONE, elems);
            matrixIn_.FreeTensor(raw);
            matrixIn_.FreeTensor(dvIn);
            AscendC::PipeBarrier<PIPE_V>();
            if (p_->has_g) {
                ApplyRowScale(rawCalc, rawScale[rowBase], broadcast, rows, V_DIM);
            }
            AscendC::Add(dvCalc, dvCalc, rawCalc, elems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(out, dvCalc, AscendC::RoundMode::CAST_RINT, elems);
            matrixOut_.EnQue(out);
            out = matrixOut_.DeQue<T>();
            AscendC::DataCopy(dv2_[TokenBase(info.seq, hv, info.start + rowBase, V_DIM, p_->hv)], out, elems);
            AscendC::DataCopy(wsT_[DhuMatrixTypedOffset<T>(core, 2, *p_) + rowBase * V_DIM], out, elems);
            matrixOut_.FreeTensor(out);
        }

        for (uint64_t rowBase = 0; rowBase < info.len; rowBase += ROW_TILE) {
            const uint32_t rows = static_cast<uint32_t>(info.len - rowBase < ROW_TILE ? info.len - rowBase : ROW_TILE);
            const uint32_t elems = rows * 128;
            auto qIn = matrixIn_.AllocTensor<T>();
            AscendC::DataCopy(qIn, q_[TokenBase(info.seq, hq, info.start + rowBase, 128, p_->hk)], elems);
            SyncMte2V();
            AscendC::Cast(dvCalc, qIn, AscendC::RoundMode::CAST_NONE, elems);
            matrixIn_.FreeTensor(qIn);
            AscendC::PipeBarrier<PIPE_V>();
            if (p_->has_g) {
                ApplyRowScale(dvCalc, gate[rowBase], broadcast, rows, 128);
            }
            auto out = matrixOut_.AllocTensor<T>();
            AscendC::Cast(out, dvCalc, AscendC::RoundMode::CAST_RINT, elems);
            matrixOut_.EnQue(out);
            out = matrixOut_.DeQue<T>();
            AscendC::DataCopy(wsT_[DhuMatrixTypedOffset<T>(core, 3, *p_) + rowBase * 128], out, elems);
            matrixOut_.FreeTensor(out);
        }
    }

    __aicore__ inline void LoadGkLast(const DhuChunkInfo &info, uint64_t hv, AscendC::LocalTensor<float> factor)
    {
        if (!p_->has_gk) return;
        const uint64_t token = info.start + info.len - 1;
        const uint64_t src = GateBase(info.seq, hv, token) * p_->k;
        if constexpr (std::is_same_v<GT, float>) {
            AscendC::Duplicate(factor, 0.0f, p_->k);
            SyncVMte2();
            AscendC::DataCopy(factor, gk_[src], p_->k);
            SyncMte2V();
        } else {
            auto in = gateIn_.Get<GT>();
            AscendC::DataCopy(in, gk_[src], p_->k);
            SyncMte2V();
            AscendC::Cast(factor, in, AscendC::RoundMode::CAST_NONE, p_->k);
            SyncVMte2();
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(factor, factor, 0.6931471805599453f, p_->k);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(factor, factor, p_->k);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UpdateState(const DhuChunkInfo &info, uint64_t hv, uint32_t core)
    {
        auto arena = stateBuf_.Get<float>();
        auto state = arena;
        auto plus = arena[STATE_RESIDENT ? STATE_ELEMS : STATE_TILE_ELEMS];
        auto minus = plus[STATE_TILE_ELEMS];
        auto gate = calcBuf_.Get<float>()[2 * ROW_FLOAT_ELEMS];
        auto gkFactor = calcBuf_.Get<float>()[2 * ROW_FLOAT_ELEMS + 128];
        auto broadcast = calcBuf_.Get<float>()[2 * ROW_FLOAT_ELEMS + 256];
        float lastExp = 1.0f;
        LoadGate(info, hv, gate, lastExp);
        LoadGkLast(info, hv, gkFactor);
        const uint64_t elems = p_->k * p_->v;
        for (uint64_t off = 0; off < elems; off += STATE_TILE_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(elems - off < STATE_TILE_ELEMS ? elems - off : STATE_TILE_ELEMS);
            auto stateTile = state[STATE_RESIDENT ? off : 0];
            if constexpr (!STATE_RESIDENT) {
                AscendC::DataCopy(stateTile, ws_[DhuStateOffset(core, *p_) + off], count);
            }
            AscendC::DataCopy(plus, ws_[DhuMatrixFloatOffset(core, 4, *p_) + off], count);
            AscendC::DataCopy(minus, ws_[DhuMatrixFloatOffset(core, 5, *p_) + off], count);
            SyncMte2V();
            AscendC::Muls(stateTile, stateTile, lastExp, count);
            AscendC::PipeBarrier<PIPE_V>();
            if (p_->has_gk) {
                const uint32_t rows = count / V_DIM;
                const uint32_t rowBase = static_cast<uint32_t>(off / V_DIM);
                ApplyGkRowScale(stateTile, gkFactor[rowBase], rows, V_DIM);
            }
            AscendC::Muls(plus, plus, p_->scale, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(plus, plus, minus, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(stateTile, stateTile, plus, count);
            if constexpr (STATE_RESIDENT) {
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                SyncVMte3();
                AscendC::DataCopy(ws_[DhuStateOffset(core, *p_) + off], stateTile, count);
                SyncMte3Mte2();
            }
        }
    }

    __aicore__ inline void StoreDh0(uint64_t seq, uint64_t hv, uint32_t core)
    {
        auto state = stateBuf_.Get<float>();
        const uint64_t elems = p_->k * p_->v;
        const uint64_t dst = (seq * p_->hv + hv) * elems;
        for (uint64_t off = 0; off < elems; off += STATE_TILE_ELEMS) {
            const uint32_t count = static_cast<uint32_t>(elems - off < STATE_TILE_ELEMS ? elems - off : STATE_TILE_ELEMS);
            if constexpr (!STATE_RESIDENT) {
                AscendC::DataCopy(state, ws_[DhuStateOffset(core, *p_) + off], count);
                SyncMte2Mte3();
            }
            AscendC::DataCopy(dh0_[dst + off], state[STATE_RESIDENT ? off : 0], count);
            if constexpr (STATE_RESIDENT) {
                SyncMte3V();
            } else {
                SyncMte3Mte2();
            }
        }
    }

    AscendC::TPipe *pipe_{nullptr};
    AscendC::GlobalTensor<T> q_, dv_, dh_, dv2_, wsT_;
    AscendC::GlobalTensor<GT> g_, gk_;
    AscendC::GlobalTensor<float> dht_, dh0_, ws_;
    GM_ADDR cu_{nullptr};
    const ChunkGatedDeltaRuleBwdDhuTilingData *p_{nullptr};
    AscendC::TBuf<AscendC::TPosition::VECCALC> stateBuf_, calcBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> matrixIn_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> matrixOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateIn_;
    AscendC::TEventID mte2VEvent_, vMte3Event_, mte3Mte2Event_, vMte2Event_, mte2Mte3Event_, mte3VEvent_;
    AscendC::TEventID vSEvent_, sVEvent_;
    Catlass::Arch::CrossCoreFlagWithReverse<> vecToCube_{DHU_VEC_TO_CUBE_READY, DHU_CUBE_TO_VEC_FREE};
    Catlass::Arch::CrossCoreFlagWithReverse<> cubeToVec_{DHU_CUBE_TO_VEC_READY, DHU_VEC_TO_CUBE_FREE};
};
} // namespace GDN
#endif
