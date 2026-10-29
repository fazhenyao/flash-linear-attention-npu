// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef __RECURRENT_KDA_KERNEL_H_
#define __RECURRENT_KDA_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../recurrent_kda_tiling_data.h"
#include "recurrent_kda_regbase.h"

namespace RecurrentKda {
using namespace AscendC;
using namespace AscendC::MicroAPI;

struct RKDAInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR gate;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR cuSeqlens;
    GM_ADDR ssmStateIndices;
    GM_ADDR aLog;
    GM_ADDR dtBias;
    GM_ADDR numAcceptedTokens;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename inType, typename outType, typename stateType>
class RKDA {
    static constexpr uint16_t D = RegBase::DIM;
    static constexpr uint32_t BANK = RegBase::STATE_BYTES;
    static constexpr uint64_t INVALID = static_cast<uint64_t>(-1);
public:
    __aicore__ explicit RKDA(const RecurrentKdaTilingData *tiling) : td_(tiling) {}

    __aicore__ inline void Init(const RKDAInitParams &args, TPipe *pipe)
    {
        args_ = args;
        x_ = td_->windowSize;
        betaSlot_ = 32 * x_;
        if (betaSlot_ < 128) betaSlot_ = 128;
        // Exact order and sizes from the protected plan, section 2.4.
        qOffset_ = 3 * BANK;
        kOffset_ = qOffset_ + 512 * x_;
        vOffset_ = kOffset_ + 512 * x_;
        gateOffset_ = vOffset_ + 512 * x_;
        betaOffset_ = gateOffset_ + 1024 * x_;
        paramOffset_ = betaOffset_ + 2 * betaSlot_;
        outOffset_ = paramOffset_ + 1088;
        eventOffset_ = outOffset_ + 512 * x_;
        pipe->InitBuffer(storage_, eventOffset_ + 2048);
        ub_ = storage_.template Get<uint8_t>();
        q_.SetGlobalBuffer((__gm__ inType *)args.query);
        k_.SetGlobalBuffer((__gm__ inType *)args.key);
        v_.SetGlobalBuffer((__gm__ inType *)args.value);
        stateIn_.SetGlobalBuffer((__gm__ stateType *)args.initState);
        stateOut_.SetGlobalBuffer((__gm__ stateType *)args.finalState);
        out_.SetGlobalBuffer((__gm__ outType *)args.attnOut);
        aLog_.SetGlobalBuffer((__gm__ float *)args.aLog);
        bias_.SetGlobalBuffer((__gm__ float *)args.dtBias);
    }

#ifdef RKDA_EXECUTION_TRACE
    // Separate development-only GM records; no new synchronization or production workspace.
    __aicore__ inline void InitExecutionTrace(GM_ADDR records)
    {
        trace_.SetGlobalBuffer((__gm__ uint64_t *)records + GetBlockIdx() * 32768);
        traceCount_ = 0;
        trace_.SetValue(0, 0);
    }
#endif

    __aicore__ inline void Process()
    {
        if (x_ == 0 || 32 * x_ > betaSlot_ || !ValidateSequences()) return;
        if (!td_->hasSsmStateIndices) { ProcessFinalState(); return; }
        uint64_t cursor = GetBlockIdx();
        bool valid = true;
        Task current{}, next{};
        bool active = PrepareTask(cursor, current, valid);
        uint32_t initialBank = 0, parameterSlot = 0;
        bool prefetched = false;
        while (active) {
            const bool hasNext = PrepareTask(cursor, next, valid);
            if (!valid) {
                if (prefetched) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                return;
            }
            initialBank_ = initialBank;
            alternateBank_ = (initialBank + 1) % 3;
            RunTask(current, hasNext ? &next : nullptr, prefetched, parameterSlot, initialBank);
            if (!hasNext) break;
            current = next;
            parameterSlot ^= 1;
            prefetched = true;
        }
    }

private:
    struct Task {
        uint64_t id, request, head;
        int64_t begin, end;
    };

    // A final-state DMA may outlive its task. The following task uses the other
    // two banks, and starts in the input/out slot opposite the old final slot.
    __aicore__ inline void ProcessFinalState()
    {
        uint64_t cursor = GetBlockIdx();
        bool valid = true, prefetched = false, pending = false;
        Task current{}, next{}, following{};
        bool active = PrepareTask(cursor, current, valid);
        if (!active || !valid) return;
        bool hasNext = PrepareTask(cursor, next, valid), hasFollowing = false;
        if (!valid) return;
        uint32_t initial = 0, alternate = 1, parameterSlot = 0, inputSlot = 0;
        // The output State row shape is invariant across all tasks on this core.
        const DataCopyExtParams stateCopy{D, D * sizeof(stateType), 0,
            static_cast<uint32_t>((td_->stateOutStride2 - D) * sizeof(stateType)), 0};
#ifdef RKDA_EXECUTION_TRACE
        auto events = (__ubuf__ volatile uint32_t *)ub_[eventOffset_].GetPhyAddr();
        events[8] = 0; // No bank-done consumer on this path.
#endif
        while (active) {
            if (!valid) {
                if (prefetched) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                if (pending) WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                return;
            }
            initialBank_ = initial;
            alternateBank_ = alternate;
            RunFinalStateTask(current, hasNext ? &next : nullptr, prefetched,
                              parameterSlot, inputSlot, pending, cursor, following, hasFollowing, valid, stateCopy);
            if (!valid) {
                if (hasNext) WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                if (pending) WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                return;
            }
            if (!hasNext) break;
            const uint32_t finalBank = ((current.end - current.begin) & 1) ? alternate : initial;
            const uint32_t nextInitial = 3 - initial - alternate;
            alternate = 3 - finalBank - nextInitial;
            initial = nextInitial;
            current = next;
            next = following;
            hasNext = hasFollowing;
            parameterSlot ^= 1;
            prefetched = true;
        }
        // The final DMA has no subsequent VF consumer; do not publish unused done.
        if (pending) {
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            Trace(12, current.id, uint32_t(current.end - current.begin), inputSlot ^ 1);
        }
    }

    __aicore__ inline void RunFinalStateTask(const Task &current, const Task *nextTask,
                                           bool prefetched, uint32_t parameterSlot,
                                           uint32_t &slot, bool &pending, uint64_t &cursor,
                                           Task &following, bool &hasFollowing, bool &validTask,
                                           const DataCopyExtParams &stateCopy)
    {
        const int64_t begin = current.begin, end = current.end;
        const uint64_t head = current.head, task = current.id;
        const uint32_t spareBank = 3 - initialBank_ - alternateBank_;
        const uint32_t finalBank = ((end - begin) & 1) ? alternateBank_ : initialBank_;
        const bool writeState = td_->inplaceFinalState || td_->outputFinalState;
        const uint64_t stateOffset = current.request * td_->stateOutStride0 + head * td_->stateOutStride1;
        const uint32_t parameter = paramOffset_ + parameterSlot * 544;
        auto events = (__ubuf__ volatile uint32_t *)ub_[eventOffset_].GetPhyAddr();
        events[0] = 0;
        hasFollowing = false;
        Trace(0, task, 0, slot);
        if (prefetched) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Trace(8, task, 0, initialBank_);
        } else {
            PrefetchTask(current, initialBank_, parameterSlot);
            Prefetch(begin, ValidTokens(begin, end), head, slot);
        }
        // The spare bank can still be read by the previous task's final DMA.
        bool deferStatePrefetch = nextTask && pending;
        for (int64_t tile = begin; tile < end; tile += x_, slot ^= 1) {
            const uint32_t valid = ValidTokens(tile, end), base = uint32_t(tile - begin);
            const bool lastTile = tile + valid == end;
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if (!lastTile) {
                Prefetch(tile + valid, ValidTokens(tile + valid, end), head, slot ^ 1);
            } else if (nextTask) {
                Prefetch(nextTask->begin, ValidTokens(nextTask->begin, nextTask->end), nextTask->head, slot ^ 1);
                Trace(9, nextTask->id, 0, slot ^ 1);
            }
            if (nextTask && base == 0 && !deferStatePrefetch)
                PrefetchNext(*nextTask, parameterSlot, spareBank);
            Trace(1, task, base, slot);
            Dispatch(base, valid, slot, parameter, lastTile);
            if (base == 0 && nextTask)
                hasFollowing = PrepareTask(cursor, following, validTask);
            // The old batch belongs to the opposite out slot. Reclaim it while
            // this VF runs, also releasing any old final-State DMA source bank.
            if (pending) {
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                pending = false;
                Trace(10, task, base, slot ^ 1);
            }
            if (deferStatePrefetch) {
                PrefetchNext(*nextTask, parameterSlot, spareBank);
                deferStatePrefetch = false;
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Trace(5, task, base + valid, slot);
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            CopyOutputTile(tile, valid, head, slot);
            if (lastTile && writeState)
                DataCopyPad(stateOut_[stateOffset], ub_[finalBank * BANK].template ReinterpretCast<stateType>(), stateCopy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            pending = true;
            Trace(11, task, base + valid, slot);
        }
        Trace(6, task, uint32_t(end - begin), slot);
    }

    __aicore__ inline bool PrepareTask(uint64_t &cursor, Task &task, bool &valid) const
    {
        const uint64_t count = uint64_t(td_->b) * td_->nv;
        while (cursor < count) {
            const uint64_t id = cursor;
            cursor += GetBlockNum();
            const uint64_t request = id / td_->nv;
            const int64_t begin = Sequence(request), end = Sequence(request + 1);
            if (begin == end) continue;
            if (!ValidateSlots(request, begin, end)) { valid = false; return false; }
            task = {id, request, id % td_->nv, begin, end};
            return true;
        }
        return false;
    }

    __aicore__ inline int64_t Metadata(GM_ADDR address, uint32_t dtype, uint64_t index) const
    {
        if (dtype == 0) {
            GlobalTensor<int32_t> values;
            values.SetGlobalBuffer((__gm__ int32_t *)address);
            return values.GetValue(index);
        }
        GlobalTensor<int64_t> values;
        values.SetGlobalBuffer((__gm__ int64_t *)address);
        return values.GetValue(index);
    }

    __aicore__ inline int64_t Sequence(uint64_t request) const
    {
        return td_->hasCuSeqlens ? Metadata(args_.cuSeqlens, td_->cuSeqlensDtype, request) :
                                   int64_t(request * td_->seqLen);
    }

    __aicore__ inline uint64_t Slot(uint64_t request, int64_t begin, int64_t token) const
    {
        if (!td_->hasSsmStateIndices) return request;
        const uint64_t offset = td_->ssmStateStride ?
            request * td_->ssmStateStride + uint64_t(token - begin) : uint64_t(token);
        const int64_t slot = Metadata(args_.ssmStateIndices, td_->ssmStateIndicesDtype, offset);
        return slot >= 0 && uint64_t(slot) < td_->sBlockNum ? uint64_t(slot) : INVALID;
    }

    __aicore__ inline bool ValidateSequences() const
    {
        if (!td_->hasCuSeqlens) return true;
        if (Sequence(0) != 0) return false;
        int64_t previous = 0;
        for (uint64_t request = 0; request < td_->b; ++request) {
            int64_t end = Sequence(request + 1);
            if (end < previous || end > td_->t ||
                (td_->hasSsmStateIndices && td_->ssmStateStride &&
                 uint64_t(end - previous) > td_->ssmStateStride)) return false;
            previous = end;
        }
        return true;
    }

    __aicore__ inline bool ValidateSlots(uint64_t request, int64_t begin, int64_t end) const
    {
        if (td_->hasAcceptedTokens) {
            int64_t accepted = Metadata(args_.numAcceptedTokens, td_->acceptedTokensDtype, request);
            if (accepted <= 0 || accepted > end - begin) return false;
        }
        if (!td_->hasSsmStateIndices) return request < td_->sBlockNum;
        for (int64_t token = begin; token < end; ++token)
            if (Slot(request, begin, token) == INVALID) return false;
        return true;
    }

    template <class T>
    __aicore__ inline void CopyRows(GlobalTensor<T> &gm, uint64_t offset, uint64_t stride,
                                    uint32_t destination, uint32_t width, uint32_t rows)
    {
        auto local = ub_[destination].template ReinterpretCast<T>();
        DataCopyExtParams copy{static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(T)),
                               static_cast<uint32_t>((stride - width) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
    }

    template <class T>
    __aicore__ inline void CopyAux(GM_ADDR address, uint64_t head, int64_t token,
                                   uint32_t slot, uint32_t rows, bool beta)
    {
        GlobalTensor<T> gm;
        gm.SetGlobalBuffer((__gm__ T *)address);
        if (beta) {
            // MTE2's short rows land at 32-byte intervals, inside betaSlot_.
            CopyRows(gm, uint64_t(token) * td_->nv + head, td_->nv,
                     betaOffset_ + slot * betaSlot_, 1, rows);
        } else {
            CopyRows(gm, (uint64_t(token) * td_->nv + head) * D, uint64_t(td_->nv) * D,
                     gateOffset_ + slot * 512 * x_, D, rows);
        }
    }

    __aicore__ inline void Prefetch(int64_t token, uint32_t rows, uint64_t head, uint32_t slot)
    {
        const uint64_t hk = head / (td_->nv / td_->nk);
        CopyRows(q_, uint64_t(token) * td_->queryTokenStride + hk * td_->queryHeadStride,
                 td_->queryTokenStride, qOffset_ + slot * 256 * x_, D, rows);
        CopyRows(k_, uint64_t(token) * td_->keyTokenStride + hk * td_->keyHeadStride,
                 td_->keyTokenStride, kOffset_ + slot * 256 * x_, D, rows);
        CopyRows(v_, uint64_t(token) * td_->valueTokenStride + head * td_->valueHeadStride,
                 td_->valueTokenStride, vOffset_ + slot * 256 * x_, D, rows);
#if defined(DTYPE_GATE) && defined(DTYPE_BETA)
        // The OPP binary's input types also determine the DMA width and strides.
        CopyAux<DTYPE_GATE>(args_.gate, head, token, slot, rows, false);
        CopyAux<DTYPE_BETA>(args_.beta, head, token, slot, rows, true);
#else
        if (td_->gateDtype == 0) CopyAux<float>(args_.gate, head, token, slot, rows, false);
        else if (td_->gateDtype == 1) CopyAux<bfloat16_t>(args_.gate, head, token, slot, rows, false);
        else CopyAux<half>(args_.gate, head, token, slot, rows, false);
        if (td_->betaDtype == 0) CopyAux<float>(args_.beta, head, token, slot, rows, true);
        else if (td_->betaDtype == 1) CopyAux<bfloat16_t>(args_.beta, head, token, slot, rows, true);
        else CopyAux<half>(args_.beta, head, token, slot, rows, true);
#endif
    }

    __aicore__ inline uint32_t ValidTokens(int64_t begin, int64_t end) const
    {
        return uint64_t(end - begin) < x_ ? uint32_t(end - begin) : x_;
    }

    __aicore__ inline void CopyOutputTile(int64_t token, uint32_t rows, uint64_t head, uint32_t slot)
    {
        // UB rows are contiguous; GM rows for one value head are Nv rows apart.
        DataCopyExtParams copy{static_cast<uint16_t>(rows), D * sizeof(outType), 0,
            static_cast<uint32_t>(uint64_t(td_->nv - 1) * D * sizeof(outType)), 0};
        DataCopyPad(out_[(uint64_t(token) * td_->nv + head) * D],
                    ub_[outOffset_ + slot * x_ * 256].template ReinterpretCast<outType>(), copy);
    }

    __aicore__ inline void Complete(uint32_t count, uint64_t task)
    {
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        auto events = (__ubuf__ volatile uint32_t *)ub_[eventOffset_].GetPhyAddr();
        events[8] = count;
        Trace(4, task, count, 0);
    }

    __aicore__ inline void PrefetchTask(const Task &task, uint32_t bank, uint32_t parameterSlot)
    {
        int64_t initialToken = task.begin;
        if (td_->hasAcceptedTokens)
            initialToken += Metadata(args_.numAcceptedTokens, td_->acceptedTokensDtype, task.request) - 1;
        const uint64_t initial = Slot(task.request, task.begin, initialToken);
        CopyRows(stateIn_, initial * td_->stateInStride0 + task.head * td_->stateInStride1,
                 td_->stateInStride2, bank * BANK, D, D);
        const uint32_t parameter = paramOffset_ + parameterSlot * 544;
        if (td_->hasALog) {
            CopyRows(aLog_, task.head, 1, parameter, 1, 1);
            if (td_->hasDtBias) CopyRows(bias_, task.head * D, D, parameter + 32, D, 1);
        }
    }

    __aicore__ inline void PrefetchNext(const Task &next, uint32_t parameterSlot, uint32_t bank)
    {
        PrefetchTask(next, bank, parameterSlot ^ 1);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Trace(7, next.id, 0, bank);
    }

    __aicore__ inline void RunTask(const Task &current, const Task *nextTask, bool prefetched,
                                  uint32_t parameterSlot, uint32_t &nextInitialBank)
    {
        const uint64_t task = current.id, request = current.request, head = current.head;
        const int64_t begin = current.begin, end = current.end;
        auto events = (__ubuf__ volatile uint32_t *)ub_[eventOffset_].GetPhyAddr();
        events[0] = 0; // monotonically increasing VF publication count
        events[8] = 0; // ordered MTE3 completion count, in a separate 32B block
        Trace(0, task, 0, 0);
        if (prefetched) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Trace(8, task, 0, initialBank_);
        } else {
            PrefetchTask(current, initialBank_, parameterSlot);
        }
        const uint32_t parameter = paramOffset_ + parameterSlot * 544;
        Prefetch(begin, ValidTokens(begin, end), head, 0);
        uint32_t pending = 0, slot = 0;
        bool pendingOut = false;
        const bool writeState = td_->inplaceFinalState || td_->outputFinalState;
        const DataCopyExtParams stateCopy{D, D * sizeof(stateType), 0,
            static_cast<uint32_t>((td_->stateOutStride2 - D) * sizeof(stateType)), 0};
        for (int64_t tile = begin; tile < end; tile += x_, slot ^= 1) {
            const uint32_t valid = ValidTokens(tile, end), base = uint32_t(tile - begin);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if (tile + valid < end)
                Prefetch(tile + valid, ValidTokens(tile + valid, end), head, slot ^ 1);
            const bool lastTile = tile + valid == end;
            // Two banks alternate throughout this task; the third belongs to the next task.
            if (nextTask && base == 0) {
                nextInitialBank = (initialBank_ + 2) % 3;
                PrefetchNext(*nextTask, parameterSlot, nextInitialBank);
            }
            Trace(1, task, base, slot);
            Dispatch(base, valid, slot, parameter, lastTile);
            // State snapshots remain token-granular: the VF can need a bank
            // released before it can finish the tile. Out has its own event.
            for (uint32_t t = 0; writeState && t < valid; ++t) {
                const uint32_t token = base + t, count = token + 1;
                // Release an old snapshot before waiting for the next publication.
                // The whole tile VF is already running asynchronously.
                if (pending) { Complete(pending, task); pending = 0; }
                while (events[0] < count) {}
                Trace(2, task, count, slot);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                {
                    uint32_t bank = (initialBank_ + ((token + 1) & 1)) % 3;
                    const uint64_t destination = Slot(request, begin, tile + t) * td_->stateOutStride0 +
                                                 head * td_->stateOutStride1;
                    DataCopyPad(stateOut_[destination], ub_[bank * BANK].template ReinterpretCast<stateType>(), stateCopy);
                }
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                pending = count;
                Trace(3, task, count, slot);
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Trace(5, task, base + valid, slot);
            // Reclaim the previous tile's out, never treating State-done as
            // proof that an out slot is free. EVENT1 is distinct from State's 0.
            if (pendingOut) {
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
                Trace(12, task, base, slot ^ 1);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            CopyOutputTile(tile, valid, head, slot);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
            pendingOut = true;
            Trace(11, task, base + valid, slot);
        }
        if (pending) Complete(pending, task);
        if (pendingOut) {
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
            Trace(12, task, uint32_t(end - begin), slot ^ 1);
        }
        Trace(6, task, uint32_t(end - begin), slot);
    }

    template <class Gate>
    __aicore__ inline void DispatchBeta(uint32_t base, uint32_t valid, uint32_t slot,
                                       uint32_t parameter, bool last)
    {
        if (td_->betaDtype == 0) DispatchLayout<Gate, float>(base, valid, slot, parameter, last);
        else if (td_->betaDtype == 1) DispatchLayout<Gate, bfloat16_t>(base, valid, slot, parameter, last);
        else DispatchLayout<Gate, half>(base, valid, slot, parameter, last);
    }

    template <class Gate, class Beta>
    __aicore__ inline void DispatchLayout(uint32_t base, uint32_t valid, uint32_t slot,
                                         uint32_t parameter, bool last)
    {
        if (td_->stateVFirst) {
            // Specialize the common fused recurrence to retire runtime flags
            // before the VF's State/out row loops. All other flags use the generic VF.
            if (td_->useQkL2norm && td_->hasALog && td_->hasDtBias && td_->useBetaSigmoid &&
                !td_->allowNegEigval && !td_->safeGate && !td_->hasSsmStateIndices) {
                if (x_ == 4) Tile<Gate, Beta, true, true, 4>(base, valid, slot, parameter, last);
                else Tile<Gate, Beta, true, true>(base, valid, slot, parameter, last);
            } else Tile<Gate, Beta, true>(base, valid, slot, parameter, last);
        } else Tile<Gate, Beta, false>(base, valid, slot, parameter, last);
    }

    __aicore__ inline void Dispatch(uint32_t base, uint32_t valid, uint32_t slot,
                                    uint32_t parameter, bool last)
    {
#if defined(DTYPE_GATE) && defined(DTYPE_BETA)
        // OPP already selects a binary for each input dtype combination.
        DispatchLayout<DTYPE_GATE, DTYPE_BETA>(base, valid, slot, parameter, last);
#else
        // Direct kernel callers may supply dtype codes only through tiling.
        if (td_->gateDtype == 0) DispatchBeta<float>(base, valid, slot, parameter, last);
        else if (td_->gateDtype == 1) DispatchBeta<bfloat16_t>(base, valid, slot, parameter, last);
        else DispatchBeta<half>(base, valid, slot, parameter, last);
#endif
    }

    template <class Gate, class Beta, bool vFirst, bool fused = false, uint32_t fixedWindow = 0>
    __aicore__ inline void Tile(uint32_t base, uint32_t valid, uint32_t slot,
                                uint32_t parameter, bool last)
    {
        auto memory = (__ubuf__ uint8_t *)ub_.GetPhyAddr();
        auto capturedQuery = (__ubuf__ inType *)(memory + qOffset_ + slot * 256 * x_);
        auto capturedKey = (__ubuf__ inType *)(memory + kOffset_ + slot * 256 * x_);
        auto capturedValue = (__ubuf__ inType *)(memory + vOffset_ + slot * 256 * x_);
        auto capturedGate = (__ubuf__ Gate *)(memory + gateOffset_ + slot * 512 * x_);
        auto capturedBeta = (__ubuf__ Beta *)(memory + betaOffset_ + slot * betaSlot_);
        auto capturedOutput = (__ubuf__ outType *)(memory + outOffset_ + slot * 256 * x_);
        auto capturedParams = (__ubuf__ float *)(memory + parameter);
        auto capturedEvents = (__ubuf__ volatile uint32_t *)(memory + eventOffset_);
        const uint32_t bankSlot = initialBank_ | (alternateBank_ << 2) | (slot << 4);
        // X=8 fits in the launch descriptor's valid-count field; the UB byte offset
        // remains encoded in the upper 24 bits.
        const uint32_t launch = bankSlot | (valid << 5) | (parameter << 8);
        const bool norm = fused || td_->useQkL2norm, rawGate = fused || td_->hasALog;
        const bool bias = fused || td_->hasDtBias;
        const bool sigmoid = fused || td_->useBetaSigmoid;
        const bool negative = fused ? false : td_->allowNegEigval, safe = fused ? false : td_->safeGate;
        const bool write = td_->inplaceFinalState || td_->outputFinalState, indexed = fused ? false : td_->hasSsmStateIndices;
        const float scale = td_->scale, lower = td_->lowerBound;
        const uint16_t tokenCount = static_cast<uint16_t>(valid);
        __VEC_SCOPE__
        {
            const uint32_t config = fixedWindow ? launch : bankSlot;
            const uint32_t initialBank = config & 3, alternateBank = (config >> 2) & 3;
            const uint32_t inputSlot = (config >> 4) & 1;
            const uint16_t count = fixedWindow ? ((config >> 5) & 7) : tokenCount;
            // Fixed-window specialization leaves general strides/dtypes on the original path.
            constexpr uint32_t betaStride = 128;
            const uint32_t rowSlot = inputSlot * 256 * fixedWindow;
            constexpr uint32_t betaOffset = 3 * BANK + 2560 * fixedWindow;
            constexpr uint32_t outputOffset = betaOffset + 2 * betaStride + 1088;
            auto query = fixedWindow ? (__ubuf__ inType *)(memory + 3 * BANK + rowSlot) : capturedQuery;
            auto key = fixedWindow ? (__ubuf__ inType *)(memory + 3 * BANK + 512 * fixedWindow + rowSlot) : capturedKey;
            auto value = fixedWindow ? (__ubuf__ inType *)(memory + 3 * BANK + 1024 * fixedWindow + rowSlot) : capturedValue;
            auto gate = fixedWindow ? (__ubuf__ Gate *)(memory + 3 * BANK + 1536 * fixedWindow + 2 * rowSlot) : capturedGate;
            auto beta = fixedWindow ? (__ubuf__ Beta *)(memory + betaOffset + inputSlot * betaStride) : capturedBeta;
            auto output = fixedWindow ? (__ubuf__ outType *)(memory + outputOffset + rowSlot) : capturedOutput;
            auto params = fixedWindow ? (__ubuf__ float *)(memory + (config >> 8)) : capturedParams;
            auto events = fixedWindow ? (__ubuf__ volatile uint32_t *)(memory + outputOffset + 512 * fixedWindow) : capturedEvents;
            MaskReg mask = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> q0, q1, k0, k1, alpha0, alpha1, strength, a;
            RegTensor<float> temporary;
            RegTensor<uint32_t> columns, pairColumns;
            MaskReg firstFour;
            if constexpr (vFirst) {
                Arange((RegTensor<int32_t> &)columns, 4);
                // Register Gather wraps indices modulo LANES, giving a four-lane rotation.
                uint32_t count = 4;
                firstFour = UpdateMask<float>(count);
            } else {
                Arange((RegTensor<int32_t> &)columns, 0);
                Muls(columns, columns, D, mask);
                if constexpr (!std::is_same<stateType, float>::value)
                    ShiftRights(pairColumns, columns, int16_t(1), mask);
            }
            // Per-head parameters are loaded once per tile and kept in registers.
            if (rawGate) {
                RegBase::Broadcast(a, params, mask);
                Exp(a, a, mask);
            }
            #pragma unroll 1
            for (uint16_t t = 0; t < count; ++t) {
                const uint32_t token = base + t;
                const uint32_t cur = (token & 1) ? alternateBank : initialBank;
                const uint32_t next = (token & 1) ? initialBank : alternateBank;
                auto source = (__ubuf__ stateType *)(memory + cur * BANK);
                auto destination = (__ubuf__ stateType *)(memory + next * BANK);
                // Only a bank still owned by an earlier snapshot needs a wait.
                if (write && indexed) {
                    if (token >= 2) while (events[8] < token - 1) {}
                }
                LocalMemBar<MemType::SCALAR_STORE, MemType::VEC_LOAD>();
                RegBase::Load(q0, query + t * D, mask);
                RegBase::Load(q1, query + t * D + 64, mask);
                RegBase::Load(k0, key + t * D, mask);
                RegBase::Load(k1, key + t * D + 64, mask);
                if (norm) { RegBase::Normalize(q0, q1, mask); RegBase::Normalize(k0, k1, mask); }
                Muls(q0, q0, scale, mask);
                Muls(q1, q1, scale, mask);
                RegBase::Broadcast(strength, beta + t * (32 / sizeof(Beta)), mask);
                if (sigmoid) {
                    RegBase::Sigmoid(strength, mask);
                    if (negative) Muls(strength, strength, 2.0f, mask);
                }
                RegBase::Load(alpha0, gate + t * D, mask);
                RegBase::Load(alpha1, gate + t * D + 64, mask);
                if (rawGate && bias) {
                    DataCopy(temporary, params + 8); Add(alpha0, alpha0, temporary, mask);
                    DataCopy(temporary, params + 8 + 64); Add(alpha1, alpha1, temporary, mask);
                }
                RegBase::Decay(alpha0, a, rawGate, safe, lower, mask);
                RegBase::Decay(alpha1, a, rawGate, safe, lower, mask);
                // Expand only V-first to four rows; keep K-first on its R2 path.
                if constexpr (vFirst) {
                    RegTensor<float> outLo;
                    Duplicate(outLo, 0.0f);
                    // Two half loops avoid selecting an accumulator on every block.
                    // Both halves stay FP32 in registers until the full token is ready.
                    #pragma unroll 1
                    for (uint16_t row = 0; row < RegBase::LANES; row += 4) {
                        RegBase::UpdateRows4<stateType, inType>(
                            destination, source, value + t * D, outLo, columns, firstFour, row,
                            q0, q1, k0, k1, alpha0, alpha1, strength, mask);
                    }
                    RegTensor<float> outHi;
                    Duplicate(outHi, 0.0f);
                    #pragma unroll 1
                    for (uint16_t row = RegBase::LANES; row < D; row += 4) {
                        RegBase::UpdateRows4<stateType, inType>(
                            destination, source, value + t * D, outHi, columns, firstFour, row,
                            q0, q1, k0, k1, alpha0, alpha1, strength, mask);
                    }
                    RegBase::Store(output + t * D, outLo, mask);
                    RegBase::Store(output + t * D + RegBase::LANES, outHi, mask);
                } else {
                    #pragma unroll 1
                    for (uint16_t row = 0; row < D; row += 2) {
                        RegBase::UpdateRows2<stateType, inType, outType, false>(
                            destination, source, value + t * D, output + t * D, row,
                            q0, q1, k0, k1, alpha0, alpha1, strength, columns, pairColumns, mask);
                    }
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                LocalMemBar<MemType::VEC_STORE, MemType::SCALAR_STORE>();
                events[0] = token + 1;
            }
        }
    }

    __aicore__ inline void Trace(uint64_t event, uint64_t task, uint64_t token, uint64_t slot)
    {
#ifdef RKDA_EXECUTION_TRACE
        const uint64_t row = 1 + traceCount_ * 8;
        trace_.SetValue(row, event); trace_.SetValue(row + 1, task);
        trace_.SetValue(row + 2, token); trace_.SetValue(row + 3, slot);
        trace_.SetValue(row + 4, GetBlockNum());
        trace_.SetValue(row + 5, event == 7 || event == 8 ? slot : ((token & 1) ? alternateBank_ : initialBank_));
        auto events = (__ubuf__ volatile uint32_t *)ub_[eventOffset_].GetPhyAddr();
        trace_.SetValue(row + 6, events[0]);
        trace_.SetValue(row + 7, event == 1 ? alternateBank_ : events[8]);
        trace_.SetValue(0, ++traceCount_);
#endif
    }

    const RecurrentKdaTilingData *td_;
    RKDAInitParams args_;
    TBuf<TPosition::VECCALC> storage_;
    LocalTensor<uint8_t> ub_;
    GlobalTensor<inType> q_, k_, v_;
    GlobalTensor<stateType> stateIn_, stateOut_;
    GlobalTensor<outType> out_;
    GlobalTensor<float> aLog_, bias_;
    uint32_t x_, betaSlot_, qOffset_, kOffset_, vOffset_, gateOffset_, betaOffset_, initialBank_, alternateBank_;
    uint32_t paramOffset_, outOffset_, eventOffset_;
#ifdef RKDA_EXECUTION_TRACE
    GlobalTensor<uint64_t> trace_;
    uint64_t traceCount_ = 0;
#endif
};
} // namespace RecurrentKda
#endif
