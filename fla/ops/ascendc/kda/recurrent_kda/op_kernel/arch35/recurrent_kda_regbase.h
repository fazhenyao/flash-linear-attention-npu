// Copyright (c) 2025-2026 Tianjin University, Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef RECURRENT_KDA_REGBASE_H
#define RECURRENT_KDA_REGBASE_H

#include "kernel_operator.h"

namespace RecurrentKda {
namespace RegBase {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t DIM = 128;
constexpr uint32_t LANES = 64;
constexpr uint32_t STATE_BYTES = DIM * DIM * sizeof(float);

// Each register contains 64 FP32 columns. Low precision inputs stay raw in UB.
template <class T>
__simd_callee__ inline void Load(RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy(dst, src);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, T, cast>(dst, raw, mask);
    }
}

template <class T>
__simd_callee__ inline void Broadcast(RegTensor<float> &dst, __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, src);
    } else {
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_BRC_B16>(raw, src);
        uint32_t one = 1;
        MaskReg first = UpdateMask<float>(one);
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, T, cast>(dst, raw, first);
        Duplicate(dst, dst, mask);
    }
}

template <class T, bool firstOnly = false>
__simd_callee__ inline void Store(__ubuf__ T *dst, RegTensor<float> &src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        if constexpr (firstOnly) DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst, src, mask);
        else DataCopy(dst, src, mask);
    } else {
        RegTensor<T> raw;
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::NO_SAT,
                                    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        if constexpr (firstOnly) {
            uint32_t one = 1;
            MaskReg first = UpdateMask<float>(one);
            Cast<T, float, cast>(raw, src, first);
            DataCopy<T, StoreDist::DIST_FIRST_ELEMENT_B16>(dst, raw, mask);
        } else {
            Cast<T, float, cast>(raw, src, mask);
            DataCopy<T, StoreDist::DIST_PACK_B32>(dst, raw, mask);
        }
    }
}

// Gather offsets are in elements; K-first physical storage is preserved in all banks.
template <class T>
__simd_callee__ inline void GatherRow(RegTensor<float> &dst, __ubuf__ T *src,
                                     RegTensor<uint32_t> &columns, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        Gather(dst, src, columns, mask);
    } else {
        RegTensor<uint16_t> offsets;
        RegTensor<T> packed;
        RegTensor<uint32_t> unpacked;
        Pack(offsets, columns);
        uint32_t count = LANES;
        MaskReg halfMask = UpdateMask<T>(count);
        Gather(packed, src, offsets, halfMask);
        UnPack(unpacked, (RegTensor<uint16_t> &)packed);
        static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        Cast<float, T, cast>(dst, (RegTensor<T> &)unpacked, mask);
    }
}

// K-first BF16 rows share 32-bit words. Access adjacent V rows as pairs,
// retaining BF16 bits in registers while each row's FP32 math stays sequential.
__simd_callee__ inline void LoadPairRow(RegTensor<float> &dst, RegTensor<uint32_t> &packed,
                                      __ubuf__ bfloat16_t *src, RegTensor<uint32_t> &columns,
                                      bool odd, MaskReg &mask)
{
    static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::UNKNOWN,
                                      MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    if (odd) {
        RegTensor<uint32_t> high;
        ShiftRights(high, packed, int16_t(16), mask);
        Cast<float, bfloat16_t, cast>(dst, (RegTensor<bfloat16_t> &)high, mask);
    } else {
        Gather(packed, (__ubuf__ uint32_t *)src, columns, mask);
        Cast<float, bfloat16_t, cast>(dst, (RegTensor<bfloat16_t> &)packed, mask);
    }
}

__simd_callee__ inline void StorePairRow(__ubuf__ bfloat16_t *dst, RegTensor<uint32_t> &low,
                                       RegTensor<float> &src, RegTensor<uint32_t> &columns,
                                       bool odd, MaskReg &mask)
{
    static constexpr CastTrait cast = {RegLayout::ZERO, SatMode::NO_SAT,
                                      MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if (odd) {
        RegTensor<bfloat16_t> raw;
        RegTensor<uint32_t> high, pair;
        Cast<bfloat16_t, float, cast>(raw, src, mask);
        ShiftLefts(high, (RegTensor<uint32_t> &)raw, int16_t(16), mask);
        Or(pair, low, high, mask);
        Scatter((__ubuf__ uint32_t *)dst, pair, columns, mask);
    } else {
        Cast<bfloat16_t, float, cast>((RegTensor<bfloat16_t> &)low, src, mask);
        ShiftLefts(low, low, int16_t(16), mask);
        ShiftRights(low, low, int16_t(16), mask);
    }
}

__simd_callee__ inline void Sum128(RegTensor<float> &sum, RegTensor<float> &a,
                                  RegTensor<float> &b, MaskReg &mask)
{
    Add(sum, a, b, mask);
    ReduceSum(sum, sum, mask);
    Duplicate(sum, sum, mask);
}

// Keep two complete mathematical V rows live. Interleave independent rows without
// changing either row's Mul/Add or reduction order; only final results reach UB.
template <class State, class Value, class Out, bool vFirst>
__simd_callee__ inline void UpdateRows2(
    __ubuf__ State *destination, __ubuf__ State *source,
    __ubuf__ Value *value, __ubuf__ Out *output, uint16_t row,
    RegTensor<float> &q0, RegTensor<float> &q1,
    RegTensor<float> &k0, RegTensor<float> &k1,
    RegTensor<float> &alpha0, RegTensor<float> &alpha1,
    RegTensor<float> &strength, RegTensor<uint32_t> &columns,
    RegTensor<uint32_t> &pairColumns, MaskReg &mask)
{
    RegTensor<float> even0, even1, odd0, odd1;
    if constexpr (vFirst) {
        Load(even0, source + row * DIM, mask);
        Load(even1, source + row * DIM + LANES, mask);
        Load(odd0, source + (row + 1) * DIM, mask);
        Load(odd1, source + (row + 1) * DIM + LANES, mask);
    } else if constexpr (std::is_same<State, float>::value) {
        GatherRow(even0, source + row, columns, mask);
        GatherRow(even1, source + row + LANES * DIM, columns, mask);
        GatherRow(odd0, source + row + 1, columns, mask);
        GatherRow(odd1, source + row + 1 + LANES * DIM, columns, mask);
    } else {
        RegTensor<uint32_t> packed0, packed1;
        LoadPairRow(even0, packed0, source + row, pairColumns, false, mask);
        LoadPairRow(even1, packed1, source + row + LANES * DIM, pairColumns, false, mask);
        LoadPairRow(odd0, packed0, source + row, pairColumns, true, mask);
        LoadPairRow(odd1, packed1, source + row + LANES * DIM, pairColumns, true, mask);
    }

    Mul(even0, even0, alpha0, mask); Mul(odd0, odd0, alpha0, mask);
    Mul(even1, even1, alpha1, mask); Mul(odd1, odd1, alpha1, mask);
    RegTensor<float> evenP0, evenP1, oddP0, oddP1;
    Mul(evenP0, even0, k0, mask); Mul(oddP0, odd0, k0, mask);
    Mul(evenP1, even1, k1, mask); Mul(oddP1, odd1, k1, mask);
    RegTensor<float> evenPrediction, oddPrediction, evenDelta, oddDelta;
    Sum128(evenPrediction, evenP0, evenP1, mask);
    Sum128(oddPrediction, oddP0, oddP1, mask);
    Broadcast(evenDelta, value + row, mask);
    Broadcast(oddDelta, value + row + 1, mask);
    Sub(evenDelta, evenDelta, evenPrediction, mask);
    Sub(oddDelta, oddDelta, oddPrediction, mask);
    Mul(evenDelta, evenDelta, strength, mask);
    Mul(oddDelta, oddDelta, strength, mask);

    // Reuse product registers after prediction instead of allocating a State shadow.
    Mul(evenP0, evenDelta, k0, mask); Mul(oddP0, oddDelta, k0, mask);
    Add(even0, even0, evenP0, mask); Add(odd0, odd0, oddP0, mask);
    Mul(evenP1, evenDelta, k1, mask); Mul(oddP1, oddDelta, k1, mask);
    Add(even1, even1, evenP1, mask); Add(odd1, odd1, oddP1, mask);

    // Both outputs consume FP32 State before either State row is rounded to BF16.
    Mul(evenP0, even0, q0, mask); Mul(oddP0, odd0, q0, mask);
    Mul(evenP1, even1, q1, mask); Mul(oddP1, odd1, q1, mask);
    Add(evenPrediction, evenP0, evenP1, mask);
    Add(oddPrediction, oddP0, oddP1, mask);
    ReduceSum(evenPrediction, evenPrediction, mask);
    ReduceSum(oddPrediction, oddPrediction, mask);
    Store<Out, true>(output + row, evenPrediction, mask);
    Store<Out, true>(output + row + 1, oddPrediction, mask);

    if constexpr (vFirst) {
        Store(destination + row * DIM, even0, mask);
        Store(destination + row * DIM + LANES, even1, mask);
        Store(destination + (row + 1) * DIM, odd0, mask);
        Store(destination + (row + 1) * DIM + LANES, odd1, mask);
    } else if constexpr (std::is_same<State, float>::value) {
        Scatter(destination + row, even0, columns, mask);
        Scatter(destination + row + LANES * DIM, even1, columns, mask);
        Scatter(destination + row + 1, odd0, columns, mask);
        Scatter(destination + row + 1 + LANES * DIM, odd1, columns, mask);
    } else {
        RegTensor<uint32_t> packed0, packed1;
        StorePairRow(destination + row, packed0, even0, pairColumns, false, mask);
        StorePairRow(destination + row + LANES * DIM, packed1, even1, pairColumns, false, mask);
        StorePairRow(destination + row, packed0, odd0, pairColumns, true, mask);
        StorePairRow(destination + row + LANES * DIM, packed1, odd1, pairColumns, true, mask);
    }
}

// Pack four valid reduction lanes, append the block, then rotate by four lanes.
// Sixteen appends produce one naturally ordered 64-element output half.
__simd_callee__ inline void CollectOutputBlock(
    RegTensor<float> &output, RegTensor<float> &sum0, RegTensor<float> &sum1,
    RegTensor<float> &sum2, RegTensor<float> &sum3,
    RegTensor<uint32_t> &rotation, MaskReg &firstFour)
{
    Interleave(sum0, sum2, sum0, sum2);
    Interleave(sum1, sum3, sum1, sum3);
    Interleave(sum0, sum1, sum0, sum1);
    Move(output, sum0, firstFour);
    Gather(output, output, rotation);
}

// Keep four complete mathematical V rows live. State results reach UB per block;
// FP32 outputs stay in a token-wide register accumulator until all rows finish.
template <class State, class Value>
__simd_callee__ inline void UpdateRows4(
    __ubuf__ State *destination, __ubuf__ State *source,
    __ubuf__ Value *value, RegTensor<float> &output,
    RegTensor<uint32_t> &outputRotation, MaskReg &firstFour, uint16_t row,
    RegTensor<float> &q0, RegTensor<float> &q1,
    RegTensor<float> &k0, RegTensor<float> &k1,
    RegTensor<float> &alpha0, RegTensor<float> &alpha1,
    RegTensor<float> &strength, MaskReg &mask)
{
    RegTensor<float> s00, s01, s10, s11, s20, s21, s30, s31;
    Load(s00, source + row * DIM, mask);
    Load(s01, source + row * DIM + LANES, mask);
    Load(s10, source + (row + 1) * DIM, mask);
    Load(s11, source + (row + 1) * DIM + LANES, mask);
    Load(s20, source + (row + 2) * DIM, mask);
    Load(s21, source + (row + 2) * DIM + LANES, mask);
    Load(s30, source + (row + 3) * DIM, mask);
    Load(s31, source + (row + 3) * DIM + LANES, mask);

    Mul(s00, s00, alpha0, mask); Mul(s10, s10, alpha0, mask);
    Mul(s20, s20, alpha0, mask); Mul(s30, s30, alpha0, mask);
    Mul(s01, s01, alpha1, mask); Mul(s11, s11, alpha1, mask);
    Mul(s21, s21, alpha1, mask); Mul(s31, s31, alpha1, mask);
    RegTensor<float> p00, p01, p10, p11, p20, p21, p30, p31;
    Mul(p00, s00, k0, mask); Mul(p10, s10, k0, mask);
    Mul(p20, s20, k0, mask); Mul(p30, s30, k0, mask);
    Mul(p01, s01, k1, mask); Mul(p11, s11, k1, mask);
    Mul(p21, s21, k1, mask); Mul(p31, s31, k1, mask);
    RegTensor<float> prediction0, prediction1, prediction2, prediction3;
    RegTensor<float> delta0, delta1, delta2, delta3;
    Sum128(prediction0, p00, p01, mask);
    Sum128(prediction1, p10, p11, mask);
    Sum128(prediction2, p20, p21, mask);
    Sum128(prediction3, p30, p31, mask);
    Broadcast(delta0, value + row, mask);
    Broadcast(delta1, value + row + 1, mask);
    Broadcast(delta2, value + row + 2, mask);
    Broadcast(delta3, value + row + 3, mask);
    Sub(delta0, delta0, prediction0, mask); Sub(delta1, delta1, prediction1, mask);
    Sub(delta2, delta2, prediction2, mask); Sub(delta3, delta3, prediction3, mask);
    Mul(delta0, delta0, strength, mask); Mul(delta1, delta1, strength, mask);
    Mul(delta2, delta2, strength, mask); Mul(delta3, delta3, strength, mask);

    // Reuse product registers after prediction instead of allocating a State shadow.
    Mul(p00, delta0, k0, mask); Mul(p10, delta1, k0, mask);
    Mul(p20, delta2, k0, mask); Mul(p30, delta3, k0, mask);
    Add(s00, s00, p00, mask); Add(s10, s10, p10, mask);
    Add(s20, s20, p20, mask); Add(s30, s30, p30, mask);
    Mul(p01, delta0, k1, mask); Mul(p11, delta1, k1, mask);
    Mul(p21, delta2, k1, mask); Mul(p31, delta3, k1, mask);
    Add(s01, s01, p01, mask); Add(s11, s11, p11, mask);
    Add(s21, s21, p21, mask); Add(s31, s31, p31, mask);

    // All four outputs consume FP32 State before any State row is rounded to BF16.
    Mul(p00, s00, q0, mask); Mul(p10, s10, q0, mask);
    Mul(p20, s20, q0, mask); Mul(p30, s30, q0, mask);
    Mul(p01, s01, q1, mask); Mul(p11, s11, q1, mask);
    Mul(p21, s21, q1, mask); Mul(p31, s31, q1, mask);
    Add(prediction0, p00, p01, mask); Add(prediction1, p10, p11, mask);
    Add(prediction2, p20, p21, mask); Add(prediction3, p30, p31, mask);
    ReduceSum(prediction0, prediction0, mask); ReduceSum(prediction1, prediction1, mask);
    ReduceSum(prediction2, prediction2, mask); ReduceSum(prediction3, prediction3, mask);
    CollectOutputBlock(output, prediction0, prediction1, prediction2, prediction3,
                       outputRotation, firstFour);

    Store(destination + row * DIM, s00, mask);
    Store(destination + row * DIM + LANES, s01, mask);
    Store(destination + (row + 1) * DIM, s10, mask);
    Store(destination + (row + 1) * DIM + LANES, s11, mask);
    Store(destination + (row + 2) * DIM, s20, mask);
    Store(destination + (row + 2) * DIM + LANES, s21, mask);
    Store(destination + (row + 3) * DIM, s30, mask);
    Store(destination + (row + 3) * DIM + LANES, s31, mask);
}

__simd_callee__ inline void Normalize(RegTensor<float> &lo, RegTensor<float> &hi, MaskReg &mask)
{
    RegTensor<float> a, b, norm;
    Mul(a, lo, lo, mask);
    Mul(b, hi, hi, mask);
    Sum128(norm, a, b, mask);
    Adds(norm, norm, 1.e-6f, mask);
    Sqrt(norm, norm, mask);
    Div(lo, lo, norm, mask);
    Div(hi, hi, norm, mask);
}

__simd_callee__ inline void Sigmoid(RegTensor<float> &dst, MaskReg &mask)
{
    RegTensor<float> magnitude, numerator, denominator, one, negative;
    MaskReg positive;
    Duplicate(one, 1.0f);
    Abs(magnitude, dst, mask);
    Muls(magnitude, magnitude, -1.0f, mask);
    Exp(magnitude, magnitude, mask);
    Adds(denominator, magnitude, 1.0f, mask);
    Div(numerator, one, denominator, mask);
    Div(negative, magnitude, denominator, mask);
    CompareScalar<float, CMPMODE::GE>(positive, dst, 0.0f, mask);
    Select(dst, numerator, negative, positive);
}

__simd_callee__ inline void Decay(RegTensor<float> &gate, RegTensor<float> &a,
                                 bool rawGate, bool safeGate, float lowerBound, MaskReg &mask)
{
    if (rawGate) {
        if (safeGate) {
            Mul(gate, gate, a, mask);
            Sigmoid(gate, mask);
            Muls(gate, gate, lowerBound, mask);
        } else {
            RegTensor<float> tail, positive;
            Abs(tail, gate, mask);
            Muls(tail, tail, -1.0f, mask);
            Exp(tail, tail, mask);
            Adds(tail, tail, 1.0f, mask);
            Ln(tail, tail, mask);
            Maxs(positive, gate, 0.0f, mask);
            Add(gate, positive, tail, mask);
            Mul(gate, gate, a, mask);
            Muls(gate, gate, -1.0f, mask);
        }
    }
    Exp(gate, gate, mask);
}

} // namespace RegBase
} // namespace RecurrentKda
#endif
