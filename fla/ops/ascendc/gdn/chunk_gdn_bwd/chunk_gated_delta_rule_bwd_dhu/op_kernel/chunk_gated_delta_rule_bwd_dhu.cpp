#include "kernel_operator.h"
#ifndef TORCH_MODE
#include "lib/matmul_intf.h"
#endif
#include "chunk_gated_delta_rule_bwd_dhu_struct.h"
#include "chunk_gated_delta_rule_bwd_dhu_cube.h"
#include "chunk_gated_delta_rule_bwd_dhu_vector.h"

namespace GDN {
template <int DTYPE_KEY>
struct DhuDtype;
template <>
struct DhuDtype<CHUNK_DHU_TPL_BF16> { using type = bfloat16_t; };
template <>
struct DhuDtype<CHUNK_DHU_TPL_FP16> { using type = half; };
template <>
struct DhuDtype<CHUNK_DHU_TPL_FP32> { using type = float; };

template <int STRATEGY, int D_T_Q, int D_T_G, int V_DIM, int CHUNK_SIZE>
__aicore__ inline void RunDhu(GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR dO, GM_ADDR dv, GM_ADDR g,
                              GM_ADDR gk, GM_ADDR dht, GM_ADDR cu, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2,
                              GM_ADDR workspace, const ChunkGatedDeltaRuleBwdDhuTilingData *p)
{
    using T = typename DhuDtype<D_T_Q>::type;
    using GT = typename DhuDtype<D_T_G>::type;
    if ASCEND_IS_AIC {
        DhuCube<T, V_DIM, CHUNK_SIZE, STRATEGY> cube;
        cube.Init(q, k, w, dO, cu, workspace, p);
        cube.Process();
    }
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        DhuVector<T, GT, V_DIM, CHUNK_SIZE, STRATEGY> vec;
        vec.Init(q, dv, g, gk, dht, cu, dh, dh0, dv2, workspace, p, &pipe);
        vec.Process();
    }
}
} // namespace GDN

template <uint64_t STRATEGY, int D_T_Q, int D_T_G, int V_DIM, int CHUNK_SIZE>
__global__ __aicore__ void chunk_gated_delta_rule_bwd_dhu(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR h0, GM_ADDR dht,
    GM_ADDR cu, GM_ADDR indices, GM_ADDR dh, GM_ADDR dh0, GM_ADDR dv2, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)h0;
    (void)indices;
    REGISTER_TILING_DEFAULT(GDN::ChunkGatedDeltaRuleBwdDhuTilingData);
    GET_TILING_DATA_WITH_STRUCT(GDN::ChunkGatedDeltaRuleBwdDhuTilingData, p, tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) return;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GDN::RunDhu<STRATEGY, D_T_Q, D_T_G, V_DIM, CHUNK_SIZE>(
        q, k, w, d_o, dv, g, gk, dht, cu, dh, dh0, dv2, userWorkspace, &p);
}
