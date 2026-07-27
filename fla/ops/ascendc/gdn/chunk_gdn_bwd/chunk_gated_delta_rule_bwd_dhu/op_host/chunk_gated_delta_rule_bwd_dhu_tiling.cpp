#include <algorithm>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/chunk_gated_delta_rule_bwd_dhu_struct.h"

namespace optiling {
using namespace GDN;
namespace {
constexpr uint64_t ALIGN_BYTES = 512;
constexpr uint64_t A2_A3_AIC_CORE_COUNT = 20;

struct ChunkGatedDeltaRuleBwdDhuCompileInfo {};

uint64_t AlignUp(uint64_t value)
{
    return (value + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
}

int DtypeKey(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) return CHUNK_DHU_TPL_BF16;
    if (dtype == ge::DT_FLOAT) return CHUNK_DHU_TPL_FP32;
    return CHUNK_DHU_TPL_FP16;
}
} // namespace

static ge::graphStatus Tiling(gert::TilingContext *c)
{
    OP_LOGI(c->GetNodeName(), "ChunkGatedDeltaRuleBwdDhu tiling start");
    auto q = c->GetInputShape(0);
    auto k = c->GetInputShape(1);
    auto w = c->GetInputShape(2);
    auto dO = c->GetInputShape(3);
    auto dv = c->GetInputShape(4);
    auto attrs = c->GetAttrs();
    auto p = c->GetTilingData<GDN::ChunkGatedDeltaRuleBwdDhuTilingData>();
    if (!q || !k || !w || !dO || !dv || !p || !attrs) {
        OP_LOGE(c->GetNodeName(), "missing required tiling context data");
        return ge::GRAPH_FAILED;
    }

    auto scaleAttr = attrs->GetAttrPointer<double>(0);
    auto chunkAttr = attrs->GetAttrPointer<int64_t>(1);
    if (!scaleAttr || !chunkAttr) {
        OP_LOGE(c->GetNodeName(), "missing attrs: scale=%p chunk_size=%p", scaleAttr, chunkAttr);
        return ge::GRAPH_FAILED;
    }

    const auto &qs = q->GetOriginShape();
    const auto &ks = k->GetOriginShape();
    const auto &ws = w->GetOriginShape();
    const auto &dos = dO->GetOriginShape();
    const auto &vs = dv->GetOriginShape();
    if (qs.GetDimNum() != 4 || ks.GetDimNum() != 4 || ws.GetDimNum() != 4 ||
        dos.GetDimNum() != 4 || vs.GetDimNum() != 4) {
        OP_LOGE(c->GetNodeName(), "q/k/w/d_o/dv must all be rank 4");
        return ge::GRAPH_FAILED;
    }
    p->b = qs.GetDim(0);
    p->hk = qs.GetDim(1);
    p->t = qs.GetDim(2);
    p->k = qs.GetDim(3);
    p->hv = vs.GetDim(1);
    p->v = vs.GetDim(3);
    p->chunk = *chunkAttr;
    p->chunks = (p->t + p->chunk - 1) / p->chunk;
    p->seqs = 1;
    p->varlen = 0;
    p->has_g = c->GetOptionalInputShape(5) != nullptr;
    p->has_gk = c->GetOptionalInputShape(6) != nullptr;
    p->has_h0 = c->GetOptionalInputShape(7) != nullptr;
    p->has_dht = c->GetOptionalInputShape(8) != nullptr;
    p->scale = static_cast<float>(*scaleAttr);
    if (c->GetOptionalInputShape(9) && c->GetOptionalInputShape(10)) {
        p->varlen = 1;
        p->seqs = c->GetOptionalInputShape(9)->GetStorageShape().GetDim(0) - 1;
        p->chunks = c->GetOptionalInputShape(10)->GetStorageShape().GetDim(0) / 2;
    }

    if (ks.GetDim(0) != p->b || ks.GetDim(1) != p->hk || ks.GetDim(2) != p->t || ks.GetDim(3) != p->k ||
        ws.GetDim(0) != p->b || ws.GetDim(1) != p->hv || ws.GetDim(2) != p->t ||
        ws.GetDim(3) != p->k || dos.GetDim(0) != p->b || dos.GetDim(1) != p->hv ||
        dos.GetDim(2) != p->t || dos.GetDim(3) != p->v || vs.GetDim(0) != p->b ||
        vs.GetDim(2) != p->t) {
        OP_LOGE(c->GetNodeName(), "required input shapes do not match [B,H,T,D] contracts");
        return ge::GRAPH_FAILED;
    }

    auto gShape = c->GetOptionalInputShape(5);
    auto gkShape = c->GetOptionalInputShape(6);
    auto h0Shape = c->GetOptionalInputShape(7);
    auto dhtShape = c->GetOptionalInputShape(8);
    if (gShape) {
        const auto &s = gShape->GetOriginShape();
        if (s.GetDimNum() != 3 || s.GetDim(0) != p->b || s.GetDim(1) != p->hv || s.GetDim(2) != p->t) {
            OP_LOGE(c->GetNodeName(), "g must be [B,Hv,T]");
            return ge::GRAPH_FAILED;
        }
    }
    if (gkShape) {
        const auto &s = gkShape->GetOriginShape();
        if (s.GetDimNum() != 4 || s.GetDim(0) != p->b || s.GetDim(1) != p->hv ||
            s.GetDim(2) != p->t || s.GetDim(3) != p->k) {
            OP_LOGE(c->GetNodeName(), "gk must be [B,Hv,T,K]");
            return ge::GRAPH_FAILED;
        }
    }
    const uint64_t stateBatch = p->varlen ? p->seqs : p->b;
    auto validStateShape = [&](auto *shape) {
        if (!shape) return true;
        const auto &s = shape->GetOriginShape();
        return s.GetDimNum() == 4 && s.GetDim(0) == stateBatch && s.GetDim(1) == p->hv &&
               s.GetDim(2) == p->k && s.GetDim(3) == p->v;
    };
    if (!validStateShape(h0Shape) || !validStateShape(dhtShape)) {
        OP_LOGE(c->GetNodeName(), "h0/dht must be [state_batch,Hv,K,V]");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(c->GetNodeName(),
        "shape b=%lu hk=%lu t=%lu k=%lu hv=%lu v=%lu chunk=%lu optional g=%lu gk=%lu h0=%lu dht=%lu",
        p->b, p->hk, p->t, p->k, p->hv, p->v, p->chunk, p->has_g, p->has_gk, p->has_h0, p->has_dht);

    if (p->k != 128 || (p->v != 128 && p->v != 256) || (p->chunk != 64 && p->chunk != 128) ||
        p->hk == 0 || p->hv == 0 || p->hv % p->hk != 0) {
        OP_LOGE(c->GetNodeName(), "unsupported shape or chunk size");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(c->GetPlatformInfo());
    uint64_t aicCoreCount = platform.GetCoreNumAic();
    const uint64_t aivCoreCount = platform.GetCoreNumAiv();
    OP_LOGI(c->GetNodeName(), "platform cores aic=%lu aiv=%lu", aicCoreCount, aivCoreCount);
    // Direct aclnn tiling can expose a MIX platform with no AIC count. A2/A3
    // both provide 20 AICs, so retain parallelism instead of emitting blockDim 0.
    if (aicCoreCount == 0) aicCoreCount = A2_A3_AIC_CORE_COUNT;
    // MIX_AIC_1_1 blockDim represents physical MIX groups. Do not shrink it
    // for small shapes: the runtime can otherwise convert it to zero groups.
    p->used_core = static_cast<uint32_t>(aicCoreCount);
    p->state_bytes = AlignUp(p->k * p->v * sizeof(float));
    p->matrix_bytes = AlignUp(std::max(p->chunk, p->k) * p->v * sizeof(float));
    // One low-precision state operand, raw K@state, dv2, q-gated, and two FP32 updates.
    p->slot_bytes = p->state_bytes + 6 * p->matrix_bytes;

    auto qDesc = c->GetInputDesc(0);
    auto gDesc = c->GetInputDesc(5);
    auto gkDesc = c->GetInputDesc(6);
    if (!qDesc) {
        OP_LOGE(c->GetNodeName(), "missing q input descriptor");
        return ge::GRAPH_FAILED;
    }
    const int qType = DtypeKey(qDesc->GetDataType());
    const int gType = (p->has_g && gDesc) ? DtypeKey(gDesc->GetDataType()) :
                      ((p->has_gk && gkDesc) ? DtypeKey(gkDesc->GetDataType()) : qType);
    c->SetTilingKey(GET_TPL_TILING_KEY(
        p->varlen ? CHUNK_DHU_STRATEGY_VARLEN : CHUNK_DHU_STRATEGY_FIXED,
        qType, gType, static_cast<int>(p->v), static_cast<int>(p->chunk)));
    c->SetBlockDim(p->used_core);
    c->SetScheduleMode(1);
    c->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + p->used_core * p->slot_bytes;
    OP_LOGI(c->GetNodeName(), "tiling success key=%lu block_dim=%u workspace=%lu",
        c->GetTilingKey(), p->used_core, c->GetWorkspaceSizes(1)[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleBwdDhu)
    .Tiling(Tiling)
    .TilingParse<ChunkGatedDeltaRuleBwdDhuCompileInfo>(TilingParse);
} // namespace optiling
