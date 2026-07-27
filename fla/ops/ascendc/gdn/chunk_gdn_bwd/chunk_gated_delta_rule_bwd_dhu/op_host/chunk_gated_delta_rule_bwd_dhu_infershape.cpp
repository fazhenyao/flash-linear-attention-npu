#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus Infer(gert::InferShapeContext *c) {
    auto q = c->GetInputShape(0); auto dv = c->GetInputShape(4); auto dh = c->GetOutputShape(0); auto dh0 = c->GetOutputShape(1); auto out = c->GetOutputShape(2);
    if (!q || !dv || !dh || !dh0 || !out) return ge::GRAPH_FAILED;
    auto a = c->GetAttrs(); int64_t chunk = a && a->GetAttrPointer<int64_t>(1) ? *a->GetAttrPointer<int64_t>(1) : 64;
    int64_t n = (q->GetDim(2) + chunk - 1) / chunk;
    auto indices = c->GetOptionalInputShape(10);
    if (indices && indices->GetDimNum() == 1) n = indices->GetDim(0) / 2;
    dh->SetDimNum(5); dh->SetDim(0,q->GetDim(0)); dh->SetDim(1,n); dh->SetDim(2,dv->GetDim(1)); dh->SetDim(3,q->GetDim(3)); dh->SetDim(4,dv->GetDim(3));
    dh0->SetDimNum(4); dh0->SetDim(0,q->GetDim(0)); dh0->SetDim(1,dv->GetDim(1)); dh0->SetDim(2,q->GetDim(3)); dh0->SetDim(3,dv->GetDim(3)); *out = *dv;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleBwdDhu).InferShape(Infer);
}
