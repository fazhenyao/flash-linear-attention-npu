#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "chunk_gated_delta_rule_bwd_dhu.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleBwdDhu);

namespace {
void SetOriginalShapeFromView(const aclTensor *tensor)
{
    if (tensor != nullptr) {
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
}
} // namespace

const std::array<const aclTensor*,3> ChunkGatedDeltaRuleBwdDhu(const aclTensor* q,const aclTensor* k,const aclTensor* w,const aclTensor* dO,const aclTensor* dv,const aclTensor* g,const aclTensor* gk,const aclTensor* h0,const aclTensor* dht,const aclIntArray* cu,const aclIntArray* indices,double scale,int64_t chunk,const aclTensor* dh,const aclTensor* dh0,const aclTensor* dv2,aclOpExecutor* executor) {
    SetOriginalShapeFromView(q);
    SetOriginalShapeFromView(k);
    SetOriginalShapeFromView(w);
    SetOriginalShapeFromView(dO);
    SetOriginalShapeFromView(dv);
    SetOriginalShapeFromView(g);
    SetOriginalShapeFromView(gk);
    SetOriginalShapeFromView(h0);
    SetOriginalShapeFromView(dht);
    SetOriginalShapeFromView(dh);
    SetOriginalShapeFromView(dh0);
    SetOriginalShapeFromView(dv2);

    const aclTensor *cuT=nullptr,*idxT=nullptr; if(cu){cuT=executor->ConvertToTensor(cu,DataType::DT_INT64);} if(indices){idxT=executor->ConvertToTensor(indices,DataType::DT_INT64);} const aclTensor *dh0k=dh0; if(!dh0k){op::Shape s; s.AppendDim(0); dh0k=executor->AllocTensor(s,DataType::DT_FLOAT,Format::FORMAT_ND);} auto r=ADD_TO_LAUNCHER_LIST_AICORE(ChunkGatedDeltaRuleBwdDhu,OP_INPUT(q,k,w,dO,dv,g,gk,h0,dht,cuT,idxT),OP_OUTPUT(dh,dh0k,dv2),OP_ATTR(scale,chunk)); if(r!=ACLNN_SUCCESS)return {nullptr,nullptr,nullptr}; return {dh,dh0k,dv2}; }
}
