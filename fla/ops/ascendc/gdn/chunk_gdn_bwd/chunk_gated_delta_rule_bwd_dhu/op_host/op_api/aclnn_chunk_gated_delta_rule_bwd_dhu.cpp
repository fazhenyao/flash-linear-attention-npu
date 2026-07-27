#include "aclnn_chunk_gated_delta_rule_bwd_dhu.h"
#include "chunk_gated_delta_rule_bwd_dhu.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
using namespace op;
extern "C" aclnnStatus aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(const aclTensor* q,const aclTensor* k,const aclTensor* w,const aclTensor* dO,const aclTensor* dv,const aclTensor* g,const aclTensor* gk,const aclTensor* h0,const aclTensor* dht,const aclIntArray* cu,const aclIntArray* indices,double scale,int64_t chunk,const aclTensor* dh,const aclTensor* dh0,const aclTensor* dv2,uint64_t* ws,aclOpExecutor** out){ auto ex=CREATE_EXECUTOR(); if(ex.get()==nullptr)return ACLNN_ERR_INNER_CREATE_EXECUTOR; auto r=l0op::ChunkGatedDeltaRuleBwdDhu(q,k,w,dO,dv,g,gk,h0,dht,cu,indices,scale,chunk,dh,dh0,dv2,ex.get()); if(!r[0])return ACLNN_ERR_PARAM_INVALID; *ws=ex->GetWorkspaceSize(); ex.ReleaseTo(out); return ACLNN_SUCCESS; }
extern "C" aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream stream){return CommonOpExecutorRun(ws,size,ex,stream);}
