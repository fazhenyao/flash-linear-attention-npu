#ifndef ACLNN_CHUNK_DHU_H
#define ACLNN_CHUNK_DHU_H
#include "aclnn/acl_meta.h"
extern "C" {
aclnnStatus aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclIntArray*,const aclIntArray*,double,int64_t,const aclTensor*,const aclTensor*,const aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(void*,uint64_t,aclOpExecutor*,aclrtStream);
}
#endif
