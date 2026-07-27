#ifndef CHUNK_DHU_API_H
#define CHUNK_DHU_API_H
#include <array>
#include "aclnn/acl_meta.h"
namespace l0op { const std::array<const aclTensor*,3> ChunkGatedDeltaRuleBwdDhu(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclIntArray*,const aclIntArray*,double,int64_t,const aclTensor*,const aclTensor*,const aclTensor*,aclOpExecutor*); }
#endif
