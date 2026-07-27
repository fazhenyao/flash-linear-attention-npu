#include "register/op_def_registry.h"
namespace ops {
class ChunkGatedDeltaRuleBwdDhu : public OpDef {
public:
    explicit ChunkGatedDeltaRuleBwdDhu(const char *name) : OpDef(name) {
        auto add = [this](const char *n, ge::DataType a, ge::DataType b, ge::DataType c, ge::DataType d) {
            this->Input(n).ParamType(REQUIRED).DataType({a,b,c,d}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).AutoContiguous();
        };
        add("q", ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16);
        add("k", ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16);
        add("w", ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16);
        add("d_o", ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16);
        add("dv", ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16);
        auto opt = [this](const char *n, std::initializer_list<ge::DataType> types) {
            this->Input(n).ParamType(OPTIONAL).DataType(types).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).AutoContiguous();
        };
        opt("g", {ge::DT_BF16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT});
        opt("gk", {ge::DT_BF16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT});
        opt("h0", {ge::DT_BF16,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_FLOAT16});
        opt("dht", {ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT});
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL).DataType({ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).AutoContiguous();
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL).DataType({ge::DT_INT64,ge::DT_INT64,ge::DT_INT64,ge::DT_INT64}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).AutoContiguous();
        this->Output("dh").ParamType(REQUIRED).DataType({ge::DT_BF16,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("dh0").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("dv2").ParamType(REQUIRED).DataType({ge::DT_BF16,ge::DT_FLOAT16,ge::DT_BF16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        OpAICoreConfig cfg;
        cfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", cfg);
        this->AICore().AddConfig("ascend910_93", cfg);
        this->AICore().AddConfig("ascend950", cfg);
    }
};
OP_ADD(ChunkGatedDeltaRuleBwdDhu);
}
