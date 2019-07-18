/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsSparsePullWithFeaStatsOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("save_ratio", &save_ratio_));
    XDL_CHECK_STATUS(ctx->GetAttr("stats_desp", &stats_desp_));
    XDL_CHECK_STATUS(ctx->GetAttr("pattern", &pattern_));
    XDL_CHECK_STATUS(XdlGetVarType(ctx, &var_type_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);

    Tensor ids;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &ids), done);

    ps::Tensor convert_ids;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(ids, &convert_ids),
        done);

    Tensor stats_delta;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &stats_delta), done);

    ps::Tensor convert_delta;
    XDL_CHECK_STATUS_ASYNC(
        XDL2PS::ConvertTensor(stats_delta, &convert_delta),
        done);
    
    Tensor t_i;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(2, &t_i), done);
    int64_t i = t_i.Scalar<int64_t>();

    ps::Tensor* result = new ps::Tensor;
    ps::Tensor* fea_stats = new ps::Tensor;

    auto cb = [result, fea_stats, ctx, done](const ps::Status& st) {
      std::unique_ptr<ps::Tensor> result_deleter(result);
      std::unique_ptr<ps::Tensor> fea_stats_deleter(fea_stats);

      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      XDL_CHECK_STATUS_ASYNC(
          PS2XDL::ConvertTensorWithCtx(*result, ctx, 0),
          done);
      
      XDL_CHECK_STATUS_ASYNC(
          PS2XDL::ConvertTensorWithCtx(*fea_stats, ctx, 1),
          done);

      done(Status::Ok());
    };

    switch(var_type_) {
    case VarType::kHash:
      client->HashPullWithFeaStats(var_name_, convert_ids, convert_delta, i,
                                   save_ratio_, stats_desp_, pattern_, result, fea_stats, cb);
      break;      
    default:
      XDL_CHECK_COND_ASYNC(
          false, 
          Status::ArgumentError("unsupported vartype"),
          done);
    }
  }

 private:
  std::string var_name_;
  VarType var_type_;
  float save_ratio_;
  std::string stats_desp_;
  std::string pattern_;
};

XDL_DEFINE_OP(PsSparsePullWithFeaStatsOp)
  .Input("ids", "dtype")
  .Input("delta", "otype")
  .Input("i", DataType::kInt64)
  .Output("output", "otype")
  .Output("fea_stats", "otype")
  .Attr("var_name", AttrValue::kString)
  .Attr("stats_desp", AttrValue::kString)
  .Attr("var_type", AttrValue::kString)
  .Attr("save_ratio", AttrValue::kFloat)
  .Attr("pattern", AttrValue::kString)
  .Attr("dtype", AttrValue::kDataType)
  .Attr("otype", AttrValue::kDataType);

#define REGISTER_KERNEL(ITYPE, OTYPE) \
  XDL_REGISTER_KERNEL(PsSparsePullWithFeaStatsOp, PsSparsePullWithFeaStatsOp)  \
  .Device("CPU")                                       \
  .AttrDataType<ITYPE>("dtype")                        \
  .AttrDataType<OTYPE>("otype");                       \

REGISTER_KERNEL(int32_t, int8_t);
REGISTER_KERNEL(int32_t, int16_t);
REGISTER_KERNEL(int32_t, int32_t);
REGISTER_KERNEL(int32_t, int64_t);
REGISTER_KERNEL(int32_t, float);
REGISTER_KERNEL(int64_t, int8_t);
REGISTER_KERNEL(int64_t, int16_t);
REGISTER_KERNEL(int64_t, int32_t);
REGISTER_KERNEL(int64_t, int64_t);
REGISTER_KERNEL(int64_t, float);

} // namespace xdl


