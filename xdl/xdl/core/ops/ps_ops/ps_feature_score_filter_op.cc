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

#include "ps-plus/client/partitioner/broadcast.h"
#include "xdl/core/lib/status.h"
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/ops/ps_ops/define_op.h"
#include "xdl/core/ops/ps_ops/convert_utils.h"
#include "xdl/core/ops/ps_ops/client.h"
#include "xdl/core/ops/ps_ops/var_type.h"

namespace xdl {

class PsFeatureScoreFilterOp : public xdl::OpKernelAsync {
 public:
  Status Init(OpKernelConstruction* ctx) override {
    XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
    XDL_CHECK_STATUS(ctx->GetAttr("decay_rate", &decay_rate_));
    XDL_CHECK_STATUS(ctx->GetAttr("nonclk_weight", &nonclk_weight_));
    XDL_CHECK_STATUS(ctx->GetAttr("clk_weight", &clk_weight_));
    XDL_CHECK_STATUS(ctx->GetAttr("train_threshold", &train_threshold_));
    XDL_CHECK_STATUS(ctx->GetAttr("export_threshold", &export_threshold_));
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);

    Tensor t_i;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_i), done);
    int64_t cur_step = t_i.Scalar<int64_t>();

    std::vector<std::unique_ptr<ps::Data>>* outputs = 
      new std::vector<std::unique_ptr<ps::Data>>;

    auto cb = [ctx, outputs, done](const ps::Status& st) {
      delete outputs;
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    ps::client::UdfData udf("HashFeatureScoreFilter",
                            ps::client::UdfData(0),
                            ps::client::UdfData(1),
                            ps::client::UdfData(2),
                            ps::client::UdfData(3),
                            ps::client::UdfData(4),
                            ps::client::UdfData(5));

    std::vector<ps::client::Partitioner*> spliters{
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast,
      new ps::client::partitioner::Broadcast};

    client->Process(udf, var_name_, client->Args(decay_rate_, nonclk_weight_, clk_weight_, train_threshold_, export_threshold_, cur_step),
                    spliters, {}, outputs, cb);
  }

 private:
  std::string var_name_;
  VarType var_type_;
  float decay_rate_;
  float nonclk_weight_;
  float clk_weight_;
  float train_threshold_;
  float export_threshold_;
};

XDL_DEFINE_OP(PsFeatureScoreFilterOp)
  .Input("cur_step", DataType::kInt64)
  .Attr("var_name", AttrValue::kString)
  .Attr("decay_rate", AttrValue::kFloat)
  .Attr("nonclk_weight", AttrValue::kFloat)
  .Attr("clk_weight", AttrValue::kFloat)
  .Attr("train_threshold", AttrValue::kFloat)
  .Attr("export_threshold", AttrValue::kFloat);
  

XDL_REGISTER_KERNEL(PsFeatureScoreFilterOp, PsFeatureScoreFilterOp).Device("CPU");

} // namespace xdl


