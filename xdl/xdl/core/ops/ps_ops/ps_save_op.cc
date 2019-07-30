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

class PsSaveOp : public xdl::OpKernelAsync {
public:
  Status Init(OpKernelConstruction* ctx) override {
    return Status::Ok();
  }

  void Compute(OpKernelContext* ctx, Callback done) override {
    ps::client::BaseClient* client;
    XDL_CHECK_STATUS_ASYNC(GetClient(&client), done);
    Tensor t_ckpt_version;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(0, &t_ckpt_version), done);
    std::string ckpt_version = t_ckpt_version.Scalar<std::string>();
    
    Tensor t_save_mode;
    XDL_CHECK_STATUS_ASYNC(ctx->GetInput(1, &t_save_mode), done);
    uint64_t save_mode = t_save_mode.Scalar<uint64_t>();

    auto cb = [ctx, done](const ps::Status& st) {
      XDL_CHECK_STATUS_ASYNC(PS2XDL::ConvertStatus(st), done);
      done(Status::Ok());
    };

    client->Save(ckpt_version, save_mode, cb);
  }
};

/**
 * save_mode: bit mark. 0x01 save binary embedding
 * click_show_threshold: use to filt out embedding ids by click show stat.
 *     click_show_threshold = -1: defuatl value. not filt
*/
XDL_DEFINE_OP(PsSaveOp)
  .Input("ckpt_version", DataType::kInt8)
  .Input("save_mode", DataType::kInt64);

XDL_REGISTER_KERNEL(PsSaveOp, PsSaveOp).Device("CPU");

} // namespace xdl


