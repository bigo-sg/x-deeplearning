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

/*
 * Copyright 1999-2017 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef XDL_CORE_OPS_UNIQUE_OP_H_
#define XDL_CORE_OPS_UNIQUE_OP_H_

#include "xdl/core/framework/cpu_device.h"

namespace xdl {

class GpuDevice;
namespace functor {

template <typename Device, typename T, typename I>
struct UniqueFunctor;

template <typename T, typename I>
struct UniqueFunctor<CpuDevice, T, I> {
  void operator()(CpuDevice* d,
                  const Tensor& in,
                  const Tensor& segment,
                  Tensor* out,
                  Tensor& out_index,
                  Tensor* out_sindex,
                  Tensor* out_ssegment);
};

template <typename T, typename I>
struct UniqueFunctor<GpuDevice, T, I> {
  void operator()(GpuDevice* d, const Tensor& in, Tensor* out, Tensor& out_index);
};

}  // namespace functor

template <typename T, typename I>
class UniqueCpuOp : public OpKernel {
 public:
  Status Compute(OpKernelContext* ctx) override;
};

}  // namespace xdl

#endif  // XDL_CORE_OPS_UNIQUE_OP_H_
