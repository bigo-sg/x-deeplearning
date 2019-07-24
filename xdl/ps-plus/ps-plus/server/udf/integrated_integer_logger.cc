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

#include <glog/logging.h>

#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/string_utils.h"

namespace ps {
namespace server {
namespace udf {

class IntegratedIntegerLogger : public SimpleUdf<Slices, std::string, Tensor, TensorSlices*> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const std::string& slot_names,
      const Tensor& vals,
      TensorSlices* result) const {

    std::vector<std::string> slots_vec = ps::StringUtils::split(slot_names, "#");
    slices.variable->SetStatsVec(slots_vec);
    size_t slots_num = slots_vec.size();

    result->tensor = *(slices.variable->GetVariableLikeSlot("fea_stats", DataType::kFloat, TensorShape({slots_num}),
                                                            []{ return new initializer::ConstantInitializer(0); }));
    LOG_ASSERT(result->tensor.Shape().Size() == 2) << "fea_stats should have 2 dims.";
    LOG_ASSERT(result->tensor.Shape()[1] == slots_num) << "fea_stats dims 1 doesn't eq fea stats num.";

    float* rst_ptr = result->tensor.Raw<float>();
    size_t slice;
    float* dst;
    CASES(vals.Type(), {
      T* vals_raw = vals.Raw<T>();
      T* src;
      for (size_t i = 0; i < slices.slice_id.size(); i++) {
        slice = slices.slice_id[i];
        if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
          continue;
        }

        src = vals_raw + i * slots_num;
        dst = rst_ptr + slice * slots_num;

        for (size_t j = 0; j < slots_num; ++j) {
          *(dst + j) += *(src + j);
        }
      }
    });

    result->slice_size = slots_num;
    result->slice_id = slices.slice_id;
    result->dim_part = slices.dim_part;

    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(IntegratedIntegerLogger, IntegratedIntegerLogger);

}
}
}

