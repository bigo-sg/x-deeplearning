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

#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/initializer/constant_initializer.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/string_utils.h"

namespace ps {
namespace server {
namespace udf {

class IntegratedIntegerLogger : public SimpleUdf<Slices, std::string, Tensor, Tensor*> {
 public:
  virtual Status SimpleRun(
      UdfContext* ctx,
      const Slices& slices,
      const std::string& slot_names,
      const Tensor& vals,
      Tensor* result) const {

    std::vector<std::string> slots_vec = ps::StringUtils::split(slot_names, "#");
    slices.variable->SetStatsVec(slots_vec);

    std::string slot_name;
    size_t slice;
    Tensor* t;
    float* dst_ptr;
    size_t slots_num = slots_vec.size();
   
    *result = Tensor(DataType::kFloat, TensorShape({slices.slice_id.size(), slots_num}), 
                    new initializer::ConstantInitializer(0));
    float* rst_ptr = result->Raw<float>();

    CASES(vals.Type(), for (size_t j = 0; j < slots_num; ++j) {
      slot_name = slots_vec[j];
      t = slices.variable->GetVariableLikeSlot(slot_name, DataType::kFloat, TensorShape(), 
                                               []{ return new initializer::ConstantInitializer(0); });
      T* vals_raw = vals.Raw<T>();
      dst_ptr = t->Raw<float>();
      for (size_t i = 0; i < slices.slice_id.size(); i++) {
          slice = slices.slice_id[i];
          if ((int64_t)slice == ps::HashMap::NOT_ADD_ID) {
            continue;
          }   

          dst_ptr[slice] += *(vals_raw + i * slots_num + j);       
          *(rst_ptr + i * slots_num + j) = dst_ptr[slice];
      }
    });

    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(IntegratedIntegerLogger, IntegratedIntegerLogger);

}
}
}

