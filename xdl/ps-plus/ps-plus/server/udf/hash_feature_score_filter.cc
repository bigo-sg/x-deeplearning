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
#include <Eigen/Dense>

#include "ps-plus/server/slice.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/server/streaming_model_utils.h"
#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/common/initializer/constant_initializer.h"

namespace ps {
namespace server {
namespace udf {

class HashFeatureScoreFilter : public SimpleUdf<float, float, float, float, float ,int64_t> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const float& decay_rate,
                           const float& nonclk_weight, const float& clk_weight,
                           const float& train_threshold,
                           const float& export_threshold,
                           const int64_t& cur_step) const {
    Variable* variable = GetVariable(ctx);
    if (variable == nullptr) {
      return Status::ArgumentError("HashFeatureScoreFilter: Variable should not be empty");
    }

    if (variable->GetData()->Shape().IsScalar()) {
      return Status::ArgumentError("HashFeatureScoreFilter: Variable should not be Scalar");
    }

    WrapperData<HashMap>* hashmap = dynamic_cast<WrapperData<HashMap>*>(variable->GetSlicer());
    if (hashmap == nullptr) {
      return Status::ArgumentError("HashFeatureScoreFilter: Variable Should be a Hash Variable");
    }

    HashMap::HashMapStruct map;
    if (hashmap->Internal().GetHashKeys(&map) != 0) {
      return Status::Unknown("HashFeatureScoreFilter Get Hash Keys Error");
    }

    auto& items = map.items;
    if (!items.size()) {
      return Status::Ok();
    }

    variable->SetFeaExportThreshold(export_threshold);

    //1. decay fea stats
    auto stats_vec = variable->GetStatsVec();
    LOG_ASSERT(stats_vec.size()) << "fea stats num eq 0.";

    Tensor* tensor = nullptr;
    PS_CHECK_STATUS(variable->GetExistSlot("fea_stats", &tensor));

    size_t stats_num = stats_vec.size();
    auto& shape = tensor->Shape();

    LOG_ASSERT(shape.Size() == 2) << "fea_stats should have 2 dims.";
    LOG_ASSERT(shape[1] == stats_num) << "fea_stats dims 1 doesn't eq fea stats num.";

    CASES(tensor->Type(), {
        T* pstats = tensor->Raw<T>();
        T* dst;
        for (size_t i = 0; i < items.size(); ++i) {
          dst = pstats + items[i].id * stats_num;
          for (size_t j = 0; j < stats_num; ++j) {
            *(dst + j) *= decay_rate;
          }
        }
    });

    //2. build show and click vector
    Eigen::VectorXd show_vector(items.size());
    Eigen::VectorXd clk_vector(items.size());

    auto show_iter = std::find(stats_vec.begin(), stats_vec.end(), "show");
    auto click_iter = std::find(stats_vec.begin(), stats_vec.end(), "click");
    assert(show_iter != stats_vec.end() && click_iter != stats_vec.end());

    size_t show_idx = show_iter - stats_vec.begin();
    size_t click_idx = click_iter - stats_vec.begin();

    CASES(tensor->Type(), {
      T* pstats = tensor->Raw<T>();
      T* dst;
      for (size_t i = 0; i < items.size(); ++i) {
        auto& item = items[i];
        dst = pstats + item.id * stats_num;
        show_vector(i) = *(dst + show_idx);
        clk_vector(i) = *(dst + click_idx);
      }
    });

    //3. compute fea score
    auto score = (show_vector - clk_vector) * nonclk_weight + clk_vector * clk_weight;
    std::string var_name = ctx->GetVariableName();
    printf("HashFeatureScoreFilter for %s fea score min %f max %f, cur_step:%ld\n",
           var_name.c_str(), score.minCoeff(), score.maxCoeff(), cur_step);

    //4. select keys and store fea score
    std::vector<int64_t> keys;
    Tensor* fea_scores = variable->GetVariableLikeSlot("fea_score", DataType::kFloat, TensorShape(),
                                                       []{ return new initializer::ConstantInitializer(0); });
    float* pscores = fea_scores->Raw<float>();
    for (size_t i = 0; i < items.size(); ++i) {
      pscores[items[i].id] = score(i);
      if (score(i) < train_threshold) {
        keys.push_back(items[i].x);
        keys.push_back(items[i].y);
      }
    }

    //5. delete
    hashmap->Internal().Del(&(keys[0]), keys.size() / 2, 2);

    if (!ctx->GetStreamingModelArgs()->streaming_hash_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelUtils::DelHash(ctx->GetVariableName(), keys));
    }

    printf("HashFeatureScoreFilter for %s origin= %ld, clear= %ld\n",
           var_name.c_str(), items.size(), keys.size() / 2);

    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(HashFeatureScoreFilter, HashFeatureScoreFilter);

}
}
}

