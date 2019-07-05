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
#include <memory>

#include "ps-plus/server/udf/simple_udf.h"
#include "ps-plus/server/slice.h"
#include "ps-plus/common/hashmap.h"
#include "ps-plus/server/streaming_model_utils.h"
#include "ps-plus/common/string_utils.h"
#include "hash_filter_utils.h"

namespace ps {
namespace server {
namespace udf {

#define OPS_SINGLE_ARGS(...) __VA_ARGS__

#define OPS_PROCESS(STMT) \
  do {                                       \
    OP_PROCESS(i, >=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, >=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(i, <=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, <=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(i, >, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(d, >, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(i, <, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(d, <, OPS_SINGLE_ARGS(STMT))   \
    OP_PROCESS(i, ==, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, ==, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(i, !=, OPS_SINGLE_ARGS(STMT))  \
    OP_PROCESS(d, !=, OPS_SINGLE_ARGS(STMT))  \
    return Status::ArgumentError("HashLogisticExpFilter cond Error"); \
  } while (0)

#define OP_PROCESS(ARG, OP, STMT)                                               \
  {                                                                             \
    static const std::string ARGOP = #ARG #OP;                                  \
    if (ARGOP.size() <= cond.size() && cond.substr(0, ARGOP.size()) == ARGOP) { \
      auto TESTFN = [=](decltype(ARG) x) {                                      \
        return ARG OP x;                                                        \
      };                                                                        \
      SLOT = cond.substr(ARGOP.size());                                         \
      FUN = "";                                                                 \
      std::vector<std::string> items = ps::StringUtils::split(SLOT, "#");       \
      if (items.size() >= 2 ) {                                                 \
        FUN = items[0];                                                         \
        SLOT = items[1];                                                        \
      }                                                                         \
                                                                                \
      STMT                                                                      \
      break;                                                                    \
    }                                                                           \
  }


class HashLogisticExpFilter : public SimpleUdf<std::string, double, int64_t> {
 public:
  virtual Status SimpleRun(UdfContext* ctx, const std::string& conds, const double& pd, const int64_t& pi) const {
    double d = pd;
    int64_t i = pi;
    Variable* variable = GetVariable(ctx);
    if (variable == nullptr) {
      return Status::ArgumentError("HashLogisticExpFilter: Variable should not be empty");
    }

    if (variable->GetData()->Shape().IsScalar()) {
      return Status::ArgumentError("HashLogisticExpFilter: Variable should not be Scalar");
    }

    if (conds.find("||") != std::string::npos) {
      return Status::ArgumentError("HashLogisticExpFilter: Cond only support && logical connectives ");
    }

    WrapperData<HashMap>* hashmap = dynamic_cast<WrapperData<HashMap>*>(variable->GetSlicer());
    if (hashmap == nullptr) {
      return Status::ArgumentError("HashLogisticExpFilter: Variable Should be a Hash Variable");
    }

    HashMap::HashMapStruct map;
    if (hashmap->Internal().GetHashKeys(&map) != 0) {
      return Status::Unknown("HashLogisticExpFilter Get Hash Keys Error");
    }

    //printf("HashLogisticExpFilter d: %f i: %ld conds: %s\n", d, i, conds.c_str());

    std::vector<std::string> conds_vec = ps::StringUtils::split(conds, "&&");
    if (!conds_vec.size() || conds_vec.size() > 2) {
      return Status::ArgumentError("HashLogisticExpFilter: Conds num must be 1 or 2");
    }

    Tensor* tensor;
    std::shared_ptr<Tensor> output;
    std::string SLOT, FUN;
    std::shared_ptr<std::vector<HashMapItem>> indexes= std::make_shared<std::vector<HashMapItem>>(map.items);
    std::shared_ptr<std::vector<HashMapItem>> res_indexes;
    
    for (auto& cond : conds_vec) {
      //printf("HashLogisticExpFilter cond: %s\n", cond.c_str());
      OPS_PROCESS(
          //printf("HashLogisticExpFilter ARGOP: %s SLOT: %s FUN: %s \n", ARGOP.c_str(), SLOT.c_str(), FUN.c_str());
          if (SLOT == "_") {
            tensor = variable->GetData();
          } else {
            PS_CHECK_STATUS(variable->GetExistSlot(SLOT, &tensor));
          }

          if (!FUN.empty()) {
            if (FUN == "L2") {
              l2_norm_op(indexes, tensor, output);
              tensor = output.get();
            } else {
              return Status::ArgumentError("HashLogisticExpFilter: Invalid FUN " + FUN);
            }
          }

          auto s = tensor->Shape().Dims();
          if (s.size() != 1) {
            return Status::ArgumentError("HashLogisticExpFilter: Slot Shape Error! Must be 1 dim");
          }

          res_indexes.reset(new std::vector<HashMapItem>());
          CASES(tensor->Type(), {
              T* data = tensor->Raw<T>();
              for (auto iter = indexes->begin(); iter != indexes->end(); ++iter) {
                if (TESTFN(data[iter->id])) {
                  res_indexes->push_back(*iter);
                }
              }
          });
         indexes = res_indexes;
      );
    }

    std::vector<int64_t> keys;
    for (auto& item : (*indexes)) {
      keys.push_back(item.x);
      keys.push_back(item.y);
    }

    hashmap->Internal().Del(&(keys[0]), keys.size() / 2, 2);

    if (!ctx->GetStreamingModelArgs()->streaming_hash_model_addr.empty()) {
      PS_CHECK_STATUS(StreamingModelUtils::DelHash(ctx->GetVariableName(), keys));
    }

    //LOG(INFO) << "Hash Filter for " << ctx->GetVariableName() << " origin=" << 
      //map.items.size() << ", clear=" << keys.size() / 2;
    printf("HashLogisticExpFilter for %s origin= %ld, clear= %ld\n", ctx->GetVariableName().c_str(), map.items.size(), keys.size() / 2);

    return Status::Ok();
  }
};

SIMPLE_UDF_REGISTER(HashLogisticExpFilter, HashLogisticExpFilter);

}
}
}
