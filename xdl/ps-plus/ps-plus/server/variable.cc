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

#include "ps-plus/server/variable.h"

#include <glog/logging.h>

namespace ps {
namespace server {

Tensor* Variable::GetSlot(const std::string& name, const std::function<Slot()>& slot_creator) {
  {
    QRWLocker lock(slots_lock_, QRWLocker::kSimpleRead);
    auto iter = slots_.find(name);
    if (iter != slots_.end()) {
      return iter->second.tensor.get();
    }
  }
  {
    QRWLocker lock(slots_lock_, QRWLocker::kWrite);
    auto iter = slots_.find(name);
    if (iter != slots_.end()) {
      return iter->second.tensor.get();
    }
    slots_[name] = slot_creator();
    return slots_[name].tensor.get();
  }
}

void Variable::SetStatsVec(const std::vector<std::string>& stats) {
  QRWLocker lock(stats_lock_, QRWLocker::kWrite);
  LOG_ASSERT(stats.size()) << "fea stats num eq 0.";

  if (!stats_vec_.size()) {
    stats_vec_.insert(stats_vec_.begin(), stats.begin(), stats.end());
    return;
  }

  LOG_ASSERT(stats.size() == stats_vec_.size()) << "fea stats num has changed.";
  for (size_t i = 0; i < stats_vec_.size(); ++i) {
    LOG_ASSERT(stats[i] == stats_vec_[i]) << "fea stats item " << i << " has changed.";
  }
}

Variable::Slot Variable::VariableLikeSlot(DataType type, const TensorShape& shape, Initializer* initializer) {
  return Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(type, shape, initializer)), .joiner = kVariableLike};
}

Variable::Slot Variable::AnyOneSlot(DataType type, const TensorShape& shape, Initializer* initializer) {
  return Slot{.tensor = std::unique_ptr<Tensor>(new Tensor(type, shape, initializer)), .joiner = kAnyOne};
}

Tensor* Variable::GetAnyOneSlot(const std::string& name, DataType type, const TensorShape& shape, const std::function<Initializer*()>& initializer_creator) {
  return GetSlot(name, [&]{ return AnyOneSlot(type, shape, initializer_creator()); });
}

Tensor* Variable::GetVariableLikeSlot(const std::string& name, DataType type, const std::function<Initializer*()>& initializer_creator) {
  return GetSlot(name, [&]{ return VariableLikeSlot(type, data_->Shape(), initializer_creator()); });
}

Tensor* Variable::GetVariableLikeSlot(const std::string& name, DataType type, const TensorShape& inner_shape, const std::function<Initializer*()>& initializer_creator) {
  return GetSlot(name, [&]{
    std::vector<size_t> dims;
    if (data_->Shape().Size() > 0) {
      dims.push_back(data_->Shape()[0]);
    }
    for (auto dim : inner_shape.Dims()) {
      dims.push_back(dim);
    }
    return VariableLikeSlot(type, TensorShape(dims), initializer_creator());
  });
}

Status Variable::GetExistSlot(const std::string& name, Tensor** result) {
  QRWLocker lock(slots_lock_, QRWLocker::kSimpleRead);
  auto iter = slots_.find(name);
  if (iter == slots_.end()) {
    return Status::NotFound("Slot Not Found " + name);
  }
  *result = iter->second.tensor.get();
  return Status::Ok();
}

Status Variable::ReShapeId(size_t id) {
  TensorShape shape = data_->Shape();
  if (shape.Size() == 0) {
    return Status::ArgumentError("Scalar Not Support ReShapeId");
  }
  shape.Set(0, id);
  data_->ReShape(shape);
  for (auto& slot : slots_) {
    if (slot.second.joiner == kVariableLike) {
      Tensor* tensor = slot.second.tensor.get();
      TensorShape shape = tensor->Shape();
      if (shape.Size() == 0) {
        return Status::ArgumentError("Scalar Not Support ReShapeId");
      }
      shape.Set(0, id);
      tensor->ReShape(shape);
    }
  }
  return Status::Ok();
}

void Variable::ClearIds(const std::vector<size_t>& ids) {
  for (size_t id : ids) {
    data_->ClearId(id);
  }
  for (auto& slot : slots_) {
    if (slot.second.joiner == kVariableLike) {
      for (size_t id : ids) {
        slot.second.tensor->ClearId(id);
      }
    }
  }
}

}
}

