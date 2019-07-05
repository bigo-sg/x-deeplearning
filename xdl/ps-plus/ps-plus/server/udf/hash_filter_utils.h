#ifndef XDL_HASH_FILTER_UTILS_H
#define XDL_HASH_FILTER_UTILS_H

#include "ps-plus/common/hashmap.h"
#include "ps-plus/common/tensor.h"

namespace ps {
namespace server {
namespace udf {

void l2_norm_op(std::shared_ptr<std::vector<HashMapItem>>& indexes, Tensor* input, std::shared_ptr<Tensor>& output);

}
}
}

#endif //XDL_HASH_FILTER_UTILS_H
