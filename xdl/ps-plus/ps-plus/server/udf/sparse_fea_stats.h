#ifndef PS_SERVER_SPARSE_FEA_STATS_H_
#define PS_SERVER_SPARSE_FEA_STATS_H_

#include "ps-plus/server/slice.h"

namespace ps {
namespace server {
namespace udf {
void UpdateSparseFeaStats(const Slices& slices);
}
}
}
#endif
