#include "ps-plus/server/udf/sparse_fea_stats.h"
#include "ps-plus/common/hashmap.h"

namespace ps {
namespace server {
namespace udf {
void UpdateSparseFeaStats(const Slices& slices) {
    if (dynamic_cast<WrapperData<HashMap>*>(slices.variable->GetSlicer())) {
        for (size_t slice : slices.slice_id) {
            if ((int64_t)slice != ps::HashMap::NOT_ADD_ID) {
                slices.variable->UpdateStatsInfo(slice, 0, 1, 1);
            }
        }
    }
}
}
}
}
