#ifndef XDL_CORE_OPS_FEA_STATS_OP_H
#define XDL_CORE_OPS_FEA_STATS_OP_H
#include "xdl/core/framework/cpu_device.h"

namespace xdl {
namespace functor {

template <typename Device, typename T, typename I, typename L>
struct FeaStatsFunctor;

template <typename T, typename I, typename L>
struct FeaStatsFunctor<CpuDevice, T, I, L> {
  void operator()(CpuDevice* d, const Tensor& sindex, const Tensor& ssegment,
                  const Tensor& labels_inputs, Tensor* fea_stats_delta, const std::string& var_name);
};
}  // namespace functor

template <typename T, typename I, typename L>
class FeaStatsCpuOp : public OpKernel {
public:
  Status Init(OpKernelConstruction* ctx) override;
  Status Compute(OpKernelContext* ctx) override;

private:
  std::string var_name_;
};
}

#endif //XDL_CORE_OPS_FEA_STATS_OP_H

