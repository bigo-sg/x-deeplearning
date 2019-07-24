#include "fea_stats_op.h"

#include <omp.h>
#include <unordered_map>
#include "xdl/core/framework/op_kernel.h"
#include "xdl/core/framework/op_define.h"
#include "xdl/core/framework/op_registry.h"
#include "xdl/core/framework/cpu_device.h"
#include "xdl/core/lib/atomic.h"

namespace xdl {
namespace functor {

template <typename T, typename I, typename L>
void FeaStatsFunctor<CpuDevice, T, I, L>::operator()(CpuDevice* d, const Tensor& sindex, const Tensor& ssegment,
                                                     const Tensor& labels_inputs, Tensor* fea_stats_delta,
                                                     const std::string& var_name) {
  T* plabel = labels_inputs.Raw<T>();

  size_t id_num = sindex.Shape()[0];
  size_t label_dim = labels_inputs.Shape()[1];
  size_t stats_dim = label_dim + 1;

  TensorShape out_shape({ssegment.Shape()[0], stats_dim});
  *fea_stats_delta = Tensor(d, out_shape, DataTypeToEnum<T>::v());

  T* pout = fea_stats_delta->Raw<T>();
  std::memset(pout, 0, sizeof(T) * out_shape.NumElements());

  I* psindex = sindex.Raw<I>();
  I* psseg = ssegment.Raw<I>();
  size_t sseg_size = ssegment.Shape().NumElements();

  #pragma omp parallel for
  for (size_t i = 0; i < id_num; ++i) {
    size_t sseg_idx = std::lower_bound(psseg, psseg + sseg_size, i + 1) - psseg;
    T* src = plabel + psindex[i] * label_dim;
    T* dst = pout + sseg_idx * stats_dim;

    // show
    common::cpu_atomic_add<T>(1, dst);

    for (size_t j = 0; j < label_dim; ++j) {
      common::cpu_atomic_add<T>(*(src + j), dst + j + 1);
    }
  }

};

template struct FeaStatsFunctor<CpuDevice, double, int32_t, int64_t>;

}  // namespace functortemplate <typename T, typename I, typename L>

template <typename T, typename I, typename L>
Status FeaStatsCpuOp<T, I, L>::Init(OpKernelConstruction* ctx) {
  XDL_CHECK_STATUS(ctx->GetAttr("var_name", &var_name_));
  return Status::Ok();
};

template <typename T, typename I, typename L>
Status FeaStatsCpuOp<T, I, L>::Compute(OpKernelContext* ctx) {
  Tensor sindex, ssegment, labels_inputs, fea_stats_delta;

  XDL_CHECK_STATUS(ctx->GetInput(0, &sindex));
  XDL_CHECK_COND(1 == sindex.Shape().Size(),
                 Status::ArgumentError("sindex input dim must be 1"));

  XDL_CHECK_STATUS(ctx->GetInput(1, &ssegment));
  XDL_CHECK_COND(1 == ssegment.Shape().Size(),
                 Status::ArgumentError("ssegment input dim must be 1"));

  XDL_CHECK_STATUS(ctx->GetInput(2, &labels_inputs));
  XDL_CHECK_COND(2 == labels_inputs.Shape().Size(),
                 Status::ArgumentError("labels_inputs input dim must be 2"));

  CpuDevice* device = dynamic_cast<CpuDevice*>(ctx->GetDevice());
  auto fn = functor::FeaStatsFunctor<CpuDevice, T, I, L>();
  fn(device, sindex, ssegment, labels_inputs, &fea_stats_delta, var_name_);
  
  ctx->SetOutput(0, fea_stats_delta);

  return Status::Ok();
}

XDL_DEFINE_OP(FeaStats)
.Input("sindex", "itype")
.Input("ssegment", "itype")
.Input("labels_inputs", "dtype")
.Output("fea_stats_delta", "dtype")
.Attr("var_name", AttrValue::kString)
.Attr("dtype", AttrValue::kDataType)
.Attr("itype", AttrValue::kDataType)
.Attr("ltype", AttrValue::kDataType);

#define REGISTER_KERNEL(T, I, L)                        \
  XDL_REGISTER_KERNEL(FeaStats, FeaStatsCpuOp<T, I, L>) \
    .Device("CPU")                                      \
    .AttrDataType<T>("dtype")                           \
    .AttrDataType<I>("itype")                           \
    .AttrDataType<L>("ltype")

REGISTER_KERNEL(float, int32_t, int64_t);

#undef REGISTER_KERNEL
}
