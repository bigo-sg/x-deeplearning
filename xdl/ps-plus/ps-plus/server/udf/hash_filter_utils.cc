#include "hash_filter_utils.h"
#include "ps-plus/common/initializer/none_initializer.h"
#include <Eigen/Dense>

namespace ps {
namespace server {
namespace udf {

void l2_norm_op(std::shared_ptr<std::vector<HashMapItem>>& indexes, Tensor* input, std::shared_ptr<Tensor>& output) {
  const auto& shape = input->Shape();
  assert(shape.Size() == 2);
  output.reset(new Tensor(input->Type(), TensorShape({shape[0]}), new initializer::NoneInitializer));
  Eigen::MatrixXf m(indexes->size(), shape[1]);
  m.setZero();

  CASES(input->Type(), {
    T* data_ptr = input->Raw<T>();
    T* data;
    for (size_t i = 0; i < indexes->size(); i++) {

      T* data = data_ptr + (*indexes)[i].id * shape[1];
      for (size_t j = 0; j < shape[1]; j++) {
        m(i, j) = *data;
        ++data;
      }      
    }
  });

  //auto& res = m.rowwise().squaredNorm();
  auto& res = m.rowwise().norm();
  
  CASES(output->Type(), {
    T* data_ptr = output->Raw<T>();
    T* data;
    for (size_t i = 0; i < indexes->size(); i++) {
      data = data_ptr + (*indexes)[i].id;
      *data = res(i);
      ++data;
    }
  });
  
  printf("l2_norm_op indexes size: %ld, l2 norm output range(%f, %f) \n", 
         indexes->size(), res.minCoeff(), res.maxCoeff());
};
}
}
}
