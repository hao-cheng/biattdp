/*!
 * \file vanilla_grad.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_GRADIENT_VANILLA_GRAD_H_
#define TIALNN_GRADIENT_VANILLA_GRAD_H_

// assert
#include <cassert>
// std::vector
#include <vector>
// std::fill
#include <algorithm>
// tialnn::IndexType
// tialnn::GradientType
// tialnn::WeightType
#include "../base.h"
// TIALNN_CBLAS_XXX
#include "../util/numeric.h"
// tialnn::GradientsBase
#include "gradients_base.h"

namespace tialnn {

//! Vanilla gradient descent.
struct VanillaGrad : public GradientsBase {
  //! Allocates gradients.
  void AllocateGradients(IndexType n) override {
    assert(vals.empty());

    vals.resize(n);
  }
  //! Resets gradients.
  void ResetGradients() override {
    assert(!vals.empty());

    std::fill(vals.begin(), vals.end(), 0.0f);
  }

  //! Updates the weights
  //! weights += learning_rate * gradients.
  void UpdateWeights(GradientType learning_rate,
                     WeightType *ptr_weights) override {
    assert(!vals.empty());

    IndexType n = static_cast<IndexType>(vals.size());
    TIALNN_CBLAS_AXPY(n, learning_rate, vals.data(), 1, ptr_weights, 1);
    std::fill(vals.begin(), vals.end(), 0.0f);
  }

  //! Caches the gradients.
  virtual void CacheCurrentGradients() override {
  }

  //! Restores the gradients.
  virtual void RestoreLastGradients() override {
  }

};

} // namespace tialnn

#endif
