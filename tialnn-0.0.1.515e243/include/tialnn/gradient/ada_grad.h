/*!
 * \file ada_grad.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_GRADIENT_ADA_GRAD_H_
#define TIALNN_GRADIENT_ADA_GRAD_H_

// assert
#include <cassert>
// std::sqrt
#include <cmath>
// std::vector
#include <vector>
// std::fill
// std::transform
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

//! AdaGrad.
struct AdaGrad : public GradientsBase {
  //! Allocates gradients.
  void AllocateGradients(IndexType n) override {
    assert(vals.empty());
    assert(sum_squares_grad.empty());
    assert(last_sum_squares_grad.empty());

    vals.resize(n);
    sum_squares_grad.resize(n);
    last_sum_squares_grad.resize(n);
  }
  //! Resets gradients.
  void ResetGradients() override {
    assert(!vals.empty());
    assert(!sum_squares_grad.empty());
    assert(!last_sum_squares_grad.empty());

    std::fill(vals.begin(), vals.end(), 0.0f);

    //! \note Empirically, it is good to initialize it as epsilon for
    //!       better condition the denominator.
    //!       The idea is to make sure the approximated Hessian is invertible.
    //!       See the original paper and \cite{Zeiler2012arXiv}.
    std::fill(sum_squares_grad.begin(), sum_squares_grad.end(), 1e-6f);
    std::fill(last_sum_squares_grad.begin(), last_sum_squares_grad.end(), 1e-6f);
  }

  //! Updates the weights.
  //! v = v + g^2.
  //! gradients = g / sqrt(v + epsilon).
  //! weights += learning_rate * gradients.
  void UpdateWeights(GradientType learning_rate,
                     WeightType *ptr_weights) override {
    assert(!vals.empty());
    assert(!sum_squares_grad.empty());

    std::transform(vals.begin(),
                   vals.end(),
                   sum_squares_grad.begin(),
                   sum_squares_grad.begin(),
                   [](GradientType g, GradientType v) { return v + g * g; });

    std::transform(vals.begin(),
                   vals.end(),
                   sum_squares_grad.begin(),
                   vals.begin(),
                   [](GradientType g, GradientType v) { return g / std::sqrt(v); });

    IndexType n = static_cast<IndexType>(vals.size());
    TIALNN_CBLAS_AXPY(n, learning_rate, vals.data(), 1, ptr_weights, 1);
    std::fill(vals.begin(), vals.end(), 0.0f);
  }

  //! Caches the gradients.
  virtual void CacheCurrentGradients() override {
    last_sum_squares_grad = sum_squares_grad;
  }

  //! Restores the gradients.
  virtual void RestoreLastGradients() override {
    sum_squares_grad = last_sum_squares_grad;
  }

 private:
  //! Sum squares of gradient.
  std::vector<GradientType> sum_squares_grad, last_sum_squares_grad;

};

} // namespace tialnn

#endif
