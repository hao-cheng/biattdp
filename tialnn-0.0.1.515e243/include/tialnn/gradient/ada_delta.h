/*!
 * \file ada_delta.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_GRADIENT_ADA_DELTA_H_
#define TIALNN_GRADIENT_ADA_DELTA_H_

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

//! AdaDelta \cite{Zeiler2012arXiv}.
struct AdaDelta : public GradientsBase {
  //! Allocates gradients.
  void AllocateGradients(IndexType n) override {
    assert(vals.empty());
    assert(sum_squares_grad.empty());
    assert(last_sum_squares_grad.empty());
    assert(sum_squares_delta.empty());
    assert(last_sum_squares_delta.empty());
    assert(aux_squares_g.empty());

    vals.resize(n);
    sum_squares_grad.resize(n);
    last_sum_squares_grad.resize(n);
    sum_squares_delta.resize(n);
    last_sum_squares_delta.resize(n);
    aux_squares_g.resize(n);
    rho = 0.95f; //<! 0.95 is stable for most cases.
  }
  //! Resets gradients.
  void ResetGradients() override {
    assert(!vals.empty());
    assert(!sum_squares_grad.empty());
    assert(!sum_squares_delta.empty());

    std::fill(vals.begin(), vals.end(), 0.0f);

    std::fill(sum_squares_grad.begin(), sum_squares_grad.end(), 0.0f);
    std::fill(last_sum_squares_grad.begin(), last_sum_squares_grad.end(), 0.0f);
    std::fill(sum_squares_delta.begin(), sum_squares_delta.end(), 0.0f);
    std::fill(last_sum_squares_delta.begin(), last_sum_squares_delta.end(), 0.0f);
    std::fill(aux_squares_g.begin(), aux_squares_g.end(), 0.0f);
  }

  //! Updates the weights.
  //! v = v + g^2.
  //! gradients = g / sqrt(v + epsilon).
  //! gradients = gradients * sqrt(d + epsilon).
  //! d = rho * d + (1 - rho) * gradients.
  //! weights += learning_rate * gradients.
  void UpdateWeights(GradientType learning_rate,
                     WeightType *ptr_weights) override {
    assert(!vals.empty());
    assert(!sum_squares_grad.empty());
    assert(!sum_squares_delta.empty());

    IndexType n = static_cast<IndexType>(vals.size());

    std::transform(vals.begin(),
                   vals.end(),
                   aux_squares_g.begin(),
                   [](GradientType g) { return g * g; });
    TIALNN_CBLAS_AXPBY(n,
                       1 - rho, aux_squares_g.data(), 1,
                       rho, sum_squares_grad.data(), 1);

    std::transform(vals.begin(),
                   vals.end(),
                   sum_squares_grad.begin(),
                   vals.begin(),
                   [](GradientType g, GradientType v) { return g / std::sqrt(v + 1e-6f); });

    std::transform(vals.begin(),
                   vals.end(),
                   sum_squares_delta.begin(),
                   vals.begin(),
                   [](GradientType g, GradientType v) { return  g * std::sqrt(v + 1e-6f); });

    std::transform(vals.begin(),
                   vals.end(),
                   aux_squares_g.begin(),
                   [](GradientType g) { return g * g; });
    TIALNN_CBLAS_AXPBY(n,
                       1 - rho, aux_squares_g.data(), 1,
                       rho, sum_squares_delta.data(), 1);

    //! \note For AdaDelta, the learning rate is always 1.
    //!       But we allow to provide a learning rate.
    TIALNN_CBLAS_AXPY(n, learning_rate, vals.data(), 1, ptr_weights, 1);
    std::fill(vals.begin(), vals.end(), 0.0f);
  }

  //! Caches the gradients.
  virtual void CacheCurrentGradients() override {
    last_sum_squares_grad = sum_squares_grad;
    last_sum_squares_delta = sum_squares_delta;
  }

  //! Restores the gradients.
  virtual void RestoreLastGradients() override {
    sum_squares_grad = last_sum_squares_grad;
    sum_squares_delta = last_sum_squares_delta;
  }

 private:
  //! Sum squares of gradient.
  std::vector<GradientType> sum_squares_grad, last_sum_squares_grad;
  //! Sum squares of delta.
  std::vector<GradientType> sum_squares_delta, last_sum_squares_delta;
  //! Auxilary variable for g * g.
  std::vector<GradientType> aux_squares_g;
  //! Exponential decay factor.
  GradientType rho;

};

} // namespace tialnn

#endif
