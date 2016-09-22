/*!
 * \file adam.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_GRADIENT_ADAM_H_
#define TIALNN_GRADIENT_ADAM_H_

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

//! Gradients for Adam.
struct Adam : public GradientsBase {
  //! Allocates gradients.
  void AllocateGradients(IndexType n) override {
    assert(vals.empty());
    assert(first_moment_vec.empty());
    assert(last_first_moment_vec.empty());
    assert(second_moment_vec.empty());
    assert(last_second_moment_vec.empty());
    assert(aux_grad.empty());

    vals.resize(n);
    first_moment_vec.resize(n);
    last_first_moment_vec.resize(n);
    second_moment_vec.resize(n);
    last_second_moment_vec.resize(n);
    aux_grad.resize(n);
    beta1 = 0.9f;
    beta2 = 0.99f;
    pow_beta1 = 1.0f;
    last_pow_beta1 = 1.0f;
    pow_beta2 = 1.0f;
    last_pow_beta2 = 1.0f;
  }
  //! Resets gradients.
  void ResetGradients() override {
    assert(!vals.empty());
    assert(!first_moment_vec.empty());
    assert(!last_first_moment_vec.empty());
    assert(!second_moment_vec.empty());
    assert(!last_second_moment_vec.empty());
    assert(!aux_grad.empty());

    std::fill(vals.begin(), vals.end(), 0.0f);

    std::fill(first_moment_vec.begin(), first_moment_vec.end(), 0.0f);
    std::fill(last_first_moment_vec.begin(), last_first_moment_vec.end(), 0.0f);
    std::fill(second_moment_vec.begin(), second_moment_vec.end(), 0.0f);
    std::fill(last_second_moment_vec.begin(), last_second_moment_vec.end(), 0.0f);
    std::fill(aux_grad.begin(), aux_grad.end(), 0.0f);

  }

  //! Updates the weights.
  //! first_moment_vec = beta1 * first_moment_vec
  //                  + (1 - beta1) * gradients.
  //! second_moment_vec = beta2 * second_moment_vec
  //                  + (1 - beta2) * gradients^2.
  //! m_hat = first_moment_vec / (1 - pow_beta1).
  //! v_hat = second_moment_vec / (1 - pow_beta2).
  //! delta = m_hat / (sqrt(v_hat) + epsilon).
  void UpdateWeights(GradientType learning_rate,
                     WeightType *ptr_weights) override {
    assert(!vals.empty());
    assert(!first_moment_vec.empty());
    assert(!second_moment_vec.empty());
    assert(!aux_grad.empty());

    IndexType n = static_cast<IndexType>(vals.size());

    std::transform(vals.begin(),
                   vals.end(),
                   aux_grad.begin(),
                   [](GradientType g) { return g * g; });
    TIALNN_CBLAS_AXPBY(n,
                       1.0f - beta1, vals.data(), 1,
                       beta1, first_moment_vec.data(), 1);
    TIALNN_CBLAS_AXPBY(n,
                       1.0f - beta2, aux_grad.data(), 1,
                       beta2, second_moment_vec.data(), 1);

    pow_beta1 *= beta1;
    pow_beta2 *= beta2;
    TIALNN_CBLAS_AXPBY(n, 1.0f / (1.0f - pow_beta1), first_moment_vec.data(), 1, 0.0f, vals.data(), 1);
    TIALNN_CBLAS_AXPBY(n, 1.0f / (1.0f - pow_beta2), second_moment_vec.data(), 1, 0.0f, aux_grad.data(), 1);

    std::transform(vals.begin(),
                   vals.end(),
                   aux_grad.begin(),
                   vals.begin(),
                   [](GradientType g, GradientType v) { return  g / (std::sqrt(v) + 1e-8f); });

    TIALNN_CBLAS_AXPY(n, learning_rate, vals.data(), 1, ptr_weights, 1);
    std::fill(vals.begin(), vals.end(), 0.0f);
  }

  //! Caches the gradients.
  virtual void CacheCurrentGradients() override {
    last_first_moment_vec = first_moment_vec;
    last_second_moment_vec = second_moment_vec;
    last_pow_beta1 = pow_beta1;
    last_pow_beta2 = pow_beta2;
  }

  //! Restores the gradients.
  virtual void RestoreLastGradients() override {
    first_moment_vec = last_first_moment_vec;
    second_moment_vec = last_second_moment_vec;
    pow_beta1 = last_pow_beta1;
    pow_beta2 = last_pow_beta2;
  }

 private:
  //! Auxilary variable for g * g.
  std::vector<GradientType> aux_grad;
  //! Moment vectors.
  std::vector<GradientType> first_moment_vec, second_moment_vec;
  std::vector<GradientType> last_first_moment_vec, last_second_moment_vec;
  //! Exponetial decay factors.
  GradientType beta1, beta2;
  //! Power of exponetial decay factors.
  GradientType pow_beta1, pow_beta2;
  GradientType last_pow_beta1, last_pow_beta2;
};

} // namespace tialnn

#endif
