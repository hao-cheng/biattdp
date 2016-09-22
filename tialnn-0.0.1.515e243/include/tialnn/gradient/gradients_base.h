/*!
 * \file gradients_base.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_GRADIENT_GRADIENTS_BASE_H_
#define TIALNN_GRADIENT_GRADIENTS_BASE_H_

// std::vector
#include <vector>
// tialnn::IndexType
// tialnn::GradientType
// tialnn::WeightType
#include "../base.h"

namespace tialnn {

//! Base struct for gradients.
//! \note We use struct for gradients to allow public access to the value.
//!       The derived classes can have private member if that makes sense.
//! \note Derived classes implements different weight update mechanisms.
struct GradientsBase {
  //! Allocates gradients
  virtual void AllocateGradients(IndexType n) = 0;
  //! Resets gradients.
  virtual void ResetGradients() = 0;
  //! Updates weights.
  virtual void UpdateWeights(GradientType learning_rate,
                             WeightType *ptr_weights) = 0;
  //! Caches the gradients.
  //! \note usually called at the end of an epoch, not an iteration.
  virtual void CacheCurrentGradients() = 0;
  //! Restores the gradients.
  //! \note: if needed, usually called at the end of an epoch, not an iteration.
  virtual void RestoreLastGradients() = 0;

  //! Values of the gradients.
  std::vector<GradientType> vals;
};

} // namespace tialnn

#endif
