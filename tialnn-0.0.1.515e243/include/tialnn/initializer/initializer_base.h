/*!
 * \file initializer_base.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_INITIALIZER_INITIALIZER_BASE_H_
#define TIALNN_INITIALIZER_INITIALIZER_BASE_H_

// std::vector
#include <vector>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::WeightType
#include "../base.h"

namespace tialnn {

//! Base class for weight intialization.
class InitializerBase {
 public:
  virtual void RandomlyInitialize(std::vector<WeightType> &weights,
                                  boost::mt19937 &rng_engine) = 0;
};

} // namespace tialnn

#endif
