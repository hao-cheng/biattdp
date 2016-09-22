/*!
 * \file gaussian_initializer.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_INITIALIZER_GAUSSIAN_INITIALIZER_H_
#define TIALNN_INITIALIZER_GAUSSIAN_INITIALIZER_H_

// std::sqrt
#include <cmath>
// std::vector
#include <vector>
// boost::normal_distribution
#include <boost/random/normal_distribution.hpp>
// boost::variate_generator
#include <boost/random/variate_generator.hpp>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::WeightType
#include "../base.h"
// tialnn::ConnectionBase
#include "../connection/connection_base.h"
// tialnn::InitiazlierBase
#include "initializer_base.h"

namespace tialnn {

//! Gaussian initializer.
class GaussianInitializer : public InitializerBase {
 public:
  explicit GaussianInitializer(const WeightType s) : normal_(0.0f, s) {}

  virtual void RandomlyInitialize(std::vector<WeightType> &weights,
                                  boost::mt19937 &rng_engine) override {
    assert(!weights.empty());

    boost::variate_generator<boost::mt19937&, boost::normal_distribution<WeightType>> gen(rng_engine, normal_);

    for (auto &w : weights) {
      w = gen();
    }
  }

 private:
  //! Random number generator related parameters.
  boost::normal_distribution<WeightType> normal_;

};

} // namespace tialnn

#endif
