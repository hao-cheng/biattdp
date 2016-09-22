/*!
 * \file mikolov_initializer.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_INITIALIZER_MIKOLOV_INITIALIZER_H_
#define TIALNN_INITIALIZER_MIKOLOV_INITIALIZER_H_

// std::vector
#include <vector>
// boost::uniform_01
#include <boost/random/uniform_01.hpp>
// boost::variate_generator
#include <boost/random/variate_generator.hpp>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::WeightType
#include "../base.h"
// tialnn::InitiazlierBase
#include "initializer_base.h"

namespace tialnn {

//! Mikolov's RNNLM initializer.
//! It approximates the Gaussian distribution N(0, 0.1) according to the central limit theory.
class MikolovInitializer : public InitializerBase {
 public:
  virtual void RandomlyInitialize(std::vector<WeightType> &weights,
                                  boost::mt19937 &rng_engine) override {
    assert(!weights.empty());

    boost::variate_generator<boost::mt19937&, boost::uniform_01<WeightType>> gen(rng_engine, uniform_01_);

    for (auto &w : weights) {
      w = (gen() * 0.2f - 0.1f)
          + (gen() * 0.2f - 0.1f)
          + (gen() * 0.2f - 0.1f);
    }
  }


 private:
  //! Random number generator related parameters.
  boost::uniform_01<WeightType> uniform_01_;
};

} // namespace tialnn

#endif
