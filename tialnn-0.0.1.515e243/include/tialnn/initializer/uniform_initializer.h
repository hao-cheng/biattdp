/*!
 * \file uniform_initializer.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_INITIALIZER_UNIFORM_INITIALIZER_H_
#define TIALNN_INITIALIZER_UNIFORM_INITIALIZER_H_

// std::sqrt
#include <cmath>
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
// tialnn::ConnectionBase
#include "../connection/connection_base.h"
// tialnn::InitiazlierBase
#include "initializer_base.h"

namespace tialnn {

//! Uniform initializer.
class UniformInitializer : public InitializerBase {
 public:
  //! Sets the scale.
  void set_scale(WeightType s) { scale_ = s; }

  //! Easier interface for setting the scale according to Xavier initialization.
  //! \see \cite{Xavier2010AISTATS},
  //!       http://deeplearning.net/tutorial/mlp.html#weight-initialization,
  //!       and http://deepdish.io/2015/02/24/network-initialization/.
  void XavierSetScale(const ConnectionBase *pconn, bool sigmoid) {
    scale_ = std::sqrt(6.0f / (pconn->ninput() + pconn->noutput()));
    if (sigmoid) {
      scale_ *= 4.0f;
    }
  }

  virtual void RandomlyInitialize(std::vector<WeightType> &weights,
                                  boost::mt19937 &rng_engine) override {
    assert(!weights.empty());

    boost::variate_generator<boost::mt19937&, boost::uniform_01<WeightType>> gen(rng_engine, uniform_01_);
    for (auto &w : weights) {
      w = (gen() * 2 * scale_ - scale_);
    }
  }

 private:
  //! Random number generator related parameters.
  boost::uniform_01<WeightType> uniform_01_;

  //! Scale.
  WeightType scale_;
};

} // namespace tialnn

#endif
