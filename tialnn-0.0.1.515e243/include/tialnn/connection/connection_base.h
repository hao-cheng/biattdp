/*!
 * \file connection_base.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_CONNECTION_CONNECTION_BASE_H_
#define TIALNN_CONNECTION_CONNECTION_BASE_H_

// assert
#include <cassert>
// std::cout
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::IndexType
// tialnn::WeightType
#include "../base.h"
// tialnn::read_xxx
// tialnn::write_xxx
#include "../util/futil.h"
// tialnn::InitializerBase
#include "../initializer/initializer_base.h"

namespace tialnn {

//! Base class of neural network connections.
//! \note It only defines the interfaces needed when registrating the
//  connections for auto update/cache/restore/read/write.
//  Propagation will be defined in XConnectionBase and YConnectionBase.
class ConnectionBase {

 public:
  //! Constructor.
  explicit ConnectionBase() : ninput_(0), noutput_(0) {}
  //! Destructor.
  virtual ~ConnectionBase() {}

  //! Returns ninput_.
  IndexType ninput() const { return ninput_; }
  //! Returns noutput_.
  IndexType noutput() const { return noutput_; }
  //! Returns the weight.
  virtual WeightType weights(IndexType i, IndexType j) const = 0;
  //! Returns the weights.
  virtual const std::vector<WeightType>& weights() const = 0;

  //! Sets the weight.
  virtual void set_weights(IndexType i, IndexType j, WeightType val) = 0;
  //! Sets the weights.
  virtual void set_weights(const std::vector<WeightType> &weights) = 0;
  //! Sets the dimensions.
  // Automatically calls AllocateConnection().
  void set_dims(IndexType ninput, IndexType noutput) {
    assert(ninput_ == 0);
    assert(noutput_ == 0);

    ninput_ = ninput;
    noutput_ = noutput;
    AllocateConnection();
  }

  //! Writes the connection to stream.
  void WriteConnection(std::ofstream &ofs) {
    assert(ninput_ > 0);
    assert(noutput_ > 0);

    write_single(ofs, ninput_);
    write_single(ofs, noutput_);
    WriteConnectionImpl(ofs);
  }
  //! Reads the connection from stream.
  void ReadConnection(std::ifstream &ifs) {
    std::cout << "***reading connection***" << std::endl;
    read_single(ifs, ninput_);
    std::cout << "ninput_: " << ninput_ << std::endl;
    read_single(ifs, noutput_);
    std::cout << "noutput_: " << noutput_ << std::endl;
    ReadConnectionImpl(ifs);
  }
  //! Writes the connection to txt.
  void WriteConnectionToTxt(std::ostream &os) {
    os << "ninput_: " << ninput_ << std::endl;
    os << "noutput_: " << noutput_ << std::endl;
    WriteConnectionToTxtImpl(os);
  }

  //! Randomly initializes the weights.
  virtual void RandomlyInitialize(InitializerBase &initializer,
                                  boost::mt19937 &rng_engine) = 0;

  //! Resets the connection.
  virtual void ResetConnection() = 0;

  //! Updates the weights of the connection.
  //! weights_ += learning_rate * gradients_.
  //! The gradients of the connection are reset automatically.
  virtual void UpdateWeights(GradientType learning_rate) = 0;

  //! Caches the weights.
  //! \note usually called at the end of an epoch, not an iteration.
  virtual void CacheCurrentWeights() = 0;
  //! Restores the weights.
  //! \note: if needed, usually called at the end of an epoch, not an iteration.
  virtual void RestoreLastWeights() = 0;

 private:
  //! Allocates the connection.
  virtual void AllocateConnection() = 0;
  //! Implementation of WriteConnection.
  virtual void WriteConnectionImpl(std::ofstream &ofs) = 0;
  //! Implementaion of ReadConnection.
  virtual void ReadConnectionImpl(std::ifstream &ifs) = 0;
  //! Implementation of WriteConnectionToTxt.
  virtual void WriteConnectionToTxtImpl(std::ostream &ofs) = 0;

  //! The number of inputs of the connection.
  IndexType ninput_;
  //! The number of outputs of the connection.
  IndexType noutput_;
};

} // namespace tialnn

#endif
