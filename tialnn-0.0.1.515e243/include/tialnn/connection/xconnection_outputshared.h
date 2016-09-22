/*!
 * \file xconnection_outputshared.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_CONNECTION_XCONNECTION_OUTPUTSHARED_H_
#define TIALNN_CONNECTION_XCONNECTION_OUTPUTSHARED_H_

// assert
#include <cassert>
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// std::vector
#include <vector>
// std::fill
#include <algorithm>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
// tialnn::GradientType
// tialnn::WeightType
#include "../base.h"
// TIALNN_CBLAS_XXX
#include "../util/numeric.h"
// tialnn::write_xxx
// tialnn::read_xxx
#include "../util/futil.h"
// tialnn::InitializerBase
#include "../initializer/initializer_base.h"
// tialnn::XConnectionBase
#include "xconnection_base.h"

namespace tialnn {

//! X-Connection.
// - All output neurons share the same weights.
//!   input0_output,
//!   input1_output,
//!   ...
template <class Gradients>
class XConnectionOutputShared : public XConnectionBase {
 public:
  //! Constructor.
  explicit XConnectionOutputShared() {}
  //! Destructor.
  virtual ~XConnectionOutputShared() {}

  //! Returns the weight.
  virtual WeightType weights(IndexType i, IndexType j) const override {
    assert(i < ninput());
    assert(j < noutput());
    assert(static_cast<IndexType>(weights_.size()) == ninput());

    return weights_[i];
  }
  //! Returns the weights.
  virtual const std::vector<WeightType>& weights() const override { return weights_; }

  //! Sets the weight.
  virtual void set_weights(IndexType i, IndexType j, WeightType val) override {
    assert(i < ninput());
    assert(j < noutput());
    assert(static_cast<IndexType>(weights_.size()) == ninput());

    weights_[i] = val;
  }
  //! Sets the weights.
  virtual void set_weights(const std::vector<WeightType> &weights) override {
    assert(static_cast<IndexType>(weights.size()) == ninput());

    weights_ = weights;
  }

  //! Randomly initializes the weights.
  virtual void RandomlyInitialize(InitializerBase &initializer,
                                  boost::mt19937 &rng_engine) override {
    assert(!weights_.empty());

    initializer.RandomlyInitialize(weights_,
                                   rng_engine);
  }

  //! Resets the connection.
  virtual void ResetConnection() override {
    assert(!weights_.empty());
    assert(!last_weights_.empty());
    assert(!gradients_.vals.empty());

    std::fill(weights_.begin(), weights_.end(), 0.0f);
    std::fill(last_weights_.begin(), last_weights_.end(), 0.0f);
    gradients_.ResetGradients();
  }

  //! Computes the weighted sum of the activations of input layer neurons and
  //! propagates them to the output layer.
  //! output_activationinputs = weights_ * input_activations + beta * output_activationinputs.
  //! \param beta                           controls whether to reset the activation inputs.
  //! \param input_activations              input activations.
  //! \param output_activationinputs        output activationinputs.
  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs) override {
    assert(static_cast<IndexType>(input_activations.size()) == ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) == noutput());

    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(noutput(), beta, output_activationinputs.data(), 1);
    }

    ActivationType val = TIALNN_CBLAS_DOT(ninput(),
                                          weights_.data(), 1,
                                          input_activations.data(), 1);
    TIALNN_CBLAS_AXPY(noutput(), 1,
                      &val, 0,
                      output_activationinputs.data(), 1);
  }

  //! Computes the weighted sum of the error derivatives of the output layer neurons and
  //! propagates them to the input layer.
  //! input_errorinputs = weights_.transpose() * output_errors + beta * intput_errorinputs.
  //! \param beta                           controls whether to reset the error inputs.
  //! \param output_errors                  output errors.
  //! \param input_errorinputs              input errorinputs.
  virtual void BackwardPropagate(WeightType beta,
                                 const std::vector<ErrorType> &output_errors,
                                 std::vector<ErrorType> &input_errorinputs) override {
    assert(static_cast<IndexType>(input_errorinputs.size()) == ninput());
    assert(static_cast<IndexType>(output_errors.size()) == noutput());

    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(ninput(), beta, input_errorinputs.data(), 1);
    }

    ErrorType val = TIALNN_CBLAS_ASUM(noutput(), output_errors.data(), 1);
    TIALNN_CBLAS_AXPY(ninput(), val,
                      weights_.data(), 1,
                      input_errorinputs.data(), 1);
  }

  //! Updates the gradients of the connection.
  //! gradients_ += output_errors * input_activations.transpose().
  //! \param output_errors                  output errors.
  //! \param input_errorinputs              input errorinputs.
  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<ActivationType> &input_activations) override {
    assert(static_cast<IndexType>(input_activations.size()) == ninput());
    assert(static_cast<IndexType>(output_errors.size()) == noutput());

    ErrorType val = TIALNN_CBLAS_ASUM(noutput(), output_errors.data(), 1);
    TIALNN_CBLAS_AXPY(ninput(), val,
                      input_activations.data(), 1,
                      gradients_.vals.data(), 1);
  }

  //! Batch computes the weighted sum of the activations of input layer neurons and
  //! propagates them to the output layer.
  //! output_activationinputs = weights_ * input_activations + beta * output_activationinputs.
  //! \param batchsize                      batch size.
  //! \param beta                           controls whether to reset the activation inputs.
  //! \param input_activations              input activations.
  //! \param bx_start                       batch offset of input_activations.
  //! \param output_activationinputs        output activationinputs.
  //! \param by_start                       batch offset of output_activationinputs.
  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start) override {
    assert(static_cast<IndexType>(input_activations.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) >= (batchsize + by_start) * noutput());

    IndexType nin = ninput();
    IndexType nout = noutput();

    ActivationType *py = output_activationinputs.data() + by_start * nout;

    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(nout * batchsize, beta, py, 1);
    }

    aux_dvec_.resize(batchsize);
    std::fill(aux_dvec_.begin(), aux_dvec_.end(), 0.0f);
    WeightType *pdvec = aux_dvec_.data();

    TIALNN_CBLAS_GEMV(CblasColMajor, CblasTrans,
                      nin, batchsize,
                      1.0, input_activations.data() + bx_start * nin, nin,
                      weights_.data(), 1,
                      0.0, pdvec, 1);

    TIALNN_CBLAS_GER(CblasColMajor, nout, batchsize, 1.0,
                     aux_ones_.data(), 1,
                     pdvec, 1,
                     py, nout);
  }

  //! Batch computes the weighted sum of the error derivatives of the output layer neurons and
  //! propagates them to the input layer.
  //! input_errorinputs = weights_.transpose() * output_errors + beta * intput_errorinputs.
  //! \param batchsize                      batch size.
  //! \param beta                           controls whether to reset the error inputs.
  //! \param output_errors                  output errors.
  //! \param by_start                       batch offset of output_errors.
  //! \param input_errorinputs              input errorinputs.
  //! \param bx_start                       batch offset of input_errorinputs.
  virtual void BatchBackwardPropagate(IndexType batchsize, WeightType beta,
                                      const std::vector<ErrorType> &output_errors,
                                      IndexType by_start,
                                      std::vector<ErrorType> &input_errorinputs,
                                      IndexType bx_start) override {
    assert(static_cast<IndexType>(input_errorinputs.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    ErrorType *px = input_errorinputs.data() + bx_start * nin;

    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(nin * batchsize, beta, px, 1);
    }

    aux_dvec_.resize(batchsize);
    std::fill(aux_dvec_.begin(), aux_dvec_.end(), 0.0f);
    WeightType *pdvec = aux_dvec_.data();

    TIALNN_CBLAS_GEMV(CblasColMajor, CblasTrans,
                      nout, batchsize,
                      1.0, output_errors.data() + by_start * nout, nout,
                      aux_ones_.data(), 1,
                      0.0, pdvec, 1);

    TIALNN_CBLAS_GER(CblasColMajor, nin, batchsize, 1.0,
                     weights_.data(), 1,
                     pdvec, 1,
                     px, nin);
  }

  //! Batch updates the gradients of the connection.
  //! gradients_ += output_errors * input_activations.transpose().
  //! \param batchsize                      batch size.
  //! \param output_errors                  output errors.
  //! \param by_start                       batch offset of output_errors.
  //! \param input_activations              input activations.
  //! \param bx_start                       batch offset of input_errorinputs.
  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<ActivationType> &input_activations,
                                        IndexType bx_start) override {
    assert(static_cast<IndexType>(input_activations.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    const ActivationType *px = input_activations.data() + bx_start * nin;
    const ErrorType *py = output_errors.data() + by_start * nout;
    GradientType *pg = gradients_.vals.data();
    for (IndexType j = 0; j < nout; j++) {
      TIALNN_CBLAS_GEMV(CblasColMajor, CblasNoTrans,
                        nin, batchsize,
                        1.0, px, nin,
                        py + j, nout,
                        1.0, pg, 1);
    }
  }

  //! Updates the weights of the connection.
  //! weights_ += learning_rate * gradients_.
  //! The gradients of the connection are reset automatically.
  virtual void UpdateWeights(GradientType learning_rate) override {
    gradients_.UpdateWeights(learning_rate, weights_.data());
  }

  //! Caches the weights.
  //! \note usually called at the end of an epoch, not an iteration.
  virtual void CacheCurrentWeights() override {
    last_weights_ = weights_; //!< should not use swap as we do not want to overwrite weights_.
    gradients_.CacheCurrentGradients();
  }

  //! Restores the weights.
  //! \note if needed, usually called at the end of an epoch, not an iteration.
  virtual void RestoreLastWeights() override {
    weights_ = last_weights_; //!< should not use swap as we do not want to overwrite last_weights_.
    gradients_.RestoreLastGradients();
  }

 private:
  //! Allocates the connection.
  virtual void AllocateConnection() override {
    assert(weights_.empty());
    assert(last_weights_.empty());
    assert(gradients_.vals.empty());

    IndexType n = ninput();

    weights_.resize(n);
    last_weights_.resize(n);
    gradients_.AllocateGradients(n);

    aux_ones_.resize(noutput());
    std::fill(aux_ones_.begin(), aux_ones_.end(), 1.0f);

    ResetConnection();
  }

  //! Implementation of WriteConnection.
  virtual void WriteConnectionImpl(std::ofstream &ofs) override {
    write_1d_vector(ofs, weights_);
  }
  //! Implementaion of ReadConnection.
  virtual void ReadConnectionImpl(std::ifstream &ifs) override {
    read_1d_vector(ifs, weights_);
  }
  //! Implementation of WriteConnetionToTxt.
  virtual void WriteConnectionToTxtImpl(std::ostream &os) override {
    for (std::vector<WeightType>::iterator wit = weights_.begin();
         wit != weights_.end(); ++wit) {
      os << *wit << std::endl;
    }
  }

  //! The weights of the connection (at current epoch and last epoch,
  //! respectively).
  std::vector<WeightType> weights_, last_weights_;
  //! Gradients of the weights.
  Gradients gradients_;

  //! Auxilary vector to store temporary variables.
  std::vector<WeightType> aux_dvec_;

  //! Auxilary vector used to represent all ones.
  std::vector<WeightType> aux_ones_;
};

} // namespace tialnn

#endif
