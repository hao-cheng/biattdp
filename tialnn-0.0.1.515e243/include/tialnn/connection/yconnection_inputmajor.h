/*!
 * \file yconnection_inputmajor.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_CONNECTION_YCONNECTION_INPUTMAJOR_H_
#define TIALNN_CONNECTION_YCONNECTION_INPUTMAJOR_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::ifstream
// std::ofstream
#include <fstream>
// std::vector
#include <vector>
// std::unordered_map
#include <unordered_map>
// std::fill
// std::sort
// std::unique
#include <algorithm>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
// tialnn::WeightType
// tialnn::GradientType
#include "../base.h"
// TIALNN_CBLAS_XXX
#include "../util/numeric.h"
// tialnn::write_xxx
// tialnn::read_xxx
#include "../util/futil.h"
// tialnn::InitializerBase
#include "../initializer/initializer_base.h"
// tialnn::YConnectionBase
#include "yconnection_base.h"

namespace tialnn {

//! Y-Connection.
//! Weights are stored using the input major mechanism.
//!   input0_output0, input0_output1, ..., input0_outputN
//!   input1_output0, ...
//!   ...
template <class Gradients>
class YConnectionInputMajor : public YConnectionBase {
 public:
  //! Constructor.
  explicit YConnectionInputMajor() {}
  //! Destructor.
  virtual ~YConnectionInputMajor() {}

  //! Returns the weight.
  virtual WeightType weights(IndexType i, IndexType j) const override {
    assert(i < ninput());
    assert(j < noutput());
    assert(static_cast<IndexType>(weights_.size()) == ninput() * noutput());
    return weights_[i * noutput() + j];
  }
  //! Returns the weights.
  virtual const std::vector<WeightType>& weights() const override { return weights_; }

  //! Sets the weight.
  virtual void set_weights(IndexType i, IndexType j, WeightType val) override {
    assert(i < ninput());
    assert(j < noutput());
    assert(static_cast<IndexType>(weights_.size()) == ninput() * noutput());
    weights_[i * noutput() + j] = val;
  }
  //! Sets the weights.
  virtual void set_weights(const std::vector<WeightType> &weights) override {
    assert(static_cast<IndexType>(weights.size()) == ninput() * noutput());
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

    std::fill(weights_.begin(), weights_.end(), 0.0f);
    std::fill(last_weights_.begin(), last_weights_.end(), 0.0f);
    for (auto &grads : gradients_) {
      assert(!grads.vals.empty());

      grads.ResetGradients();
    }

    gradients_touched_.clear();
  }

  //! ======================================================================
  //! Computes the weighted sum of the activations of input layer neurons
  //! and propagates them to the output layer.
  //! output_activationinputs = weights_ * input_activations  + beta * output_activationinputs.
  //! \param beta                           controls whether to reset the activation inputs.
  //! \param input_activations              input activations (can be sparse unordered_map).
  //! \param input_neurons                  (Optional) active neurons.
  //! \param output_activationinputs        output activationinputs.
  //! \param output_neurons                 (Optional) active neurons.
  //! ======================================================================

  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs) override {
    assert(static_cast<IndexType>(input_activations.size()) == ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) == noutput());

    const IndexType nin = ninput();
    const IndexType nout = noutput();
    TIALNN_CBLAS_GEMV(CblasColMajor, CblasNoTrans,
                      nout, nin,
                      1.0, weights_.data(), nout,
                      input_activations.data(), 1,
                      beta, output_activationinputs.data(), 1);
  }

  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                const std::vector<IndexType> &input_neurons,
                                std::vector<ActivationType> &output_activationinputs) override {
    assert(static_cast<IndexType>(input_activations.size()) == ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) == noutput());
    assert(is_unique(input_neurons));

    const IndexType nout = noutput();
    const WeightType *pw = weights_.data();
    ActivationType *py = output_activationinputs.data();

    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(nout, beta, py, 1);
    }

    for (auto &i : input_neurons) {
      assert(i < ninput());

      const ActivationType x = input_activations[i];
      if (x == 0) {
        continue;
      }
      TIALNN_CBLAS_AXPY(nout, x,
                        pw + i * nout, 1,
                        py, 1);
    }
  }

  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs,
                                const std::vector<IndexType> &output_neurons) override {
    TIALNN_ERR("Error: InputMajor but only set output_neurons.");
    std::exit(EXIT_FAILURE);
  }

  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                const std::vector<IndexType> &input_neurons,
                                std::vector<ActivationType> &output_activationinputs,
                                const std::vector<IndexType> &output_neurons) override {
    assert(static_cast<IndexType>(input_activations.size()) == ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) == noutput());
    assert(is_unique(input_neurons));
    assert(is_unique(output_neurons));

    if (beta != 1.0f) {
      for (auto &j : output_neurons) {
        output_activationinputs[j] *= beta;
      }
    }

    const IndexType nout = noutput();
    for (auto &i : input_neurons) {
      assert(i < ninput());

      const ActivationType x = input_activations[i];
      if (x == 0) {
        continue;
      }
      const IndexType offset = i * nout;
      for (auto &j : output_neurons) {
        assert(j < nout);

        output_activationinputs[j] += x * weights_[offset + j];
      }
    }
  }

  virtual void ForwardPropagate(WeightType beta,
                                const std::unordered_map<IndexType, ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs) override {
    assert(static_cast<IndexType>(output_activationinputs.size()) == noutput());

    const IndexType nout = noutput();
    const WeightType *pw = weights_.data();
    ActivationType *py = output_activationinputs.data();

    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(nout, beta, py, 1);
    }

    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->first < ninput());
      assert(xit->second != 0);

      const IndexType offset = xit->first * nout;
      TIALNN_CBLAS_AXPY(nout, xit->second,
                        pw + offset, 1,
                        py, 1);
    }
  }

  virtual void ForwardPropagate(WeightType beta,
                                const std::unordered_map<IndexType, ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs,
                                const std::vector<IndexType> &output_neurons) override {
    assert(static_cast<IndexType>(output_activationinputs.size()) == noutput());

    const IndexType nout = noutput();

    if (beta != 1.0f) {
      for (auto &j : output_neurons) {
        output_activationinputs[j] *= beta;
      }
    }

    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->second != 0);

      const IndexType offset = xit->first * nout;
      for (auto &j : output_neurons) {
        assert(xit->first < ninput());
        assert(j < nout);

        output_activationinputs[j] += xit->second * weights_[offset + j];
      }
    }
  }

  //! ======================================================================
  //! Computes the weighted sum of the error derivatives of the output layer neurons
  //! and propagates them to the input layer.
  //! input_errorinputs = weights_.transpose() * output_errors + beta * intput_errorinputs.
  //! \param beta                           controls whether to reset the activation inputs.
  //! \param output_errors                  output errors.
  //! \param input_neurons                  (Optional) active neurons.
  //! \param input_errorinputs              input errorinputs.
  //! \param output_neurons                 (Optional) active neurons.
  //! ======================================================================

  virtual void BackwardPropagate(WeightType beta,
                                 const std::vector<ErrorType> &output_errors,
                                 const std::vector<IndexType> &output_neurons,
                                 std::vector<ErrorType> &input_errorinputs) override {
    TIALNN_ERR("Error: InputMajor but only set output_neurons.");
    std::exit(EXIT_FAILURE);
  }

  virtual void BackwardPropagate(WeightType beta,
                                 const std::vector<ErrorType> &output_errors,
                                 std::vector<ErrorType> &input_errorinputs,
                                 const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_errorinputs.size()) == ninput());
    assert(static_cast<IndexType>(output_errors.size()) == noutput());
    assert(is_unique(input_neurons));

    const IndexType nout = noutput();

    const WeightType *pw = weights_.data();
    const ErrorType *py = output_errors.data();
    for (auto &i : input_neurons) {
      assert(i < ninput());

      ErrorType &x = input_errorinputs[i];
      x *= beta;

      const IndexType offset = i * nout;
      x += TIALNN_CBLAS_DOT(nout,
                            py, 1,
                            pw + offset, 1);
    }
  }

  virtual void BackwardPropagate(WeightType beta,
                                 const std::vector<ErrorType> &output_errors,
                                 const std::vector<IndexType> &output_neurons,
                                 std::vector<ErrorType> &input_errorinputs,
                                 const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_errorinputs.size()) == ninput());
    assert(static_cast<IndexType>(output_errors.size()) == noutput());
    assert(is_unique(input_neurons));
    assert(is_unique(output_neurons));

    const IndexType nout = noutput();

    for (auto &i : input_neurons) {
      assert(i < ninput());

      ErrorType &x = input_errorinputs[i];
      x *= beta;

      const IndexType offset = i * nout;
      for (auto &j : output_neurons) {
        assert(j < noutput());

        x += output_errors[j] * weights_[offset + j];
      }
    }
  }

  //! ======================================================================
  //! Updates the gradients of the connection.
  //! gradients_ += output_errors * input_activations.transpose().
  //! \param output_errors                  noutput_.
  //! \param output_neurons                 (Optional) active neurons.
  //! \param input_activations              ninput_.
  //! \param input_neurons                  (Optional) active neurons.
  //! ======================================================================

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<ActivationType> &input_activations,
                                   const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_activations.size()) == ninput());
    assert(static_cast<IndexType>(output_errors.size()) == noutput());
    assert(is_unique(input_neurons));

    const IndexType nout = noutput();

    const ErrorType *py = output_errors.data();
    for (auto &i : input_neurons) {
      assert(i < ninput());

      const ActivationType x = input_activations[i];
      if (x == 0) {
        continue;
      }
      TIALNN_CBLAS_AXPY(nout, x,
                        py, 1,
                        gradients_[i].vals.data(), 1);
      gradients_touched_.push_back(i);
    }
  }

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<IndexType> &output_neurons,
                                   const std::vector<ActivationType> &input_activations) override {
    TIALNN_ERR("Error: InputMajor but only set output_neurons.");
    std::exit(EXIT_FAILURE);
  }

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<IndexType> &output_neurons,
                                   const std::vector<ActivationType> &input_activations,
                                   const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_activations.size()) == ninput());
    assert(static_cast<IndexType>(output_errors.size()) == noutput());
    assert(is_unique(input_neurons));
    assert(is_unique(output_neurons));

    for (auto &i : input_neurons) {
      assert(i < ninput());

      const ActivationType x = input_activations[i];
      if (x == 0) {
        continue;
      }
      Gradients &grads = gradients_[i];
      for (auto &j : output_neurons) {
        assert(j < noutput());

        grads.vals[j] += x * output_errors[j];
      }
      gradients_touched_.push_back(i);
    }
  }

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::unordered_map<IndexType, ActivationType> &input_activations) override {
    assert(static_cast<IndexType>(output_errors.size()) == noutput());

    const IndexType nout = noutput();

    const ErrorType *py = output_errors.data();
    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->first < ninput());
      assert(xit->second != 0);

      TIALNN_CBLAS_AXPY(nout, xit->second,
                        py, 1,
                        gradients_[xit->first].vals.data(), 1);
      gradients_touched_.push_back(xit->first);
    }
  }

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<IndexType> &output_neurons,
                                   const std::unordered_map<IndexType, ActivationType> &input_activations) override {
    assert(static_cast<IndexType>(output_errors.size()) == noutput());
    assert(is_unique(output_neurons));

    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->first < ninput());
      assert(xit->second != 0);

      Gradients &grads = gradients_[xit->first];
      for (auto &j : output_neurons) {
        assert(j < noutput());

        grads.vals[j] += xit->second * output_errors[j];
      }
      gradients_touched_.push_back(xit->first);
    }
  }

  //! ======================================================================
  //! Batch computes the weighted sum of the activations of input layer neurons
  //! and propagates them to the output layer.
  //! output_activationinputs = weights_  * input_activations + beta * output_activationinputs.
  //! \param batchsize                      batch size.
  //! \param beta                           controls whether to reset the activation inputs.
  //! \param input_activations              input activations (can be sparse unordered_map).
  //! \param input_neurons                  (Optional) active neurons (will skip those < bx_start).
  //! \param bx_start                       batch offset of input_activations.
  //! \param output_activationinputs        output activationinputs.
  //! \param by_start                       batch offset of output_activationinputs.
  //! \param output_neurons                 (Optional) active neurons (will skip those < by_start).
  //! ======================================================================

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start) override {
    assert(static_cast<IndexType>(input_activations.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) >= (batchsize + by_start) * noutput());

    const IndexType nin = ninput();
    const IndexType nout = noutput();
    TIALNN_CBLAS_GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans,
                      nout, batchsize, nin,
                      1.0f, weights_.data(), nout,
                      input_activations.data() + bx_start * nin, nin,
                      beta, output_activationinputs.data() + by_start * nout, nout);
  }

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     const std::vector<IndexType> &input_neurons,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start) override {
    assert(static_cast<IndexType>(input_activations.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(input_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    const WeightType *pw = weights_.data();
    const ActivationType *px = input_activations.data() + bx_start * nin;
    ActivationType *py = output_activationinputs.data() + by_start * nout;

    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(nout * batchsize, beta, py, 1);
    }

    for (auto &i : input_neurons) {
      assert(i < nin);

      TIALNN_CBLAS_GER(CblasColMajor, nout, batchsize, 1.0,
                       pw + i * nout, 1,
                       px + i, nin,
                       py, nout);
    }
  }

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     std::vector<ActivationType> &output_activationinputs,
                                     const std::vector<IndexType> &output_neurons,
                                     IndexType by_start) override {
    TIALNN_ERR("Error: InputMajor but only set output_neurons.");
    std::exit(EXIT_FAILURE);
  }

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     const std::vector<IndexType> &input_neurons,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start,
                                     const std::vector<IndexType> &output_neurons) override {
    assert(static_cast<IndexType>(input_activations.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_activationinputs.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(input_neurons));
    assert(is_unique(output_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    IndexType nx_offset = bx_start * nin;
    ActivationType *py = output_activationinputs.data() + by_start * nout;

    if (beta != 1.0f) {
      for (auto &j : output_neurons) {
        TIALNN_CBLAS_SCAL(batchsize, beta, py + j, nout);
      }
    }

    for (auto &i : input_neurons) {
      assert(i < nin);

      const ActivationType *px = input_activations.data() + nx_offset + i;
      const WeightType *pw = weights_.data() + i * nout;

      for (auto &j : output_neurons) {
        assert(j < nout);

        TIALNN_CBLAS_AXPY(batchsize, *(pw + j),
                          px, nin,
                          py + j, nout);
      }
    }
  }

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::unordered_map<IndexType, ActivationType> &input_activations,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start) override {
    assert(static_cast<IndexType>(output_activationinputs.size()) >= (batchsize + by_start) * noutput());

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    const WeightType *pw = weights_.data();
    ActivationType *py = output_activationinputs.data() + by_start * nout;

    //! \note All output_activationinputs are scaled,
    //        even though some batches of them may not be included in input_activations.
    if (beta != 1.0f) {
      TIALNN_CBLAS_SCAL(nout * batchsize, beta, py, 1);
    }

    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->first < nin * batchsize);
      assert(xit->second != 0);

      const IndexType k = xit->first / nin;
      const IndexType i = xit->first - k * nin;
      TIALNN_CBLAS_AXPY(nout, xit->second,
                        pw + i * nout, 1,
                        py + k * nout, 1);
    }
  }

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::unordered_map<IndexType, ActivationType> &input_activations,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start,
                                     const std::vector<IndexType> &output_neurons) override {
    assert(static_cast<IndexType>(output_activationinputs.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(output_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    const WeightType *pw = weights_.data();
    ActivationType *py = output_activationinputs.data() + by_start * nout;

    if (beta != 1.0f) {
      //! \note All output_activationinputs specified by output_neurons are scaled,
      //        even though some batches of them may not be included in input_activations.
      for (auto &j : output_neurons) {
        TIALNN_CBLAS_SCAL(batchsize, beta, py + j, nout);
      }
    }

    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->first < nin * batchsize);
      assert(xit->second != 0);

      const IndexType k = xit->first / nin;
      const IndexType i = xit->first - k * nin;
      const IndexType ny_offset = k * nout;
      const IndexType nw_offset = i * nout;
      for (auto &j : output_neurons) {
        assert(j < nout);

        *(py + ny_offset + j) += xit->second * *(pw + nw_offset + j);
      }
    }
  }

  //! ======================================================================
  //! Batch computes the weighted sum of the error derivatives of the output layer neurons
  //! and propagates them to the input layer.
  //! input_errorinputs = weights_.transpose() * output_errors + beta * intput_errorinputs.
  //! \param batchsize                      batch size.
  //! \param beta                           controls whether to reset the activation inputs.
  //! \param output_errors                  output errors.
  //! \param input_neurons                  (Optional) active neurons (will skip those < bx_start).
  //! \param bx_start                       batch offset of input_activations.
  //! \param input_errorinputs              input errorinputs.
  //! \param by_start                       batch offset of output_activationinputs.
  //! \param output_neurons                 (Optional) active neurons (will skip those < by_start).
  //! ======================================================================

  virtual void BatchBackwardPropagate(IndexType batchsize, WeightType beta,
                                      const std::vector<ErrorType> &output_errors,
                                      IndexType by_start,
                                      const std::vector<IndexType> &output_neurons,
                                      IndexType bx_start,
                                      std::vector<ErrorType> &input_errorinputs) override {
    TIALNN_ERR("Error: InputMajor but only set output_neurons.");
    std::exit(EXIT_FAILURE);
  }

  virtual void BatchBackwardPropagate(IndexType batchsize, WeightType beta,
                                      const std::vector<ErrorType> &output_errors,
                                      IndexType by_start,
                                      std::vector<ErrorType> &input_errorinputs,
                                      IndexType bx_start,
                                      const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_errorinputs.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(input_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    const WeightType *pw = weights_.data();
    const ErrorType *py = output_errors.data() + by_start * nout;
    ErrorType *px = input_errorinputs.data() + bx_start * nin;
    for (auto &i : input_neurons) {
      assert(i < nin);

      TIALNN_CBLAS_GEMV(CblasColMajor, CblasTrans,
                        nout, batchsize,
                        1.0, py, nout,
                        pw + i * nout, 1,
                        beta, px + i, nin);
    }
  }

  virtual void BatchBackwardPropagate(IndexType batchsize, WeightType beta,
                                      const std::vector<ErrorType> &output_errors,
                                      IndexType by_start,
                                      const std::vector<IndexType> &output_neurons,
                                      std::vector<ErrorType> &input_errorinputs,
                                      IndexType bx_start,
                                      const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_errorinputs.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(input_neurons));
    assert(is_unique(output_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    IndexType nx_offset = bx_start * nin;
    const ErrorType *py = output_errors.data() + by_start * nout;
    for (auto &i : input_neurons) {
      assert(i < nin);

      ErrorType *px = input_errorinputs.data() + nx_offset + i;
      const WeightType *pw = weights_.data() + i * nout;

      if (beta != 1.0f) {
        TIALNN_CBLAS_SCAL(batchsize, beta, px, nin);
      }

      for (auto &j : output_neurons) {
        assert(j < nout);

        TIALNN_CBLAS_AXPY(batchsize, *(pw + j),
                          py + j, nout,
                          px, nin);
      }
    }
  }

  //! ======================================================================
  //! Batch updates the gradients of the connection.
  //! gradients_ += output_errors * input_activations.transpose().
  //! \param batchsize                      batch size.
  //! \param output_errors                  noutput_.
  //! \param by_start                       batch offset of output_activationinputs.
  //! \param output_neurons                 (Optional) active neurons (will skip those < by_start).
  //! \param input_activations              ninput_.
  //! \param bx_start                       batch offset of input_activations.
  //! \param input_neurons                  (Optional) active neurons (will skip those < bx_start).
  //! ======================================================================

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<ActivationType> &input_activations,
                                        IndexType bx_start,
                                        const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_activations.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(input_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    const ActivationType *px = input_activations.data() + bx_start * nin;
    const ErrorType *py = output_errors.data() + by_start * nout;
    for (auto &i : input_neurons) {
      assert(i < nin);

      TIALNN_CBLAS_GEMV(CblasColMajor, CblasNoTrans,
                        nout, batchsize,
                        1.0, py, nout,
                        px + i, nin,
                        1.0, gradients_[i].vals.data(), 1);

      gradients_touched_.push_back(i);
    }
  }

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<IndexType> &output_neurons,
                                        const std::vector<ActivationType> &input_activations,
                                        IndexType bx_start) override {
    TIALNN_ERR("Error: InputMajor but only set output_neurons.");
    std::exit(EXIT_FAILURE);

  }

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<IndexType> &output_neurons,
                                        const std::vector<ActivationType> &input_activations,
                                        IndexType bx_start,
                                        const std::vector<IndexType> &input_neurons) override {
    assert(static_cast<IndexType>(input_activations.size()) >= (batchsize + bx_start) * ninput());
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(input_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    IndexType nx_offset = bx_start * nin;
    const ErrorType *py = output_errors.data() + by_start * nout;
    for (auto &i : input_neurons) {
      assert(i < nin);

      const ActivationType *px = input_activations.data() + nx_offset + i;
      Gradients &grads = gradients_[i];
      for (auto &j : output_neurons) {
        assert(j < nout);

        grads.vals[j] += TIALNN_CBLAS_DOT(batchsize,
                                          py + j, nout,
                                          px, nin);
      }

      gradients_touched_.push_back(i);
    }
  }

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::unordered_map<IndexType, ActivationType> &input_activations) override {
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    const ErrorType *py = output_errors.data() + by_start * nout;
    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->first < batchsize * nin);
      assert(xit->second != 0);

      const IndexType k = xit->first / nin;
      const IndexType i = xit->first - k * nin;
      TIALNN_CBLAS_AXPY(nout, xit->second,
                        py + k * nout, 1, //<! \note py needs to be moved to current batch.
                        gradients_[i].vals.data(), 1);

      gradients_touched_.push_back(i);
    }
  }

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<IndexType> &output_neurons,
                                        const std::unordered_map<IndexType, ActivationType> &input_activations) override {
    assert(static_cast<IndexType>(output_errors.size()) >= (batchsize + by_start) * noutput());
    assert(is_unique(output_neurons));

    const IndexType nin = ninput();
    const IndexType nout = noutput();

    IndexType ny_offset = by_start * nout;
    for (std::unordered_map<IndexType, ActivationType>::const_iterator xit = input_activations.begin();
         xit != input_activations.end(); ++xit) {
      assert(xit->first < batchsize * nin);
      assert(xit->second != 0);

      const IndexType k = xit->first / nin;
      const IndexType i = xit->first - k * nin;
      const ErrorType *py = output_errors.data() + ny_offset + k * nout; //<! \note py needs to be moved to current batch.
      Gradients &grads = gradients_[i];
      for (auto &j : output_neurons) {
        assert(j < nout);

        grads.vals[j] += xit->second * *(py + j);
      }

      gradients_touched_.push_back(i);
    }
  }

  //! Updates the weights of the touched rows.
  virtual void UpdateWeights(GradientType learning_rate) override {
    //! Process the gradients_touched_.
    std::sort(gradients_touched_.begin(), gradients_touched_.end());
    std::vector<IndexType>::iterator last = unique(gradients_touched_.begin(), gradients_touched_.end());
    gradients_touched_.erase(last, gradients_touched_.end());

    WeightType *pw = weights_.data();
    IndexType nout = noutput();
    for (auto &i : gradients_touched_) {
      assert(i < ninput());
      gradients_[i].UpdateWeights(learning_rate,  pw + i * nout);
    }

    gradients_touched_.clear();
  }

  //! Caches the weights.
  //! \note usually called at the end of an epoch, not an iteration.
  virtual void CacheCurrentWeights() override {
    last_weights_ = weights_; //!< should not use swap as we do not want to overwrite weights_.
    for (auto &grad : gradients_) {
      grad.CacheCurrentGradients();
    }
  }

  //! Restores the weights.
  //! \note if needed, usually called at the end of an epoch, not an iteration.
  virtual void RestoreLastWeights() override {
    weights_ = last_weights_; //!< should not use swap as we do not want to overwrite last_weights_.
    for (auto &grad : gradients_) {
      grad.RestoreLastGradients();
    }
  }

 private:
  //! Allocates the connection.
  virtual void AllocateConnection() override {
    assert(weights_.empty());
    assert(last_weights_.empty());
    assert(gradients_.empty());

    IndexType n = ninput() * noutput();

    weights_.resize(n);
    last_weights_.resize(n);
    gradients_.resize(ninput());
    for (auto &grads : gradients_) {
      assert(grads.vals.empty());

      grads.AllocateGradients(noutput());
    }

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
  //! Gradients of the weights for each input.
  std::vector<Gradients> gradients_;
  //! Touched rows (for inputs) of gradients_.
  std::vector<IndexType> gradients_touched_;
};

} // namespace tialnn

#endif
