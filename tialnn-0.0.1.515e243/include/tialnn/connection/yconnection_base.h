/*!
 * \file yconnection_base.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_CONNECTION_YCONNECTION_BASE_H_
#define TIALNN_CONNECTION_YCONNECTION_BASE_H_

// std::cout
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// std::vector
#include <vector>
// std::unordered_map
#include <unordered_map>
// std::sort
// std::unique
#include <algorithm>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
#include "../base.h"
// tialnn::write_xxx
// tialnn::read_xxx
#include "../util/futil.h"
// tialnn::ConnectionBase
#include "connection_base.h"

namespace tialnn {

//! Checks whether the vector is unique.
//! \note should use copy not reference.
inline bool is_unique(std::vector<IndexType> v) {
  std::sort(v.begin(), v.end());
  std::vector<IndexType>::iterator last = std::unique(v.begin(), v.end());
  return last == v.end();
}

//! Base class of neural network Y-connections.
//! \note For a Y-connection, only a subset of weights in the connection are used in one
//! propagation, and therefore, we can use use fast update trick.
class YConnectionBase : public ConnectionBase {

 public:
  //! Constructor.
  explicit YConnectionBase() {}
  //! Destructor.
  virtual ~YConnectionBase() {}

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
                                std::vector<ActivationType> &output_activationinputs) = 0;

  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                const std::vector<IndexType> &input_neurons,
                                std::vector<ActivationType> &output_activationinputs) = 0;

  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs,
                                const std::vector<IndexType> &output_neurons) = 0;

  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                const std::vector<IndexType> &input_neurons,
                                std::vector<ActivationType> &output_activationinputs,
                                const std::vector<IndexType> &output_neurons) = 0;

  virtual void ForwardPropagate(WeightType beta,
                                const std::unordered_map<IndexType, ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs) = 0;

  virtual void ForwardPropagate(WeightType beta,
                                const std::unordered_map<IndexType, ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs,
                                const std::vector<IndexType> &output_neurons) = 0;

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
                                 std::vector<ErrorType> &input_errorinputs) = 0;

  virtual void BackwardPropagate(WeightType beta,
                                 const std::vector<ErrorType> &output_errors,
                                 std::vector<ErrorType> &input_errorinputs,
                                 const std::vector<IndexType> &input_neurons) = 0;

  virtual void BackwardPropagate(WeightType beta,
                                 const std::vector<ErrorType> &output_errors,
                                 const std::vector<IndexType> &output_neurons,
                                 std::vector<ErrorType> &input_errorinputs,
                                 const std::vector<IndexType> &input_neurons) = 0;

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
                                   const std::vector<IndexType> &input_neurons) = 0;

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<IndexType> &output_neurons,
                                   const std::vector<ActivationType> &input_activations) = 0;

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<IndexType> &output_neurons,
                                   const std::vector<ActivationType> &input_activations,
                                   const std::vector<IndexType> &input_neurons) = 0;

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::unordered_map<IndexType, ActivationType> &input_activations) = 0;

  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<IndexType> &output_neurons,
                                   const std::unordered_map<IndexType, ActivationType> &input_activations) = 0;

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
                                     IndexType by_start) = 0;

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     const std::vector<IndexType> &input_neurons,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start) = 0;

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     std::vector<ActivationType> &output_activationinputs,
                                     const std::vector<IndexType> &output_neurons,
                                     IndexType by_start) = 0;

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::vector<ActivationType> &input_activations,
                                     IndexType bx_start,
                                     const std::vector<IndexType> &input_neurons,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start,
                                     const std::vector<IndexType> &output_neurons) = 0;

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::unordered_map<IndexType, ActivationType> &input_activations,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start) = 0;

  virtual void BatchForwardPropagate(IndexType batchsize, WeightType beta,
                                     const std::unordered_map<IndexType, ActivationType> &input_activations,
                                     std::vector<ActivationType> &output_activationinputs,
                                     IndexType by_start,
                                     const std::vector<IndexType> &output_neurons) = 0;

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
                                      std::vector<ErrorType> &input_errorinputs) = 0;

  virtual void BatchBackwardPropagate(IndexType batchsize, WeightType beta,
                                      const std::vector<ErrorType> &output_errors,
                                      IndexType by_start,
                                      std::vector<ErrorType> &input_errorinputs,
                                      IndexType bx_start,
                                      const std::vector<IndexType> &input_neurons) = 0;

  virtual void BatchBackwardPropagate(IndexType batchsize, WeightType beta,
                                      const std::vector<ErrorType> &output_errors,
                                      IndexType by_start,
                                      const std::vector<IndexType> &output_neurons,
                                      std::vector<ErrorType> &input_errorinputs,
                                      IndexType bx_start,
                                      const std::vector<IndexType> &input_neurons) = 0;

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
                                        const std::vector<IndexType> &input_neurons) = 0;

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<IndexType> &output_neurons,
                                        const std::vector<ActivationType> &input_activations,
                                        IndexType bx_start) = 0;

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<IndexType> &output_neurons,
                                        const std::vector<ActivationType> &input_activations,
                                        IndexType bx_start,
                                        const std::vector<IndexType> &input_neurons) = 0;

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::unordered_map<IndexType, ActivationType> &input_activations) = 0;

  virtual void BatchAccumulateGradients(IndexType batchsize,
                                        const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        const std::vector<IndexType> &output_neurons,
                                        const std::unordered_map<IndexType, ActivationType> &input_activations) = 0;
};

} // namespace tialnn

#endif
