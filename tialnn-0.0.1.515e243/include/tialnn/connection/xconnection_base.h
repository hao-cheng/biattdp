/*!
 * \file xconnection_base.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_CONNECTION_XCONNECTION_BASE_H_
#define TIALNN_CONNECTION_XCONNECTION_BASE_H_

// std::cout
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// std::vector
#include <vector>
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

//! Base class of neural network X-connections.
//! \note For an X-connection, all weights in the connection are used in one
//! propagation, and therefore, no need to keep track of which ones are used.
class XConnectionBase : public ConnectionBase {

 public:
  //! Constructor.
  explicit XConnectionBase() {}
  //! Destructor.
  virtual ~XConnectionBase() {}

  //! Computes the weighted sum of the activations of input layer neurons and
  //! propagates them to the output layer.
  //! output_activationinputs = weights_ * input_activations + beta * output_activationinputs.
  //! \param beta                           controls whether to reset the activation inputs.
  //! \param input_activations              input activations.
  //! \param output_activationinputs        output activationinputs.
  virtual void ForwardPropagate(WeightType beta,
                                const std::vector<ActivationType> &input_activations,
                                std::vector<ActivationType> &output_activationinputs) = 0;

  //! Computes the weighted sum of the error derivatives of the output layer neurons and
  //! propagates them to the input layer.
  //! input_errorinputs = weights_.transpose() * output_errors + beta * intput_errorinputs.
  //! \param beta                           controls whether to reset the error inputs.
  //! \param output_errors                  output errors.
  //! \param input_errorinputs              input errorinputs.
  virtual void BackwardPropagate(WeightType beta,
                                 const std::vector<ErrorType> &output_errors,
                                 std::vector<ErrorType> &input_errorinputs) = 0;

  //! Updates the gradients of the connection.
  //! gradients_ += output_errors * input_activations.transpose().
  //! \param output_errors                  output errors.
  //! \param input_errorinputs              input errorinputs.
  virtual void AccumulateGradients(const std::vector<ErrorType> &output_errors,
                                   const std::vector<ActivationType> &input_activations) = 0;

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
                                     IndexType by_start) = 0;

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
                                      IndexType bx_start) = 0;

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
                                        IndexType bx_start) = 0;
};

} // namespace tialnn

#endif
