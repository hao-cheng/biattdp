/*!
 * \file batchlayer_base.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_LAYER_BATCHLAYER_BASE_H_
#define TIALNN_LAYER_BATCHLAYER_BASE_H_

// std::vector
#include <vector>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
#include "../base.h"
// tialnn::XConnectionBase
#include "../connection/xconnection_base.h"

namespace tialnn {

//! Base class of neural network batch layers.
//!
//! For batch layer, neurons are stored in neuron-major format.
//! dimension: (nh, bs)
//! \note For dimension (x, y), x changes more frequent than y.
//!
//! [ (:, b = 0) b0_h0, b0_h1, b0_h2, ...,
//!   (:, b = 1) b1_h0, b1_h1, b1_h2, ...,
//!  ... ]
//!
//! \note It only defines the common interfaces for XBatchLayer and YBatchLayer.
//! Actual activations, errors, and propagation will be declared in XBatchLayer and YBatchLayer.
class BatchLayerBase {
 public:
  //! Constructor.
  explicit BatchLayerBase() : nneurons_(0), batchsize_(0) {}
  //! Destructor.
  virtual ~BatchLayerBase() {}

  //! Batch layer types.
  enum BatchLayerType {
    //! XBatchLayer: all neurons are active all the time.
    xbatchlayer,
    //! YBatchLayer: can propagate a subset of neurons.
    ybatchlayer
  };

  //! Returns the number of neurons.
  IndexType nneurons() const { return nneurons_; }
  //! Returns the batch size.
  IndexType batchsize() const { return batchsize_; }

  //! Returns the batch layer type.
  virtual const BatchLayerType type() const = 0;
  //! Returns the activations of all neurons.
  virtual const std::vector<ActivationType>& activations() const = 0;
  //! Returns the errors of all neurons.
  virtual const std::vector<ErrorType>& errors() const = 0;
  //! Returns the active neurons.
  virtual const std::vector<IndexType>& active_neurons() const = 0;


  //! Sets the number of neurons and batch size.
  //! Automatically calls AllocateLayer().
  void set_nneurons_batchsize(IndexType n, IndexType b) {
    assert(nneurons_ == 0);
    assert(batchsize_ == 0);

    nneurons_ = n;
    batchsize_ = b;
    AllocateLayer();
  }

#ifdef _TIALNN_DEBUG
  //! Sets the ac state of neurons in [b_start, b_start + bs).
  //! \param s    true: in ac_states; false: not in ac_states.
  virtual void set_ac_states(IndexType b_start, IndexType bs, bool s) = 0;
  //! Sets the er state of neurons in [b_start, b_start + bs).
  //! \param s    true: in er_states; false: not in er_states.
  virtual void set_er_states(IndexType b_start, IndexType bs, bool s) = 0;
  //! Check the ac state of neurons in [b_start, b_start + bs).
  //! \param s    true: in ac_states; false: not in ac_states.
  virtual bool CheckAcStates(IndexType b_start, IndexType bs, bool s) const = 0;
  //! Check the er state of neurons in [b_start, b_start + bs).
  //! \param s    true: in er_states; false: not in er_states.
  virtual bool CheckErStates(IndexType b_start, IndexType bs, bool s) const = 0;
#endif

  //!=========================================
  //! {X,Y}BackwardPropagate{R,A,N}{A,N}From
  //! - R,A,N: for error inputs
  //!   * R: resets and accumulates
  //!   * A: not resets but accumulates
  //!   * N: not accumulates
  //! - R,A: for connection gradients
  //!   * A: accumulates
  //!   * N: not accumulates
  //! {X,Y}BackwardPropagate{R,A}From
  //! - R,A: for error inputs
  //!   * R: resets and accumulates
  //!   * A: not resets but accumulates
  //!=========================================

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs).
  //! - Resets and accumulates error inputs.
  //! - Accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateRAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) = 0;

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs).
  //! - Not resets but accumulates error inputs.
  //! - Accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateAAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) = 0;

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs).
  //! - Not accumulates error inputs.
  //! - Accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateNAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) const = 0;

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs).
  //! - Resets and accumulates error inputs.
  //! - Not accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateRNFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) = 0;

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs).
  //! - Not resets but accumulates error inputs.
  //! - Not accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateANFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) = 0;

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + ninput)
  //! to {all,active} hidden neurons in [bx1_start, bx1_start + ninput).
  //! - Resets and accumulates error inputs.
  //! neurons_.errorinputs = output_layer.errors() .* input_layer.activations().
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateRFrom(const std::vector<ErrorType> output_errors,
                                       IndexType by_start,
                                       const std::vector<ActivationType> &input_activations,
                                       IndexType bx2_start, IndexType noutput,
                                       IndexType bx1_start, IndexType bs) = 0;

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + ninput)
  //! to {all,active} hidden neurons in [bx1_start, bx1_start + ninput).
  //! - Not resets but accumulates error inputs.
  //! neurons_.errorinputs += output_layer.errors() .* input_layer.activations().
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateAFrom(const std::vector<ErrorType> output_errors,
                                       IndexType by_start,
                                       const std::vector<ActivationType> &input_activations,
                                       IndexType bx2_start, IndexType noutput,
                                       IndexType bx1_start, IndexType bs) = 0;

  //! Backward propagates (broadcast connection)
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs)
  //! with a broadcasted scaler connection.
  //! - Resets and accumulates error inputs.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardBPropagateRFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        WeightType alpha, IndexType noutput,
                                        IndexType bx_start, IndexType bs) = 0;

  //! Backward propagates (broadcast connection)
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs)
  //! with a broadcasted scaler connection.
  //! - Not resets but accumulates error inputs.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardBPropagateAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        WeightType alpha, IndexType noutput,
                                        IndexType bx_start, IndexType bs) = 0;

 private:
  //! Allocates the layer.
  virtual void AllocateLayer() = 0;

  //! Number of neurons.
  IndexType nneurons_;
  //! Batch size.
  IndexType batchsize_;
};

} // namespace tialnn

#endif
