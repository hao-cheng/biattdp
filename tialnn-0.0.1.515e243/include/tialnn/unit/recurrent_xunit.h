/*!
 * \file recurrent_xunit.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_UNIT_RECURRENT_XUNIT_H_
#define TIALNN_UNIT_RECURRENT_XUNIT_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::vector
#include <vector>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
#include "../base.h"
// tialnn::XConnectionBase
#include "../connection/xconnection_base.h"
// tialnn::YConnectionBase
#include "../connection/yconnection_base.h"
// tialnn::GeneircNeurons
#include "../neuron/generic_neurons.h"
// tialnn::IdentityNeurons
#include "../neuron/identity_neurons.h"
// tialnn::XBatchLayer
#include "../layer/xbatchlayer.h"
// tialnn::BatchLayerBase
#include "../layer/batchlayer_base.h"

namespace tialnn {

//! Vanilla recurrent x-unit.
//!
//! Be careful when using the unit, specifically, the batch idx.
//! - The unit stores the batch layers for every timestep.
//! - Each block of batch layers for a timestep is an atom.
//! - batchsize_ = atom_batchsize_ * capacity_.
template <class Neurons>
class RecurrentXUnit : public BatchLayerBase {
 public:
  //! Constructor.
  explicit RecurrentXUnit() :
      atom_batchsize_(0), capacity_(0),
      ptr_conn_hprev_hidden_(nullptr) {}
  //! Destructor.
  virtual ~RecurrentXUnit() {}

  //! Returns the batch layer type.
  virtual const BatchLayerType type() const override { return xbatchlayer; }
  //! Returns the activations of all hidden neurons.
  virtual const std::vector<ActivationType>& activations() const override {
    return hidden_layer_.activations();
  }
  //! Returns the errors of of all neurons.
  virtual const std::vector<ErrorType>& errors() const override {
    TIALNN_ERR("Error: Invalid method for RecurrentXUnit");
    std::exit(EXIT_FAILURE);
    return dummy_errors_;
  }
  //! Returns the active neurons.
  virtual const std::vector<IndexType>& active_neurons() const override {
    TIALNN_ERR("Error: RecurrentXUnit does not have active_neurons");
    std::exit(EXIT_FAILURE);
    return dummy_active_neurons_;
  }

  //! Returns the atom batch size.
  IndexType atom_batchsize() const { return atom_batchsize_; }
  //! Returns the capacity,
  IndexType capacity() const { return capacity_; }

#ifdef _TIALNN_DEBUG
  //! Sets the ac state of neurons in [b_start, b_start + bs).
  //! \param s    true: in ac_states; false: not in ac_states.
  virtual void set_ac_states(IndexType b_start, IndexType bs, bool s) override {
    hidden_layer_.set_ac_states(b_start, bs, s);
  }
  //! Sets the er state of neurons in [b_start, b_start + bs).
  //! \param s    true: in er_states; false: not in er_states.
  virtual void set_er_states(IndexType b_start, IndexType bs, bool s) override {
    hidden_layer_.set_er_states(b_start, bs, s);
  }

  //! Check the ac state of neurons in [b_start, b_start + bs).
  //! \param s    true: in ac_states; false: not in ac_states.
  virtual bool CheckAcStates(IndexType b_start, IndexType bs, bool s) const override {
    return hidden_layer_.CheckAcStates(b_start, bs, s);
  }
  //! Check the er state of neurons in [b_start, b_start + bs).
  //! \param s    true: in er_states; false: not in er_states.
  //! \param flip true if flip the er_state.
  virtual bool CheckErStates(IndexType b_start, IndexType bs, bool s) const override {
    return hidden_layer_.CheckErStates(b_start, bs, s);
  }
#endif

  //! Sets the atom bath size.
  void set_atom_batchsize(IndexType bs) { atom_batchsize_ = bs; }
  //! Sets the capacity.
  void set_capacity(IndexType c) { capacity_ = c; }

  //! Sets the ptr_conn_hprev_hidden_.
  void set_ptr_conn_hprev_hidden(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_hidden_);
    ptr_conn_hprev_hidden_ = pconn;
  }

  //! Sets input of all errors in [b_start, b_start + bs) to value.
  //! \note If val = 0, consider to use {X,Y}BackwardPropagateR when possible.
  void SetInputOfErrorsToValue(IndexType b_start, IndexType bs, ActivationType val) {
    hidden_layer_.SetInputOfErrorsToValue(b_start, bs, val);
  }

  //! Computes activations of neurons for [0, atom_batchsize_) according to their inputs.
  void ComputeActivations(const XBatchLayer<IdentityNeurons> &init_hidden_layer) {
    assert(init_hidden_layer.nneurons() == nneurons());
    assert(init_hidden_layer.batchsize() == atom_batchsize_);
    assert(ptr_conn_hprev_hidden_);

    hidden_layer_.XForwardPropagateA(init_hidden_layer, 0,
                                     *ptr_conn_hprev_hidden_,
                                     0, atom_batchsize_);
    hidden_layer_.ComputeActivations(0, atom_batchsize_);
  }

  //! Computes errors of neurons for [0, atom_batchsize_) according to their inputs.
  void ComputeErrors(const XBatchLayer<IdentityNeurons> &init_hidden_layer) {
    assert(init_hidden_layer.nneurons() == nneurons());
    assert(init_hidden_layer.batchsize() == atom_batchsize_);
    assert(ptr_conn_hprev_hidden_);

    hidden_layer_.ComputeErrors(0, atom_batchsize_);
    hidden_layer_.XBackwardPropagateNA(init_hidden_layer, 0,
                                       *ptr_conn_hprev_hidden_,
                                       0, atom_batchsize_);
  }


  //! Computes activations of neurons for [t, (t + 1) * atom_batchsize_) according to their inputs.
  //! t > 0
  void ComputeActivations(IndexType t) {
    assert(t > 0);
    assert(t < capacity_);
    assert(ptr_conn_hprev_hidden_);

    IndexType prev_b_start = (t - 1) * atom_batchsize_;
    IndexType b_start = t * atom_batchsize_;

    hidden_layer_.XForwardPropagateA(hidden_layer_, prev_b_start,
                                     *ptr_conn_hprev_hidden_,
                                     b_start, atom_batchsize_);
    hidden_layer_.ComputeActivations(b_start, atom_batchsize_);
  }

  //! Computes errors of neurons for [t * atom_batchsize_, (t + 1) * atom_batchsize_) according to their inputs.
  void ComputeErrors(IndexType t) {
    assert(t > 0);
    assert(t < capacity_);
    assert(ptr_conn_hprev_hidden_);

    IndexType prev_b_start = (t - 1) * atom_batchsize_;
    IndexType b_start = t * atom_batchsize_;

    hidden_layer_.ComputeErrors(b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateAA(hidden_layer_, prev_b_start,
                                       *ptr_conn_hprev_hidden_,
                                       b_start, atom_batchsize_);
  }

  //! Forward propagates
  //! from all input neurons in [bx_start, bx_start + bs)
  //! to all output neurons in [by_start, by_start + bs).
  //! - Resets and accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardPropagateR(const BatchLayerBase &input_layer,
                          IndexType bx_start,
                          XConnectionBase &conn,
                          IndexType by_start, IndexType bs) {
    assert(input_layer.type() == xbatchlayer);

    hidden_layer_.XForwardPropagateR(input_layer, bx_start,
                                     conn,
                                     by_start, bs);
  }

  //! Forward propagates
  //! from all input neurons in [bx_start, bx_start + bs)
  //! to all output neurons in [by_start, by_start + bs).
  //! - Not resets but accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardPropagateA(const BatchLayerBase &input_layer,
                          IndexType bx_start,
                          XConnectionBase &conn,
                          IndexType by_start, IndexType bs) {
    assert(input_layer.type() == xbatchlayer);

    hidden_layer_.XForwardPropagateA(input_layer, bx_start,
                                     conn,
                                     by_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs).
  //! - Resets and accumulates error inputs of input layer.
  //! - Accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateRA(BatchLayerBase &input_layer,
                            IndexType bx_start,
                            XConnectionBase &conn,
                            IndexType by_start, IndexType bs) const {
    assert(input_layer.type() == xbatchlayer);

    hidden_layer_.XBackwardPropagateRA(input_layer, bx_start,
                                       conn,
                                       by_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs).
  //! - Not resets but accumulates error inputs of input layer.
  //! - Accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateAA(BatchLayerBase &input_layer,
                            IndexType bx_start,
                            XConnectionBase &conn,
                            IndexType by_start, IndexType bs) const {
    assert(input_layer.type() == xbatchlayer);

    hidden_layer_.XBackwardPropagateAA(input_layer, bx_start,
                                       conn,
                                       by_start, bs);
  }

  //!=========================================
  //! Virtual functions
  //! {X,Y}BackwardPropagate{R,A,N}{A,N}From
  //! {X,Y}BackwardPropagate{R,A}From
  //!=========================================

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to all hidden neurons in [bx_start, bx_start + bs).
  //! - Resets and accumulates error inputs.
  //! - Accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateRAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) override {
    hidden_layer_.XBackwardPropagateRAFrom(output_errors, by_start,
                                           conn,
                                           bx_start, bs);
  }

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to all hidden neurons in [bx_start, bx_start + bs).
  //! - Not resets but accumulates error inputs.
  //! - Accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateAAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) override {
    hidden_layer_.XBackwardPropagateAAFrom(output_errors, by_start,
                                           conn,
                                           bx_start, bs);
  }

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to all hidden neurons in [bx_start, bx_start + bs).
  //! - Not accumulates error inputs.
  //! - Accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateNAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) const override {
    hidden_layer_.XBackwardPropagateNAFrom(output_errors, by_start,
                                           conn,
                                           bx_start, bs);
  }

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to all hidden neurons in [bx_start, bx_start + bs).
  //! - Resets and accumulates error inputs.
  //! - Not accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateRNFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) override {
    hidden_layer_.XBackwardPropagateRNFrom(output_errors, by_start,
                                           conn,
                                           bx_start, bs);
  }

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + bs)
  //! to all hidden neurons in [bx_start, bx_start + bs).
  //! - Not resets but accumulates error inputs.
  //! - Not accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateANFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        XConnectionBase &conn,
                                        IndexType bx_start, IndexType bs) override {
    hidden_layer_.XBackwardPropagateANFrom(output_errors, by_start,
                                           conn,
                                           bx_start, bs);
  }

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + ninput)
  //! to all hidden neurons in [bx1_start, bx1_start + ninput).
  //! - Resets and accumulates error inputs.
  //! neurons_.errorinputs = output_layer.errors() .* input_layer.activations().
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateRFrom(const std::vector<ErrorType> output_errors,
                                       IndexType by_start,
                                       const std::vector<ActivationType> &input_activations,
                                       IndexType bx2_start, IndexType noutput,
                                       IndexType bx1_start, IndexType bs) override {
    TIALNN_ERR("Error: experimental methods (RecurrentXUnit)");
    std::exit(EXIT_FAILURE);
    hidden_layer_.XBackwardPropagateRFrom(output_errors, by_start,
                                          input_activations, bx2_start, noutput,
                                          bx1_start, bs);
  }

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + ninput)
  //! to all hidden neurons in [bx1_start, bx1_start + ninput).
  //! - Not resets but accumulates error inputs.
  //! neurons_.errorinputs += output_layer.errors() .* input_layer.activations().
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardPropagateAFrom(const std::vector<ErrorType> output_errors,
                                       IndexType by_start,
                                       const std::vector<ActivationType> &input_activations,
                                       IndexType bx2_start, IndexType noutput,
                                       IndexType bx1_start, IndexType bs) override {
    TIALNN_ERR("Error: experimental methods (RecurrentXUnit)");
    std::exit(EXIT_FAILURE);
    hidden_layer_.XBackwardPropagateAFrom(output_errors, by_start,
                                          input_activations, bx2_start, noutput,
                                          bx1_start, bs);
  }

  //! Backward propagates (broadcast connection)
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs)
  //! with a broadcasted scaler connection.
  //! - Resets and accumulates error inputs.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardBPropagateRFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        WeightType alpha, IndexType noutput,
                                        IndexType bx_start, IndexType bs) override {
    hidden_layer_.XBackwardBPropagateRFrom(output_errors, by_start,
                                           alpha, noutput,
                                           bx_start, bs);
  }

  //! Backward propagates (broadcast connection)
  //! from all output neurons in [by_start, by_start + bs)
  //! to {all,active} hidden neurons in [bx_start, bx_start + bs)
  //! with a broadcasted scaler connection.
  //! - Not resets but accumulates error inputs.
  //! \note X- means output_layer is XBatchLayer.
  virtual void XBackwardBPropagateAFrom(const std::vector<ErrorType> &output_errors,
                                        IndexType by_start,
                                        WeightType alpha, IndexType noutput,
                                        IndexType bx_start, IndexType bs) override {
    hidden_layer_.XBackwardBPropagateAFrom(output_errors, by_start,
                                           alpha, noutput,
                                           bx_start, bs);
  }

 private:
  //! Allocates the layer.
  virtual void AllocateLayer() override {
    if (nneurons() == 0) {
      TIALNN_ERR("nneurons_ should be greater than 0!");
      std::exit(EXIT_FAILURE);
    }
    if (batchsize() == 0) {
      TIALNN_ERR("batchsize_ should be greater than 0!");
      std::exit(EXIT_FAILURE);
    }
    if (batchsize() != atom_batchsize_ * capacity_) {
      TIALNN_ERR("batchsize_ should be equal to atom_batchsize_ * capacity_!");
      std::exit(EXIT_FAILURE);
    }

    hidden_layer_.set_nneurons_batchsize(nneurons(), batchsize());
  };

  //! Atom batch size.
  IndexType atom_batchsize_;
  //! Capacity.
  IndexType capacity_;

  //! Pointer to the recurrent connections.
  XConnectionBase *ptr_conn_hprev_hidden_;

  //! Hidden layer.
  XBatchLayer<Neurons> hidden_layer_;

  //! Dummy variable for BatchLayerBase interface.
  std::vector<ErrorType> dummy_errors_;
  std::vector<IndexType> dummy_active_neurons_;
};

} // namespace tialnn

#endif
