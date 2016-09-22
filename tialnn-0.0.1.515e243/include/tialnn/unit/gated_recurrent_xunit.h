/*!
 * \file gated_recurrent_xunit.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_UNIT_GATED_RECURRENT_XUNIT_H_
#define TIALNN_UNIT_GATED_RECURRENT_XUNIT_H_

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
// tialnn::tanh_f
// tialnn::tanh_g
// tialnn::sigmoid_f
// tialnn::sigmoid_g
#include "../util/numeric.h"
// tialnn::GeneircNeurons
#include "../neuron/generic_neurons.h"
// tialnn::IdentityNeurons
#include "../neuron/identity_neurons.h"
// tialnn::XBatchLayer
#include "../layer/xbatchlayer.h"
// tialnn::BatchLayerBase
#include "../layer/batchlayer_base.h"

namespace tialnn {

//! Gated recurrent x-unit.
//! See \cite{Cho2014EMNLP}.
//!
//! Be careful when using the unit, specifically, the batch idx.
//! - The unit stores the batch layers for every timestep.
//! - Each block of batch layers for a timestep is an atom.
//! - batchsize_ = atom_batchsize_ * capacity_.
template <class Neuron>
class GatedRecurrentXUnit : public BatchLayerBase {
 public:
  //! Constructor.
  explicit GatedRecurrentXUnit() :
      atom_batchsize_(0), capacity_(0),
      ptr_negones_(nullptr),
      ptr_bias_layer_(nullptr),
      ptr_conn_hprev_reset_(nullptr),
      ptr_conn_hprev_update_(nullptr),
      ptr_conn_hprev_htilde_(nullptr),
      ptr_conn_bias_reset_(nullptr),
      ptr_conn_bias_update_(nullptr),
      ptr_conn_bias_htilde_(nullptr),
      ptr_conn_globalbias_reset_(nullptr),
      ptr_conn_globalbias_update_(nullptr),
      ptr_conn_globalbias_htilde_(nullptr) {}
  //! Destructor.
  virtual ~GatedRecurrentXUnit() {}

  //! Returns the batch layer type.
  virtual const BatchLayerType type() const override { return xbatchlayer; }
  //! Returns the activations of all hidden neurons.
  virtual const std::vector<ActivationType>& activations() const override {
    return hidden_layer_.activations();
  }
  //! Returns the errors of of all neurons.
  virtual const std::vector<ErrorType>& errors() const override {
    TIALNN_ERR("Error: Invalid method for GatedRecurrentXUnit");
    std::exit(EXIT_FAILURE);
    return dummy_errors_;
  }
  //! Returns the active neurons.
  virtual const std::vector<IndexType>& active_neurons() const override {
    TIALNN_ERR("Error: GatedRecurrentXUnit does not have active_neurons");
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

  //! Sets the ptr_negones_.
  void set_ptr_negones(const XBatchLayer<IdentityNeurons> *pn) {
    assert(!ptr_negones_);
    ptr_negones_ = pn;
  }
  //! Sets the ptr_conn_hprev_reset_.
  void set_ptr_conn_hprev_reset(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_reset_);
    ptr_conn_hprev_reset_ = pconn;
  }
  //! Sets the ptr_conn_hprev_update_.
  void set_ptr_conn_hprev_update(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_update_);
    ptr_conn_hprev_update_ = pconn;
  }
  //! Sets the ptr_conn_hprev_htilde_.
  void set_ptr_conn_hprev_htilde(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_htilde_);
    ptr_conn_hprev_htilde_ = pconn;
  }
  //! Sets the ptr_bias_layer_.
  void set_ptr_bias_layer(const XBatchLayer<IdentityNeurons> *pb) {
    assert(!ptr_bias_layer_);
    ptr_bias_layer_ = pb;
  }
  //! Sets the ptr_conn_bias_reset_.
  void set_ptr_conn_bias_reset(XConnectionBase *pconn) {
    assert(!ptr_conn_bias_reset_);
    ptr_conn_bias_reset_ = pconn;
  }
  //! Sets the ptr_conn_bias_update_.
  void set_ptr_conn_bias_update(XConnectionBase *pconn) {
    assert(!ptr_conn_bias_update_);
    ptr_conn_bias_update_ = pconn;
  }
  //! Sets the ptr_conn_bias_htilde_.
  void set_ptr_conn_bias_htilde(XConnectionBase *pconn) {
    assert(!ptr_conn_bias_htilde_);
    ptr_conn_bias_htilde_ = pconn;
  }
  //! Sets the ptr_conn_globalbias_reset_.
  void set_ptr_conn_globalbias_reset(XConnectionBase *pconn) {
    assert(!ptr_conn_globalbias_reset_);
    ptr_conn_globalbias_reset_ = pconn;
  }
  //! Sets the ptr_conn_globalbias_update_.
  void set_ptr_conn_globalbias_update(XConnectionBase *pconn) {
    assert(!ptr_conn_globalbias_update_);
    ptr_conn_globalbias_update_ = pconn;
  }
  //! Sets the ptr_conn_globalbias_htilde_.
  void set_ptr_conn_globalbias_htilde(XConnectionBase *pconn) {
    assert(!ptr_conn_globalbias_htilde_);
    ptr_conn_globalbias_htilde_ = pconn;
  }

  //! Sets input of all errors in [b_start, b_start + bs) to value.
  //! \note If val = 0, consider to use {X,Y}BackwardPropagateR when possible.
  void SetInputOfErrorsToValue(IndexType b_start, IndexType bs, ActivationType val) {
    hidden_layer_.SetInputOfErrorsToValue(b_start, bs, val);
  }

  //! Computes activations of neurons for [t, (t + 1) * atom_batchsize_) according to their inputs,
  //! with init_hidden_layer as the previous (initial) hidden layer.
  void ComputeActivations(const XBatchLayer<IdentityNeurons> &init_hidden_layer, IndexType t) {
    assert(init_hidden_layer.nneurons() == nneurons());
    assert(init_hidden_layer.batchsize() == atom_batchsize_);
    assert(ptr_negones_);
    assert(ptr_conn_hprev_reset_);
    assert(ptr_conn_hprev_update_);
    assert(ptr_conn_hprev_htilde_);
    assert(ptr_conn_bias_reset_);
    assert(ptr_conn_bias_update_);
    assert(ptr_conn_bias_htilde_);
    assert(ptr_conn_globalbias_reset_);
    assert(ptr_conn_globalbias_update_);
    assert(ptr_conn_globalbias_htilde_);

    IndexType b_start = t * atom_batchsize_;

    //! hprev        -> reset
    //! bias         -> reset
    //! globalbias   -> reset
    reset_gate_.XForwardPropagateA(init_hidden_layer, 0,
                                   *ptr_conn_hprev_reset_,
                                   b_start, atom_batchsize_);
    reset_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                   *ptr_conn_bias_reset_,
                                   b_start, atom_batchsize_);
    reset_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                   *ptr_conn_globalbias_reset_,
                                   b_start, atom_batchsize_);
    reset_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> update
    //! bias         -> update
    //! globalbias   -> update
    update_gate_.XForwardPropagateA(init_hidden_layer, 0,
                                    *ptr_conn_hprev_update_,
                                    b_start, atom_batchsize_);
    update_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_bias_update_,
                                    b_start, atom_batchsize_);
    update_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_globalbias_update_,
                                    b_start, atom_batchsize_);
    update_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! update  -> aux_update
    aux_update_gate_.SetInputOfActivationsToValue(b_start, atom_batchsize_, 1.0f);
    aux_update_gate_.XForwardPropagateA(update_gate_, b_start,
                                        *ptr_negones_, 0,
                                        b_start, atom_batchsize_);
    aux_update_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev .* reset  -> rhprev
    rhprev_layer_.XForwardPropagateR(init_hidden_layer, 0,
                                     reset_gate_, b_start,
                                     b_start, atom_batchsize_);
    rhprev_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! rhprev       -> htilde
    //! bias         -> htilde
    //! globalbias   -> htilde
    htilde_layer_.XForwardPropagateA(rhprev_layer_, b_start,
                                     *ptr_conn_hprev_htilde_,
                                     b_start, atom_batchsize_);
    htilde_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                     *ptr_conn_bias_htilde_,
                                     b_start, atom_batchsize_);
    htilde_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                     *ptr_conn_globalbias_htilde_,
                                     b_start, atom_batchsize_);
    htilde_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev .* update         -> hidden
    //! htilde .* aux_update    -> hidden
    hidden_layer_.XForwardPropagateR(init_hidden_layer, 0,
                                     update_gate_, b_start,
                                     b_start, atom_batchsize_);
    hidden_layer_.XForwardPropagateA(htilde_layer_, b_start,
                                     aux_update_gate_, b_start,
                                     b_start, atom_batchsize_);
    hidden_layer_.ComputeActivations(b_start, atom_batchsize_);
  }

  //! Computes errors of neurons for [t * atom_batchsize_, (t + 1) * atom_batchsize_) according to their inputs.
  //! with init_hidden_layer as the previous (initial) hidden layer.
  void ComputeErrors(const XBatchLayer<IdentityNeurons> &init_hidden_layer, IndexType t) {
    assert(init_hidden_layer.nneurons() == nneurons());
    assert(init_hidden_layer.batchsize() == atom_batchsize_);
    assert(ptr_negones_);
    assert(ptr_conn_hprev_reset_);
    assert(ptr_conn_hprev_update_);
    assert(ptr_conn_hprev_htilde_);
    assert(ptr_conn_bias_reset_);
    assert(ptr_conn_bias_update_);
    assert(ptr_conn_bias_htilde_);
    assert(ptr_conn_globalbias_reset_);
    assert(ptr_conn_globalbias_update_);
    assert(ptr_conn_globalbias_htilde_);

    IndexType b_start = t * atom_batchsize_;

    //! hprev .* update         <- hidden
    //! htilde .* aux_update    <- hidden
    hidden_layer_.ComputeErrors(b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(aux_update_gate_, b_start,
                                      htilde_layer_, b_start,
                                      b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(htilde_layer_, b_start,
                                      aux_update_gate_, b_start,
                                      b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(update_gate_, b_start,
                                      init_hidden_layer, 0,
                                      b_start, atom_batchsize_);
    //! \note No need to propagate errors to init_hidden_layer.

    //! rhprev       <- htilde
    //! bias         <- htilde
    //! globalbias   <- htilde
    htilde_layer_.ComputeErrors(b_start, atom_batchsize_);
    htilde_layer_.XBackwardPropagateRA(rhprev_layer_, b_start,
                                       *ptr_conn_hprev_htilde_,
                                       b_start, atom_batchsize_);
    htilde_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                       *ptr_conn_bias_htilde_,
                                       b_start, atom_batchsize_);
    htilde_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                       *ptr_conn_globalbias_htilde_,
                                       b_start, atom_batchsize_);

    //! hprev .* reset  <- rhprev
    rhprev_layer_.ComputeErrors(b_start, atom_batchsize_);
    //! \note Errors of reset gate have not been reset yet.
    rhprev_layer_.XBackwardPropagateR(reset_gate_, b_start,
                                      init_hidden_layer, 0,
                                      b_start, atom_batchsize_);
    //! \note No need to propagate errors to init_hidden_layer.

    //! update  <- aux_update
    aux_update_gate_.ComputeErrors(b_start, atom_batchsize_);
    aux_update_gate_.XBackwardPropagateA(update_gate_, b_start,
                                         *ptr_negones_, 0,
                                         b_start, atom_batchsize_);
    //! \note *ptr_negones_ is constant, no need to backward.

    //! hprev        <- update
    //! bias         <- update
    //! globalbias   <- update
    update_gate_.ComputeErrors(b_start, atom_batchsize_);
    //! \note No need to propagate errors to init_hidden_layer.
    update_gate_.XBackwardPropagateNA(init_hidden_layer, 0,
                                      *ptr_conn_hprev_update_,
                                      b_start, atom_batchsize_);
    update_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_bias_update_,
                                      b_start, atom_batchsize_);
    update_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_globalbias_update_,
                                      b_start, atom_batchsize_);

    //! hprev        <- reset
    //! bias         <- reset
    //! globalbias   <- reset
    reset_gate_.ComputeErrors(b_start, atom_batchsize_);
    //! \note No need to propagate errors to init_hidden_layer.
    reset_gate_.XBackwardPropagateNA(init_hidden_layer, 0,
                                     *ptr_conn_hprev_reset_,
                                     b_start, atom_batchsize_);
    reset_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                     *ptr_conn_bias_reset_,
                                     b_start, atom_batchsize_);
    reset_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                     *ptr_conn_globalbias_reset_,
                                     b_start, atom_batchsize_);
  }


  //! Computes activations of neurons for [t, (t + 1) * atom_batchsize_) according to their inputs,
  //! with [prev_t, (prev_t + 1) * atom_batchsize_) as the previous hidden layer.
  void ComputeActivations(IndexType prev_t, IndexType t) {
    assert(t >= 0);
    assert(t < capacity_);
    assert(prev_t >= 0);
    assert(prev_t < capacity_);
    assert(prev_t != t);
    assert(ptr_negones_);
    assert(ptr_conn_hprev_reset_);
    assert(ptr_conn_hprev_update_);
    assert(ptr_conn_hprev_htilde_);
    assert(ptr_conn_bias_reset_);
    assert(ptr_conn_bias_update_);
    assert(ptr_conn_bias_htilde_);
    assert(ptr_conn_globalbias_reset_);
    assert(ptr_conn_globalbias_update_);
    assert(ptr_conn_globalbias_htilde_);

    IndexType prev_b_start = prev_t * atom_batchsize_;
    IndexType b_start = t * atom_batchsize_;

    //! hprev        -> reset
    //! bias         -> reset
    //! globalbias   -> reset
    reset_gate_.XForwardPropagateA(hidden_layer_, prev_b_start,
                                   *ptr_conn_hprev_reset_,
                                   b_start, atom_batchsize_);
    reset_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                   *ptr_conn_bias_reset_,
                                   b_start, atom_batchsize_);
    reset_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                   *ptr_conn_globalbias_reset_,
                                   b_start, atom_batchsize_);
    reset_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> update
    //! bias         -> update
    //! globalbias   -> update
    update_gate_.XForwardPropagateA(hidden_layer_, prev_b_start,
                                    *ptr_conn_hprev_update_,
                                    b_start, atom_batchsize_);
    update_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_bias_update_,
                                    b_start, atom_batchsize_);
    update_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_globalbias_update_,
                                    b_start, atom_batchsize_);
    update_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! update  -> aux_update
    aux_update_gate_.SetInputOfActivationsToValue(b_start, atom_batchsize_, 1.0f);
    aux_update_gate_.XForwardPropagateA(update_gate_, b_start,
                                        *ptr_negones_, 0,
                                        b_start, atom_batchsize_);
    aux_update_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev .* reset  -> rhprev
    rhprev_layer_.XForwardPropagateR(hidden_layer_, prev_b_start,
                                     reset_gate_, b_start,
                                     b_start, atom_batchsize_);
    rhprev_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! rhprev       -> htilde
    //! bias         -> htilde
    //! globalbias   -> htilde
    htilde_layer_.XForwardPropagateA(rhprev_layer_, b_start,
                                     *ptr_conn_hprev_htilde_,
                                     b_start, atom_batchsize_);
    htilde_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                     *ptr_conn_bias_htilde_,
                                     b_start, atom_batchsize_);
    htilde_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                     *ptr_conn_globalbias_htilde_,
                                     b_start, atom_batchsize_);
    htilde_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev .* update         -> hidden
    //! htilde .* aux_update    -> hidden
    hidden_layer_.XForwardPropagateR(hidden_layer_, prev_b_start,
                                     update_gate_, b_start,
                                     b_start, atom_batchsize_);
    hidden_layer_.XForwardPropagateA(htilde_layer_, b_start,
                                     aux_update_gate_, b_start,
                                     b_start, atom_batchsize_);
    hidden_layer_.ComputeActivations(b_start, atom_batchsize_);
  }

  //! Computes errors of neurons for [t * atom_batchsize_, (t + 1) * atom_batchsize_) according to their inputs,
  //! with [prev_t, (prev_t + 1) * atom_batchsize_) as the previous hidden layer.
  void ComputeErrors(IndexType prev_t, IndexType t) {
    assert(t >= 0);
    assert(t < capacity_);
    assert(prev_t >= 0);
    assert(prev_t < capacity_);
    assert(prev_t != t);
    assert(ptr_negones_);
    assert(ptr_conn_hprev_reset_);
    assert(ptr_conn_hprev_update_);
    assert(ptr_conn_hprev_htilde_);

    IndexType prev_b_start = prev_t * atom_batchsize_;
    IndexType b_start = t * atom_batchsize_;

    //! hprev .* update         <- hidden
    //! htilde .* aux_update    <- hidden
    hidden_layer_.ComputeErrors(b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(aux_update_gate_, b_start,
                                      htilde_layer_, b_start,
                                      b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(htilde_layer_, b_start,
                                      aux_update_gate_, b_start,
                                      b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(update_gate_, b_start,
                                      hidden_layer_, prev_b_start,
                                      b_start, atom_batchsize_);
    //! \note Errors of hidden layer have been reset.
    hidden_layer_.XBackwardPropagateA(hidden_layer_, prev_b_start,
                                      update_gate_, b_start,
                                      b_start, atom_batchsize_);

    //! rhprev       <- htilde
    //! bias         <- htilde
    //! globalbias   <- htilde
    htilde_layer_.ComputeErrors(b_start, atom_batchsize_);
    htilde_layer_.XBackwardPropagateRA(rhprev_layer_, b_start,
                                       *ptr_conn_hprev_htilde_,
                                       b_start, atom_batchsize_);
    htilde_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                       *ptr_conn_bias_htilde_,
                                       b_start, atom_batchsize_);
    htilde_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                       *ptr_conn_globalbias_htilde_,
                                       b_start, atom_batchsize_);

    //! hprev .* reset  <- rhprev
    rhprev_layer_.ComputeErrors(b_start, atom_batchsize_);
    //! \note Errors of reset gate have not been reset yet.
    rhprev_layer_.XBackwardPropagateR(reset_gate_, b_start,
                                      hidden_layer_, prev_b_start,
                                      b_start, atom_batchsize_);
    //! \note Errors of hidden layer have been reset.
    rhprev_layer_.XBackwardPropagateA(hidden_layer_, prev_b_start,
                                      reset_gate_, b_start,
                                      b_start, atom_batchsize_);

    //! update  <- aux_update
    aux_update_gate_.ComputeErrors(b_start, atom_batchsize_);
    aux_update_gate_.XBackwardPropagateA(update_gate_, b_start,
                                         *ptr_negones_, 0,
                                         b_start, atom_batchsize_);
    //! \note *ptr_negones_ is constant, no need to backward.

    //! hprev        <- update
    //! bias         <- update
    //! globalbias   <- update
    update_gate_.ComputeErrors(b_start, atom_batchsize_);
    update_gate_.XBackwardPropagateAA(hidden_layer_, prev_b_start,
                                      *ptr_conn_hprev_update_,
                                      b_start, atom_batchsize_);
    update_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_bias_update_,
                                      b_start, atom_batchsize_);
    update_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_globalbias_update_,
                                      b_start, atom_batchsize_);

    //! hprev        <- reset
    //! bias         <- reset
    //! globalbias   <- reset
    reset_gate_.ComputeErrors(b_start, atom_batchsize_);
    reset_gate_.XBackwardPropagateAA(hidden_layer_, prev_b_start,
                                     *ptr_conn_hprev_reset_,
                                     b_start, atom_batchsize_);
    reset_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                     *ptr_conn_bias_reset_,
                                     b_start, atom_batchsize_);
    reset_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                     *ptr_conn_globalbias_reset_,
                                     b_start, atom_batchsize_);
  }

  //! Forward propagates
  //! from all input neurons in [bx_start, bx_start + bs)
  //! to all output neurons in [by_start, by_start + bs).
  //! - Resets and accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardPropagateR(const BatchLayerBase &input_layer,
                          IndexType bx_start,
                          XConnectionBase &conn_reset,
                          XConnectionBase &conn_update,
                          XConnectionBase &conn_htilde,
                          IndexType by_start, IndexType bs) {
    assert(input_layer.type() == xbatchlayer);

    reset_gate_.XForwardPropagateR(input_layer, bx_start,
                                   conn_reset,
                                   by_start, bs);
    update_gate_.XForwardPropagateR(input_layer, bx_start,
                                    conn_update,
                                    by_start, bs);
    htilde_layer_.XForwardPropagateR(input_layer, bx_start,
                                     conn_htilde,
                                     by_start, bs);
  }

  //! Forward propagates
  //! from all input neurons in [bx_start, bx_start + bs)
  //! to all output neurons in [by_start, by_start + bs).
  //! - Not resets but accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardPropagateA(const BatchLayerBase &input_layer,
                          IndexType bx_start,
                          XConnectionBase &conn_reset,
                          XConnectionBase &conn_update,
                          XConnectionBase &conn_htilde,
                          IndexType by_start, IndexType bs) {
    assert(input_layer.type() == xbatchlayer);

    reset_gate_.XForwardPropagateA(input_layer, bx_start,
                                   conn_reset,
                                   by_start, bs);
    update_gate_.XForwardPropagateA(input_layer, bx_start,
                                    conn_update,
                                    by_start, bs);
    htilde_layer_.XForwardPropagateA(input_layer, bx_start,
                                     conn_htilde,
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
                            XConnectionBase &conn_reset,
                            XConnectionBase &conn_update,
                            XConnectionBase &conn_htilde,
                            IndexType by_start, IndexType bs) const {
    assert(input_layer.type() == xbatchlayer);

    //! \note Only the first one needs to call RA,
    //!       the rest only needs to call NA.
    htilde_layer_.XBackwardPropagateRA(input_layer, bx_start,
                                       conn_htilde,
                                       by_start, bs);
    update_gate_.XBackwardPropagateAA(input_layer, bx_start,
                                      conn_update,
                                      by_start, bs);
    reset_gate_.XBackwardPropagateAA(input_layer, bx_start,
                                     conn_reset,
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
                            XConnectionBase &conn_reset,
                            XConnectionBase &conn_update,
                            XConnectionBase &conn_htilde,
                            IndexType by_start, IndexType bs) const {
    assert(input_layer.type() == xbatchlayer);

    htilde_layer_.XBackwardPropagateAA(input_layer, bx_start,
                                       conn_htilde,
                                       by_start, bs);
    update_gate_.XBackwardPropagateAA(input_layer, bx_start,
                                      conn_update,
                                      by_start, bs);
    reset_gate_.XBackwardPropagateAA(input_layer, bx_start,
                                     conn_reset,
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
    TIALNN_ERR("Error: experimental methods (GatedRecurrentXUnit)");
    std::exit(EXIT_FAILURE);
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
    TIALNN_ERR("Error: experimental methods (GatedRecurrentXUnit)");
    std::exit(EXIT_FAILURE);
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
    htilde_layer_.set_nneurons_batchsize(nneurons(), batchsize());
    rhprev_layer_.set_nneurons_batchsize(nneurons(), batchsize());
    reset_gate_.set_nneurons_batchsize(nneurons(), batchsize());
    update_gate_.set_nneurons_batchsize(nneurons(), batchsize());
    aux_update_gate_.set_nneurons_batchsize(nneurons(), batchsize());
  };

  //! Atom batch size.
  IndexType atom_batchsize_;
  //! Capacity.
  IndexType capacity_;

  //! Pointer to auxilary batchlayer of all -1s.
  //! Used for compute 1 - update_gate_.
  //! \note It points to a constant layer.
  const XBatchLayer<IdentityNeurons> *ptr_negones_;

  //! Pointer to the bias layer (shared by the whole network).
  const XBatchLayer<IdentityNeurons> *ptr_bias_layer_;

  //! Pointer to the recurrent connections.
  XConnectionBase *ptr_conn_hprev_reset_;
  XConnectionBase *ptr_conn_hprev_update_;
  XConnectionBase *ptr_conn_hprev_htilde_;

  //! Pointer to the bias connections.
  XConnectionBase *ptr_conn_bias_reset_;
  XConnectionBase *ptr_conn_bias_update_;
  XConnectionBase *ptr_conn_bias_htilde_;

  //! Pointer to the global bias connections.
  XConnectionBase *ptr_conn_globalbias_reset_;
  XConnectionBase *ptr_conn_globalbias_update_;
  XConnectionBase *ptr_conn_globalbias_htilde_;

  //! Actual hidden layer.
  //! equation (7) in \cite{Cho2014EMNLP}.
  XBatchLayer<IdentityNeurons> hidden_layer_;
  //! Intermidiate layer.
  //! htilde = tanh( U * rhprev )
  //! equation (8) in \cite{Cho2014EMNLP}.
  XBatchLayer<Neuron> htilde_layer_;
  //! Intermidiate layer.
  //! rhprev = r .* hprev.
  XBatchLayer<IdentityNeurons> rhprev_layer_;
  //! Reset gate.
  XBatchLayer<GenericNeurons<sigmoid_f, sigmoid_g>> reset_gate_;
  //! Update gate.
  XBatchLayer<GenericNeurons<sigmoid_f, sigmoid_g>> update_gate_;

  //! Auxilary update gate.
  //! 1 - update_gate_.
  XBatchLayer<IdentityNeurons> aux_update_gate_;

  //! Dummy variable for BatchLayerBase interface.
  std::vector<ErrorType> dummy_errors_;
  std::vector<IndexType> dummy_active_neurons_;
};

} // namespace tialnn

#endif
