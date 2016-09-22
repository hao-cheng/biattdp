/*!
 * \file lstm_recurrent_xunit.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_UNIT_LSTM_RECURRENT_XUNIT_H_
#define TIALNN_UNIT_LSTM_RECURRENT_XUNIT_H_

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

//! LSTM recurrent x-unit without peephole connections
//! See ?
//!
//! Be careful when using the unit, specifically, the batch idx.
//! - The unit stores the batch layers for every timestep.
//! - Each block of batch layers for a timestep is an atom.
//! - batchsize_ = atom_batchsize_ * capacity_.
template <class Neuron>
class LSTMRecurrentXUnit : public BatchLayerBase {
 public:
  //! Constructor.
  explicit LSTMRecurrentXUnit() :
      atom_batchsize_(0), capacity_(0),
      ptr_bias_layer_(nullptr),
      ptr_conn_hprev_in_(nullptr),
      ptr_conn_hprev_ig_(nullptr),
      ptr_conn_hprev_fg_(nullptr),
      ptr_conn_hprev_og_(nullptr),
      ptr_conn_bias_in_(nullptr),
      ptr_conn_bias_ig_(nullptr),
      ptr_conn_bias_fg_(nullptr),
      ptr_conn_bias_og_(nullptr),
      ptr_conn_globalbias_in_(nullptr),
      ptr_conn_globalbias_ig_(nullptr),
      ptr_conn_globalbias_fg_(nullptr),
      ptr_conn_globalbias_og_(nullptr) {}
  //! Destructor.
  virtual ~LSTMRecurrentXUnit() {}

  //! Returns the batch layer type.
  virtual const BatchLayerType type() const override { return xbatchlayer; }
  //! Returns the activations of all hidden neurons.
  virtual const std::vector<ActivationType>& activations() const override {
    return hidden_layer_.activations();
  }
  //! Returns the errors of of all neurons.
  virtual const std::vector<ErrorType>& errors() const override {
    TIALNN_ERR("Error: Invalid method for LSTMRecurrentXUnit");
    std::exit(EXIT_FAILURE);
    return dummy_errors_;
  }
  //! Returns the active neurons.
  virtual const std::vector<IndexType>& active_neurons() const override {
    TIALNN_ERR("Error: LSTMRecurrentXUnit does not have active_neurons");
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

  //! Sets the ptr_conn_hprev_in_.
  void set_ptr_conn_hprev_in(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_in_);
    ptr_conn_hprev_in_ = pconn;
  }
  //! Sets the ptr_conn_hprev_ig_.
  void set_ptr_conn_hprev_ig(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_ig_);
    ptr_conn_hprev_ig_ = pconn;
  }
  //! Sets the ptr_conn_hprev_fg_.
  void set_ptr_conn_hprev_fg(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_fg_);
    ptr_conn_hprev_fg_ = pconn;
  }
  //! Sets the ptr_conn_hprev_og_.
  void set_ptr_conn_hprev_og(XConnectionBase *pconn) {
    assert(!ptr_conn_hprev_og_);
    ptr_conn_hprev_og_ = pconn;
  }
  //! Sets the ptr_bias_layer_.
  void set_ptr_bias_layer(const XBatchLayer<IdentityNeurons> *pb) {
    assert(!ptr_bias_layer_);
    ptr_bias_layer_ = pb;
  } 
  //! Sets the ptr_conn_bias_in_.
  void set_ptr_conn_bias_in(XConnectionBase *pconn) {
    assert(!ptr_conn_bias_in_);
    ptr_conn_bias_in_ = pconn;
  }
  //! Sets the ptr_conn_bias_ig_.
  void set_ptr_conn_bias_ig(XConnectionBase *pconn) {
    assert(!ptr_conn_bias_ig_);
    ptr_conn_bias_ig_ = pconn;
  }
  //! Sets the ptr_conn_bias_fg_.
  void set_ptr_conn_bias_fg(XConnectionBase *pconn) {
    assert(!ptr_conn_bias_fg_);
    ptr_conn_bias_fg_ = pconn;
  }
  //! Sets the ptr_conn_bias_og_.
  void set_ptr_conn_bias_og(XConnectionBase *pconn) {
    assert(!ptr_conn_bias_og_);
    ptr_conn_bias_og_ = pconn;
  }

  //! Sets the ptr_conn_globalbias_in_.
  void set_ptr_conn_globalbias_in(XConnectionBase *pconn) {
    assert(!ptr_conn_globalbias_in_);
    ptr_conn_globalbias_in_ = pconn;
  }
  //! Sets the ptr_conn_globalbias_ig_.
  void set_ptr_conn_globalbias_ig(XConnectionBase *pconn) {
    assert(!ptr_conn_globalbias_ig_);
    ptr_conn_globalbias_ig_ = pconn;
  }
  //! Sets the ptr_conn_globalbias_fg_.
  void set_ptr_conn_globalbias_fg(XConnectionBase *pconn) {
    assert(!ptr_conn_globalbias_fg_);
    ptr_conn_globalbias_fg_ = pconn;
  }
  //! Sets the ptr_conn_globalbias_og_.
  void set_ptr_conn_globalbias_og(XConnectionBase *pconn) {
    assert(!ptr_conn_globalbias_og_);
    ptr_conn_globalbias_og_ = pconn;
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
    assert(ptr_conn_hprev_in_);
    assert(ptr_conn_hprev_ig_);
    assert(ptr_conn_hprev_fg_);
    assert(ptr_conn_hprev_og_);
    assert(ptr_conn_bias_in_);
    assert(ptr_conn_bias_ig_);
    assert(ptr_conn_bias_fg_);
    assert(ptr_conn_bias_og_);
    assert(ptr_conn_globalbias_in_);
    assert(ptr_conn_globalbias_ig_);
    assert(ptr_conn_globalbias_fg_);
    assert(ptr_conn_globalbias_og_);

    IndexType b_start = t * atom_batchsize_;

    //! hprev        -> input
    //! bias         -> input
    //! globalbias   -> input
    input_layer_.XForwardPropagateA(init_hidden_layer, 0,
                                    *ptr_conn_hprev_in_,
                                    b_start, atom_batchsize_);
    input_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_bias_in_,
                                    b_start, atom_batchsize_);
    input_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_globalbias_in_,
                                    b_start, atom_batchsize_);
    input_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> in_gate_
    //! bias         -> in_gate_
    //! globalbias   -> in_gate_
    in_gate_.XForwardPropagateA(init_hidden_layer, 0,
                                *ptr_conn_hprev_ig_,
                                b_start, atom_batchsize_);
    in_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_bias_ig_,
                                b_start, atom_batchsize_);
    in_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_globalbias_ig_,
                                b_start, atom_batchsize_);
    in_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> fg_gate_
    //! bias         -> fg_gate_
    //! globalbias   -> fg_gate_
    fg_gate_.XForwardPropagateA(init_hidden_layer, 0,
                                *ptr_conn_hprev_fg_,
                                b_start, atom_batchsize_);
    fg_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_bias_fg_,
                                b_start, atom_batchsize_);
    fg_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_globalbias_fg_,
                                b_start, atom_batchsize_);
    fg_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! mprev .* fg -> mcurrent 
    //! in .* ig    -> mcurrent
    memcell_layer_.XForwardPropagateR(init_hidden_layer, 0,
                                      fg_gate_, b_start,
                                      b_start, atom_batchsize_);
    memcell_layer_.XForwardPropagateA(input_layer_, b_start,
                                      in_gate_, b_start,
                                      b_start, atom_batchsize_);

    memcell_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> out_gate_
    //! bias         -> out_gate_
    //! globalbias   -> out_gate_
    out_gate_.XForwardPropagateA(init_hidden_layer, 0,
                                 *ptr_conn_hprev_og_,
                                 b_start, atom_batchsize_);
    out_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                 *ptr_conn_bias_og_,
                                 b_start, atom_batchsize_);
    out_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                 *ptr_conn_globalbias_og_,
                                 b_start, atom_batchsize_);
    out_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! out_gate_ .* memcurrent -> hidden
    hidden_layer_.XForwardPropagateR(memcell_layer_, b_start,
                                     out_gate_, b_start,
                                     b_start, atom_batchsize_);
    hidden_layer_.ComputeActivations(b_start, atom_batchsize_);
  }

  //! Computes errors of neurons for [t * atom_batchsize_, (t + 1) * atom_batchsize_) according to their inputs.
  //! with init_hidden_layer as the previous (initial) hidden layer.
  void ComputeErrors(const XBatchLayer<IdentityNeurons> &init_hidden_layer, IndexType t) {
    assert(init_hidden_layer.nneurons() == nneurons());
    assert(init_hidden_layer.batchsize() == atom_batchsize_);
    assert(ptr_conn_hprev_in_);
    assert(ptr_conn_hprev_ig_);
    assert(ptr_conn_hprev_fg_);
    assert(ptr_conn_hprev_og_);
    assert(ptr_conn_bias_in_);
    assert(ptr_conn_bias_ig_);
    assert(ptr_conn_bias_fg_);
    assert(ptr_conn_bias_og_);
    assert(ptr_conn_globalbias_in_);
    assert(ptr_conn_globalbias_ig_);
    assert(ptr_conn_globalbias_fg_);
    assert(ptr_conn_globalbias_og_);

    IndexType b_start = t * atom_batchsize_;

    //! out_gate_ .* memcurrent <- hidden
    //! \note No need to propagate errors to init_hidden_layer.
    hidden_layer_.ComputeErrors(b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(out_gate_, b_start,
                                      memcell_layer_, b_start,
                                      b_start, atom_batchsize_);

    //! \note No need to propagate errors to init_hidden_layer.

    //! bias         <- out_gate_ 
    //! globalbias   <- out_gate_
    out_gate_.ComputeErrors(b_start, atom_batchsize_);
    out_gate_.XBackwardPropagateNA(init_hidden_layer, 0,
                                   *ptr_conn_hprev_og_,
                                   b_start, atom_batchsize_);
    out_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                   *ptr_conn_bias_og_,
                                   b_start, atom_batchsize_);
    out_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                   *ptr_conn_globalbias_og_,
                                   b_start, atom_batchsize_);

    //! ig .* input_layer_ + fg .* memprev <- memcurrent
    memcell_layer_.ComputeErrors(b_start, atom_batchsize_);
    memcell_layer_.XBackwardPropagateR(in_gate_, b_start,
                                       input_layer_, b_start,
                                       b_start, atom_batchsize_);
    memcell_layer_.XBackwardPropagateR(input_layer_, b_start,
                                       in_gate_, b_start,
                                       b_start, atom_batchsize_);
    memcell_layer_.XBackwardPropagateR(fg_gate_, b_start,
                                       init_hidden_layer, 0,
                                       b_start, atom_batchsize_);

    //! bias         <- fg_gate_ 
    //! globalbias   <- fg_gate_
    fg_gate_.ComputeErrors(b_start, atom_batchsize_);
    fg_gate_.XBackwardPropagateNA(init_hidden_layer, 0,
                                  *ptr_conn_hprev_fg_,
                                  b_start, atom_batchsize_);
    fg_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_bias_fg_,
                                  b_start, atom_batchsize_);
    fg_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_globalbias_fg_,
                                  b_start, atom_batchsize_);

    //! bias         <- in_gate_ 
    //! globalbias   <- in_gate_
    in_gate_.ComputeErrors(b_start, atom_batchsize_);
    in_gate_.XBackwardPropagateNA(init_hidden_layer, 0,
                                  *ptr_conn_hprev_ig_,
                                  b_start, atom_batchsize_);
    in_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_bias_ig_,
                                  b_start, atom_batchsize_);
    in_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_globalbias_ig_,
                                  b_start, atom_batchsize_);

    //! bias         <- input_layer_ 
    //! globalbias   <- input_layer_
    input_layer_.ComputeErrors(b_start, atom_batchsize_);
    input_layer_.XBackwardPropagateNA(init_hidden_layer, 0,
                                      *ptr_conn_hprev_in_,
                                      b_start, atom_batchsize_);
    input_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_bias_in_,
                                      b_start, atom_batchsize_);
    input_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_globalbias_in_,
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
    assert(ptr_conn_hprev_in_);
    assert(ptr_conn_hprev_ig_);
    assert(ptr_conn_hprev_fg_);
    assert(ptr_conn_hprev_og_);
    assert(ptr_conn_bias_in_);
    assert(ptr_conn_bias_ig_);
    assert(ptr_conn_bias_fg_);
    assert(ptr_conn_bias_og_);
    assert(ptr_conn_globalbias_in_);
    assert(ptr_conn_globalbias_ig_);
    assert(ptr_conn_globalbias_fg_);
    assert(ptr_conn_globalbias_og_);


    IndexType prev_b_start = prev_t * atom_batchsize_;
    IndexType b_start = t * atom_batchsize_;

    //! hprev        -> input
    //! bias         -> input
    //! globalbias   -> input
    input_layer_.XForwardPropagateA(hidden_layer_, prev_b_start,
                                    *ptr_conn_hprev_in_,
                                    b_start, atom_batchsize_);
    input_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_bias_in_,
                                    b_start, atom_batchsize_);
    input_layer_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                    *ptr_conn_globalbias_in_,
                                    b_start, atom_batchsize_);
    input_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> in_gate
    //! bias         -> in_gate
    //! globalbias   -> in_gate
    in_gate_.XForwardPropagateA(hidden_layer_, prev_b_start,
                                *ptr_conn_hprev_ig_,
                                b_start, atom_batchsize_);
    in_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_bias_ig_,
                                b_start, atom_batchsize_);
    in_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_globalbias_ig_,
                                b_start, atom_batchsize_);
    in_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> fg_gate
    //! bias         -> fg_gate
    //! globalbias   -> fg_gate
    fg_gate_.XForwardPropagateA(hidden_layer_, prev_b_start,
                                *ptr_conn_hprev_fg_,
                                b_start, atom_batchsize_);
    fg_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_bias_fg_,
                                b_start, atom_batchsize_);
    fg_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                *ptr_conn_globalbias_fg_,
                                b_start, atom_batchsize_);
    fg_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! mprev .* fg -> mcurrent 
    //! in .* ig    -> mcurrent
    memcell_layer_.XForwardPropagateR(memcell_layer_, prev_b_start,
                                      fg_gate_, b_start,
                                      b_start, atom_batchsize_);
    memcell_layer_.XForwardPropagateA(input_layer_, b_start,
                                      in_gate_, b_start,
                                      b_start, atom_batchsize_);
    memcell_layer_.ComputeActivations(b_start, atom_batchsize_);

    //! hprev        -> out_gate_
    //! bias         -> out_gate_
    //! globalbias   -> out_gate_
    out_gate_.XForwardPropagateA(hidden_layer_, prev_b_start,
                                 *ptr_conn_hprev_og_,
                                 b_start, atom_batchsize_);
    out_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                 *ptr_conn_bias_og_,
                                 b_start, atom_batchsize_);
    out_gate_.XForwardPropagateA(*ptr_bias_layer_, 0,
                                 *ptr_conn_globalbias_og_,
                                 b_start, atom_batchsize_);
    out_gate_.ComputeActivations(b_start, atom_batchsize_);

    //! out_gate_ .* memcurrent -> hidden
    hidden_layer_.XForwardPropagateR(memcell_layer_, b_start,
                                     out_gate_, b_start,
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
    assert(ptr_conn_hprev_in_);
    assert(ptr_conn_hprev_ig_);
    assert(ptr_conn_hprev_fg_);
    assert(ptr_conn_hprev_og_);
    assert(ptr_conn_bias_in_);
    assert(ptr_conn_bias_ig_);
    assert(ptr_conn_bias_fg_);
    assert(ptr_conn_bias_og_);
    assert(ptr_conn_globalbias_in_);
    assert(ptr_conn_globalbias_ig_);
    assert(ptr_conn_globalbias_fg_);
    assert(ptr_conn_globalbias_og_);

    IndexType prev_b_start = prev_t * atom_batchsize_;
    IndexType b_start = t * atom_batchsize_;

    //! memcell .* og         <- hidden
    hidden_layer_.ComputeErrors(b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateA(memcell_layer_, b_start,
                                      out_gate_, b_start,
                                      b_start, atom_batchsize_);
    hidden_layer_.XBackwardPropagateR(out_gate_, b_start,
                                      memcell_layer_, b_start,
                                      b_start, atom_batchsize_);
    //! \note Errors of hidden layer have been reset.

    //! hprev        <- out_gate_
    //! bias         <- out_gate_ 
    //! globalbias   <- out_gate_
    out_gate_.ComputeErrors(b_start, atom_batchsize_);
    out_gate_.XBackwardPropagateAA(hidden_layer_, prev_b_start,
                                   *ptr_conn_hprev_og_,
                                   b_start, atom_batchsize_);
    out_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                   *ptr_conn_bias_og_,
                                   b_start, atom_batchsize_);
    out_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                   *ptr_conn_globalbias_og_,
                                   b_start, atom_batchsize_);

    //! ig .* input_layer_ + fg .* memprev <- memcurrent
    memcell_layer_.ComputeErrors(b_start, atom_batchsize_);
    memcell_layer_.XBackwardPropagateR(in_gate_, b_start,
                                       input_layer_, b_start,
                                       b_start, atom_batchsize_);
    memcell_layer_.XBackwardPropagateR(input_layer_, b_start,
                                       in_gate_, b_start,
                                       b_start, atom_batchsize_);
    memcell_layer_.XBackwardPropagateR(fg_gate_, b_start,
                                       memcell_layer_, prev_b_start,
                                       b_start, atom_batchsize_);
    memcell_layer_.XBackwardPropagateA(memcell_layer_, prev_b_start,
                                       fg_gate_, b_start,
                                       b_start, atom_batchsize_);

    //! hprev        <- fg_gate_
    //! bias         <- fg_gate_ 
    //! globalbias   <- fg_gate_
    fg_gate_.ComputeErrors(b_start, atom_batchsize_);
    fg_gate_.XBackwardPropagateAA(hidden_layer_, prev_b_start,
                                  *ptr_conn_hprev_fg_,
                                  b_start, atom_batchsize_);
    fg_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_bias_fg_,
                                  b_start, atom_batchsize_);
    fg_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_globalbias_fg_,
                                  b_start, atom_batchsize_);

    //! hprev        <- in_gate_
    //! bias         <- in_gate_ 
    //! globalbias   <- in_gate_
    in_gate_.ComputeErrors(b_start, atom_batchsize_);
    in_gate_.XBackwardPropagateAA(hidden_layer_, prev_b_start,
                                  *ptr_conn_hprev_ig_,
                                  b_start, atom_batchsize_);
    in_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_bias_ig_,
                                  b_start, atom_batchsize_);
    in_gate_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                  *ptr_conn_globalbias_ig_,
                                  b_start, atom_batchsize_);

    //! hprev        <- input_layer_
    //! bias         <- input_layer_ 
    //! globalbias   <- input_layer_
    input_layer_.ComputeErrors(b_start, atom_batchsize_);
    input_layer_.XBackwardPropagateAA(hidden_layer_, prev_b_start,
                                      *ptr_conn_hprev_ig_,
                                      b_start, atom_batchsize_);
    input_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_bias_ig_,
                                      b_start, atom_batchsize_);
    input_layer_.XBackwardPropagateNA(*ptr_bias_layer_, 0,
                                      *ptr_conn_globalbias_ig_,
                                      b_start, atom_batchsize_);

  }

  //! Forward propagates
  //! from all input neurons in [bx_start, bx_start + bs) 
  //! to all output neurons in [by_start, by_start + bs).
  //! - Resets and accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardPropagateR(const BatchLayerBase &x_input_layer,
                          IndexType bx_start,
                          XConnectionBase &conn_in,
                          XConnectionBase &conn_ig,
                          XConnectionBase &conn_fg,
                          XConnectionBase &conn_og,
                          IndexType by_start, IndexType bs) {
    assert(x_input_layer.type() == xbatchlayer);

    input_layer_.XForwardPropagateR(x_input_layer, bx_start,
                                    conn_in,
                                    by_start, bs);
    in_gate_.XForwardPropagateR(x_input_layer, bx_start,
                                conn_ig,
                                by_start, bs);
    fg_gate_.XForwardPropagateR(x_input_layer, bx_start,
                                conn_fg,
                                by_start, bs);
    out_gate_.XForwardPropagateR(x_input_layer, bx_start,
                                 conn_og,
                                 by_start, bs);
  }

  //! Forward propagates
  //! from all input neurons in [bx_start, bx_start + bs) 
  //! to all output neurons in [by_start, by_start + bs).
  //! - Not resets but accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardPropagateA(const BatchLayerBase &x_input_layer,
                          IndexType bx_start,
                          XConnectionBase &conn_in,
                          XConnectionBase &conn_ig,
                          XConnectionBase &conn_fg,
                          XConnectionBase &conn_og,
                          IndexType by_start, IndexType bs) {
    assert(x_input_layer.type() == xbatchlayer);

    input_layer_.XForwardPropagateA(x_input_layer, bx_start,
                                    conn_in,
                                    by_start, bs);
    in_gate_.XForwardPropagateA(x_input_layer, bx_start,
                                conn_ig,
                                by_start, bs);
    fg_gate_.XForwardPropagateA(x_input_layer, bx_start,
                                conn_fg,
                                by_start, bs);
    out_gate_.XForwardPropagateA(x_input_layer, bx_start,
                                 conn_og,
                                 by_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs).
  //! - Resets and accumulates error inputs of input layer.
  //! - Accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateRA(BatchLayerBase &x_input_layer,
                            IndexType bx_start,
                            XConnectionBase &conn_in,
                            XConnectionBase &conn_ig,
                            XConnectionBase &conn_fg,
                            XConnectionBase &conn_og,
                            IndexType by_start, IndexType bs) const {
    assert(x_input_layer.type() == xbatchlayer);

    //! \note Only the first one needs to call RA, 
    //!       the rest only needs to call NA.
    input_layer_.XBackwardPropagateRA(x_input_layer, bx_start,
                                      conn_in,
                                      by_start, bs);
    in_gate_.XBackwardPropagateAA(x_input_layer, bx_start,
                                  conn_ig,
                                  by_start, bs);
    fg_gate_.XBackwardPropagateAA(x_input_layer, bx_start,
                                  conn_fg,
                                  by_start, bs);
    out_gate_.XBackwardPropagateAA(x_input_layer, bx_start,
                                   conn_og,
                                   by_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs).
  //! - Not resets but accumulates error inputs of input layer.
  //! - Accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateAA(BatchLayerBase &x_input_layer,
                            IndexType bx_start,
                            XConnectionBase &conn_in,
                            XConnectionBase &conn_ig,
                            XConnectionBase &conn_fg,
                            XConnectionBase &conn_og,
                            IndexType by_start, IndexType bs) const {
    assert(x_input_layer.type() == xbatchlayer);

    input_layer_.XBackwardPropagateAA(x_input_layer, bx_start,
                                      conn_in,
                                      by_start, bs);
    in_gate_.XBackwardPropagateAA(x_input_layer, bx_start,
                                  conn_ig,
                                  by_start, bs);
    fg_gate_.XBackwardPropagateAA(x_input_layer, bx_start,
                                  conn_fg,
                                  by_start, bs);
    out_gate_.XBackwardPropagateAA(x_input_layer, bx_start,
                                   conn_og,
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
    memcell_layer_.SetInputOfErrorsToValue(bx_start, bs, 0.0f);
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
    memcell_layer_.SetInputOfErrorsToValue(bx_start, bs, 0.0f);
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
    memcell_layer_.SetInputOfErrorsToValue(bx_start, bs, 0.0f);
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
    memcell_layer_.set_nneurons_batchsize(nneurons(), batchsize());
    input_layer_.set_nneurons_batchsize(nneurons(), batchsize());
    in_gate_.set_nneurons_batchsize(nneurons(), batchsize());
    fg_gate_.set_nneurons_batchsize(nneurons(), batchsize());
    out_gate_.set_nneurons_batchsize(nneurons(), batchsize());
  };

  //! Atom batch size.
  IndexType atom_batchsize_;
  //! Capacity.
  IndexType capacity_;

  //! Pointer to the bias layer (shared by the whole network). 
  const XBatchLayer<IdentityNeurons> *ptr_bias_layer_;

  //! Pointer to the recurrent connections.
  XConnectionBase *ptr_conn_hprev_in_;
  XConnectionBase *ptr_conn_hprev_ig_;
  XConnectionBase *ptr_conn_hprev_fg_;
  XConnectionBase *ptr_conn_hprev_og_;

  //! Pointer to the bias connections.
  XConnectionBase *ptr_conn_bias_in_;
  XConnectionBase *ptr_conn_bias_ig_;
  XConnectionBase *ptr_conn_bias_fg_;
  XConnectionBase *ptr_conn_bias_og_;

  //! Pointer to the global bias connections.
  XConnectionBase *ptr_conn_globalbias_in_;
  XConnectionBase *ptr_conn_globalbias_ig_;
  XConnectionBase *ptr_conn_globalbias_fg_;
  XConnectionBase *ptr_conn_globalbias_og_;

  //! Actual hidden layer.
  XBatchLayer<IdentityNeurons> hidden_layer_;
  //! Memory cell layer without nonlinear transformation
  XBatchLayer<IdentityNeurons> memcell_layer_;
  //! Input layer.
  XBatchLayer<Neuron> input_layer_;
  //! Input gate.
  XBatchLayer<GenericNeurons<sigmoid_f, sigmoid_g>> in_gate_;
  //! Forget gate.
  XBatchLayer<GenericNeurons<sigmoid_f, sigmoid_g>> fg_gate_;
  //! Output gate.
  XBatchLayer<GenericNeurons<sigmoid_f, sigmoid_g>> out_gate_;

  //! Dummy variable for BatchLayerBase interface.
  std::vector<ErrorType> dummy_errors_;
  std::vector<IndexType> dummy_active_neurons_;
};

} // namespace tialnn

#endif
