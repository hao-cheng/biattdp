/*!
 * \file attention_softmax_xbatchlayer.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_LAYER_ATTENTION_SOFTMAX_XBATCHLAYER_H_
#define TIALNN_LAYER_ATTENTION_SOFTMAX_XBATCHLAYER_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::vector
#include <vector>
// std::fill
#include <algorithm>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
// tialnn::WeightType
#include "../base.h"
// TIALNN_ERR
#include "../util/logging.h"
// TIALNN_CBLAS_XXX
#include "../util/numeric.h"
// tialnn::AttentionSoftmaxNeurons
#include "../neuron/attention_softmax_neurons.h"
// tialnn::XConnectionBase
#include "../connection/xconnection_base.h"
// tialnn::YConnectionBase
#include "../connection/yconnection_base.h"
// tialnn::SparseInputYBatchLayer
#include "sparse_input_ybatchlayer.h"
// tialnn::BatchLayerBase
#include "batchlayer_base.h"

namespace tialnn {

//! Attention softmax X-BatchLayer.
//! nneurons == 1.
//! batchsize = atom_batchsize_ * capacity_.
//!   kk0_t0, kk1_t0, kk2_t0, ...,
//!   kk0_t1, kk1_t1, ...
//!   ...
//!
//! When used for element-wise multiplication, the nneurons are broadcasted.
class AttentionSoftmaxXBatchLayer : public BatchLayerBase {
 public:
  //! Constructor.
  explicit AttentionSoftmaxXBatchLayer() {}
  //! Destructor.
  virtual ~AttentionSoftmaxXBatchLayer() {}

  //! Returns the batch layer type.
  virtual const BatchLayerType type() const override { return xbatchlayer; }
  //! Returns the activations of all neurons.
  virtual const std::vector<ActivationType>& activations() const override {
    return neurons_.activations;
  }
  //! Returns the errors of all neurons.
  virtual const std::vector<ErrorType>& errors() const override {
    return neurons_.errors;
  }
  //! Returns the active neurons.
  virtual const std::vector<IndexType>& active_neurons() const override {
    TIALNN_ERR("Error: XBatchLayer does not have active_neurons");
    std::exit(EXIT_FAILURE);
    return dummy_active_neurons_;
  }

  //! Returns the activation of a neuron.
  ActivationType activations(IndexType k, IndexType t) const {
    assert(neurons_.nneurons == 1);
    return neurons_.get_activations(0, k, t);
  }
  //! Returns the atom batch size.
  IndexType atom_batchsize() const { return atom_batchsize_; }

#ifdef _TIALNN_DEBUG
  //! Sets the ac state of neurons in [b_start, b_start + bs).
  //! \param s    true: in ac_states; false: not in ac_states.
  virtual void set_ac_states(IndexType b_start, IndexType bs, bool s) override {
    assert(bs > 0);
    assert(b_start + bs <= batchsize());
    if (s) {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        neurons_.ac_states.insert(k);
      }
    } else {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        auto it = neurons_.ac_states.find(k);
        if (neurons_.ac_states.find(k) != neurons_.ac_states.end()) {
          neurons_.ac_states.erase(it);
        }
      }
    }
  }
  //! Sets the er state of neurons in [b_start, b_start + bs).
  //! \param s    true: in er_states; false: not in er_states.
  virtual void set_er_states(IndexType b_start, IndexType bs, bool s) override {
    assert(bs > 0);
    assert(b_start + bs <= batchsize());
    if (s) {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        neurons_.er_states.insert(k);
      }
    } else {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        auto it = neurons_.er_states.find(k);
        if (neurons_.er_states.find(k) != neurons_.er_states.end()) {
          neurons_.er_states.erase(it);
        }
      }
    }
  }

  //! Check the ac state of neurons in [b_start, b_start + bs).
  //! \param s    true: in ac_states; false: not in ac_states.
  virtual bool CheckAcStates(IndexType b_start, IndexType bs, bool s) const override {
    assert(bs > 0);
    assert(b_start + bs <= batchsize());
    if (s) {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        if (neurons_.ac_states.find(k) == neurons_.ac_states.end()) {
          return false;
        }
      }
    } else {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        if (neurons_.ac_states.find(k) != neurons_.ac_states.end()) {
          return false;
        }
      }
    }

    return true;
  }
  //! Check the er state of neurons in [b_start, b_start + bs).
  virtual bool CheckErStates(IndexType b_start, IndexType bs, bool s) const override {
    assert(bs > 0);
    assert(b_start + bs <= batchsize());
    if (s) {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        if (neurons_.er_states.find(k) == neurons_.er_states.end()) {
          return false;
        }
      }
    } else {
      for (IndexType k = b_start; k < b_start + bs; k++) {
        if (neurons_.er_states.find(k) != neurons_.er_states.end()) {
          return false;
        }
      }
    }

    return true;
  }
#endif

  //! Sets the atom batch size.
  void set_atom_batchsize(IndexType bs) { atom_batchsize_ = bs; }
  //! Sets the capacity.
  void set_capacity(IndexType c) { capacity_ = c; }

  //! Accumulate errors of neurons in [b_start, bstart + bs)
  void AccumulateErrors(std::vector<ErrorType> errors, IndexType b_start, IndexType bs) {
    assert(errors.size() == neurons_.errors.size());
    IndexType offset = nneurons() * b_start;
    TIALNN_CBLAS_AXPY(nneurons() * bs,
                      1.0f, errors.data() + offset, 1,
                      neurons_.errors.data() + offset, 1);
  }

  //! Computes activations of neurons in [0, atom_batchsize_ * sz) according to their inputs.
  void ComputeActivations(IndexType sz) {
    neurons_.ComputeActivations(sz);
  }
  //! Compute errors of neurons in [0, atom_batchsize_ * sz) according to their inputs.
  void ComputeErrors(IndexType sz) {
    neurons_.ComputeErrors(sz);
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
    assert(conn.ninput() == input_layer.nneurons());
    assert(conn.noutput() == 1);
    assert(nneurons() == 1);
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input_layer.CheckAcStates(bx_start, bs, true));
    assert(CheckAcStates(by_start, bs, true));
    set_ac_states(by_start, bs, false);
#endif

    conn.BatchForwardPropagate(bs, 0.0f,
                               input_layer.activations(), bx_start,
                               neurons_.activations, by_start);
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
    assert(conn.ninput() == input_layer.nneurons());
    assert(conn.noutput() == 1);
    assert(nneurons() == 1);
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input_layer.CheckAcStates(bx_start, bs, true));
    assert(CheckAcStates(by_start, bs, false));
#endif

    conn.BatchForwardPropagate(bs, 1.0f,
                               input_layer.activations(), bx_start,
                               neurons_.activations, by_start);
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
    assert(conn.ninput() == input_layer.nneurons());
    assert(conn.noutput() == 1);
    assert(nneurons() == 1);
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(by_start, bs, true));
#endif

    input_layer.XBackwardPropagateRAFrom(neurons_.errors, by_start,
                                         conn,
                                         bx_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs).
  //! - Not accumulates error inputs of input layer.
  //! - Accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateNA(const BatchLayerBase &input_layer,
                            IndexType bx_start,
                            XConnectionBase &conn,
                            IndexType by_start, IndexType bs) const {
    assert(input_layer.type() == xbatchlayer);
    assert(conn.ninput() == input_layer.nneurons());
    assert(conn.noutput() == nneurons());
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(by_start, bs, true));
#endif

    input_layer.XBackwardPropagateNAFrom(neurons_.errors, by_start,
                                         conn,
                                         bx_start, bs);
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
    TIALNN_ERR("Error: experimental methods.");
    std::exit(EXIT_FAILURE);
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
    TIALNN_ERR("Error: experimental methods.");
    std::exit(EXIT_FAILURE);
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
    TIALNN_ERR("Error: experimental methods.");
    std::exit(EXIT_FAILURE);
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
    TIALNN_ERR("Error: experimental methods.");
    std::exit(EXIT_FAILURE);
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
    TIALNN_ERR("Error: experimental methods.");
    std::exit(EXIT_FAILURE);
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
    assert(bs > 0);
    assert(bx1_start + bs <= batchsize());
    assert((bx2_start + bs) * noutput <= static_cast<IndexType>(input_activations.size()));
    assert((by_start + bs) * noutput <= static_cast<IndexType>(output_errors.size()));
    assert(nneurons() == 1);
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx1_start, bs, true));
    set_er_states(bx1_start, bs, false);
#endif

    auto it_er = neurons_.errors.begin();
    it_er += bx1_start; //<! \note nneurons == 1
    const ActivationType *px2 = input_activations.data() + noutput * bx2_start;
    const ErrorType *py = output_errors.data() + noutput * by_start;
    for (IndexType k = 0; k < bs; k++) {
      *it_er = TIALNN_CBLAS_DOT(noutput,
                                px2, 1,
                                py, 1);
      ++it_er;
      px2 += noutput;
      py += noutput;
    }
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
    assert(bs > 0);
    assert(bx1_start + bs <= batchsize());
    assert((bx2_start + bs) * noutput <= static_cast<IndexType>(input_activations.size()));
    assert((by_start + bs) * noutput <= static_cast<IndexType>(output_errors.size()));
    assert(nneurons() == 1);
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx1_start, bs, false));
#endif

    auto it_er = neurons_.errors.begin();
    it_er += bx1_start; //<! \note nneurons == 1
    const ActivationType *px2 = input_activations.data() + noutput * bx2_start;
    const ErrorType *py = output_errors.data() + noutput * by_start;
    for (IndexType k = 0; k < bs; k++) {
      *it_er += TIALNN_CBLAS_DOT(noutput,
                                 px2, 1,
                                 py, 1);
      ++it_er;
      px2 += noutput;
      py += noutput;
    }
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
    TIALNN_ERR("Error: experimental methods.");
    std::exit(EXIT_FAILURE);
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
    TIALNN_ERR("Error: experimental methods.");
    std::exit(EXIT_FAILURE);
  }

 private:
  //! Allocates the layer.
  virtual void AllocateLayer() override {
    if (nneurons() != 1) {
      TIALNN_ERR("nneurons_ should be 1!");
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

    const IndexType n = nneurons() * batchsize();
    neurons_.atom_batchsize = atom_batchsize_;
    neurons_.capacity = capacity_;
    neurons_.activations.resize(n);
    neurons_.errors.resize(n);

    std::fill(neurons_.activations.begin(), neurons_.activations.end(), 0.0f);
    std::fill(neurons_.errors.begin(), neurons_.errors.end(), 0.0f);

#ifdef _TIALNN_DEBUG
    assert(CheckAcStates(0, batchsize(), false));
    assert(CheckErStates(0, batchsize(), false));
    set_ac_states(0, batchsize(), true);
    set_er_states(0, batchsize(), true);
#endif
  };

  //! Atom batch size.
  IndexType atom_batchsize_;
  //! Capacity.
  IndexType capacity_;

  //! Neurons.
  AttentionSoftmaxNeurons neurons_;

  //! Dummy variable for BatchLayerBase interface.
  std::vector<IndexType> dummy_active_neurons_;
};

} // namespace tialnn

#endif
