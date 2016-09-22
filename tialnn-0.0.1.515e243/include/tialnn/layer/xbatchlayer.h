/*!
 * \file xbatchlayer.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_LAYER_XBATCHLAYER_H_
#define TIALNN_LAYER_XBATCHLAYER_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::vector
#include <vector>
// std::fill
// std::copy
#include <algorithm>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
// tialnn::WeightType
#include "../base.h"
// TIALNN_CBLAS_XXX
#include "../util/numeric.h"
// tialnn::XConnectionBase
#include "../connection/xconnection_base.h"
// tialnn::YConnectionBase
#include "../connection/yconnection_base.h"
// tialnn::SparseInputYBatchLayer
#include "sparse_input_ybatchlayer.h"
// tialnn::BatchLayerBase
#include "batchlayer_base.h"

namespace tialnn {

//! X-BatchLayer.
template <class Neurons>
class XBatchLayer : public BatchLayerBase {
 public:
  //! Constructor.
  explicit XBatchLayer() {}
  //! Destructor.
  virtual ~XBatchLayer() {}

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
  ActivationType activations(IndexType idx, IndexType k) const {
    return neurons_.get_activations(idx, k);
  }
  //! Returns the error of a neuron.
  ErrorType errors(IndexType idx, IndexType k) const {
    return neurons_.get_errors(idx, k);
  }

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

  //! Sets the errors of neurons in [b_start, b_start + bs).
  void set_errors(IndexType b_start, IndexType bs, const std::vector<ErrorType> &errors) {
    assert(errors.size() == nneurons() * batchsize());
    IndexType n_offset = nneurons() * b_start;
    auto it_er = errors.begin() + n_offset;
    std::copy(it_er, it_er + nneurons() * bs,
              neurons_.errors.begin() + n_offset);

#ifdef _TIALNN_DEBUG
    for (IndexType k = b_start; k < b_start + bs; k++) {
      neurons_.er_states.insert(k);
    }
#endif
  }

  //! Sets activation input of neurons in [b_start, b_start + bs) to value.
  //! \note If val = 0, consider to use {X,Y}ForwardPropagateR when possible.
  void SetInputOfActivationsToValue(IndexType b_start, IndexType bs, ActivationType val) {
    std::fill(neurons_.activations.begin() + nneurons() * b_start,
              neurons_.activations.begin() + nneurons() * (b_start + bs),
              val);

#ifdef _TIALNN_DEBUG
    for (IndexType k = b_start; k < b_start + bs; k++) {
      auto it = neurons_.ac_states.find(k);
      if(it != neurons_.ac_states.end()) {
        neurons_.ac_states.erase(it);
      }
    }
#endif
  }
  //! Sets error input of neurons in [b_start, b_start + bs) to value.
  //! \note If val = 0, consider to use {X,Y}BackwardPropagateR when possible.
  void SetInputOfErrorsToValue(IndexType b_start, IndexType bs, ActivationType val) {
    std::fill(neurons_.errors.begin() + nneurons() * b_start,
              neurons_.errors.begin() + nneurons() * (b_start + bs),
              val);

#ifdef _TIALNN_DEBUG
    for (IndexType k = b_start; k < b_start + bs; k++) {
      auto it = neurons_.er_states.find(k);
      if (it != neurons_.er_states.end()) {
        neurons_.er_states.erase(it);
      }
    }
#endif
  }

  //! Sets all activations to value.
  void SetActivationsToValue(ActivationType val) {
    std::fill(neurons_.activations.begin(), neurons_.activations.end(), val);

#ifdef _TIALNN_DEBUG
    for (IndexType k = 0; k < batchsize(); k++) {
      neurons_.ac_states.insert(k);
    }
#endif
  }

  //! Computes activations of all neurons according to their inputs.
  void ComputeActivations() {
    neurons_.ComputeActivations();
  }
  //! Computes errors of all neurons according to their inputs.
  void ComputeErrors() {
    neurons_.ComputeErrors();
  }

  //! Computes activations of neurons in [b_start, b_start + bs) according to their inputs.
  void ComputeActivations(IndexType b_start, IndexType bs) {
    neurons_.ComputeActivations(b_start, bs);
  }
  //! Computes errors of neurons in [b_start, b_start + bs) according to their inputs.
  void ComputeErrors(IndexType b_start, IndexType bs) {
    neurons_.ComputeErrors(b_start, bs);
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
    assert(conn.noutput() == nneurons());
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
    assert(conn.noutput() == nneurons());
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

  //! Elementwise forward propagates
  //! from all neurons in [bx{1,2}_start, bx{1,2}_start + bs)
  //! in input1 layer and input2_layer
  //! to all hidden neurons in [by_start, by_start + bs).
  //! - Resets and accumulates activation inputs.
  //! \note X- means input1_layer and input2_layer are both XBatchLayer.
  void XForwardPropagateR(const BatchLayerBase &input1_layer,
                          IndexType bx1_start,
                          const BatchLayerBase &input2_layer,
                          IndexType bx2_start,
                          IndexType by_start, IndexType bs) {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == nneurons());
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input1_layer.CheckAcStates(bx1_start, bs, true));
    assert(input2_layer.CheckAcStates(bx2_start, bs, true));
    assert(CheckAcStates(by_start, bs, true));
    set_ac_states(by_start, bs, false);
#endif

    TIALNN_CBLAS_SBMV(CblasColMajor, CblasUpper,
                      nneurons() * bs, 0,
                      1.0f, input1_layer.activations().data() + nneurons() * bx1_start, 1,
                      input2_layer.activations().data() + nneurons() * bx2_start, 1,
                      0.0f, neurons_.activations.data() + nneurons() * by_start, 1);
  }

  //! Elementwise forward propagates
  //! from all neurons in [bx{1,2}_start, bx{1,2}_start + bs)
  //! in input1 layer and input2_layer
  //! to all hidden neurons in [by_start, by_start + bs).
  //! - Not resets but accumulates activation inputs.
  //! \note X- means input1_layer and input2_layer are both XBatchLayer.
  void XForwardPropagateA(const BatchLayerBase &input1_layer,
                          IndexType bx1_start,
                          const BatchLayerBase &input2_layer,
                          IndexType bx2_start,
                          IndexType by_start, IndexType bs) {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == nneurons());
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input1_layer.CheckAcStates(bx1_start, bs, true));
    assert(input2_layer.CheckAcStates(bx2_start, bs, true));
    assert(CheckAcStates(by_start, bs, false));
#endif

    TIALNN_CBLAS_SBMV(CblasColMajor, CblasUpper,
                      nneurons() * bs, 0,
                      1.0f, input1_layer.activations().data() + nneurons() * bx1_start, 1,
                      input2_layer.activations().data() + nneurons() * bx2_start, 1,
                      1.0f, neurons_.activations.data() + nneurons() * by_start, 1);
  }

  //! Forward propagates (broadcast connection)
  //! from all input neurons in [bx_start, bx_start + bs)
  //! to all output neurons in [by_start, by_start + bs)
  //! with a broadcasted scaler connection.
  //! - Resets and accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardBPropagateR(const BatchLayerBase &input_layer,
                           IndexType bx_start,
                           WeightType alpha,
                           IndexType by_start, IndexType bs) {
    assert(input_layer.type() == xbatchlayer);
    assert(input_layer.nneurons() == nneurons());
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input_layer.CheckAcStates(bx_start, bs, true));
    assert(CheckAcStates(by_start, bs, true));
    set_ac_states(by_start, bs, false);
#endif

    TIALNN_CBLAS_AXPBY(nneurons() * bs,
                       alpha, input_layer.activations().data() + nneurons() * bx_start, 1,
                       0.0f, neurons_.activations.data() + nneurons() * by_start, 1);
  }

  //! Forward propagates (broadcast connection)
  //! from all input neurons in [bx_start, bx_start + bs)
  //! to all output neurons in [by_start, by_start + bs)
  //! with a broadcasted scaler connection.
  //! - Not resets but accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardBPropagateA(const BatchLayerBase &input_layer,
                           IndexType bx_start,
                           WeightType alpha,
                           IndexType by_start, IndexType bs) {
    assert(input_layer.type() == xbatchlayer);
    assert(input_layer.nneurons() == nneurons());
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input_layer.CheckAcStates(bx_start, bs, true));
    assert(CheckAcStates(by_start, bs, false));
#endif

    TIALNN_CBLAS_AXPY(nneurons() * bs,
                      alpha, input_layer.activations().data() + nneurons() * bx_start, 1,
                      neurons_.activations.data() + nneurons() * by_start, 1);
  }

  //! Elementwise forward propagates (broadcast input2 neurons)
  //! from all neurons in [bx{1,2}_start, bx{1,2}_start + bs)
  //! in input1 layer and input2_layer
  //! to all hidden neurons in [by_start, by_start + bs).
  //! - Resets and accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardBPropagateR(const BatchLayerBase &input1_layer,
                           IndexType bx1_start,
                           const BatchLayerBase &input2_layer,
                           IndexType bx2_start,
                           IndexType by_start, IndexType bs) {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == 1);
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input1_layer.CheckAcStates(bx1_start, bs, true));
    assert(input2_layer.CheckAcStates(bx2_start, bs, true));
    assert(CheckAcStates(by_start, bs, true));
    //! \note ac_states are set in for-loop.
#endif

    auto it_input2_activation = input2_layer.activations().begin();
    it_input2_activation += bx2_start; //<! input2_layer.nneurons() == 1.
    IndexType bx1 = bx1_start;
    IndexType by = by_start;
    for (IndexType k = 0; k < bs; k++) {
      XForwardBPropagateR(input1_layer, bx1,
                          *it_input2_activation,
                          by, 1);

      ++it_input2_activation;
      bx1++;
      by++;
    }
  }

  //! Elementwise forward propagates (broadcast input2 neurons)
  //! from all neurons in [bx{1,2}_start, bx{1,2}_start + bs)
  //! in input1 layer and input2_layer
  //! to all hidden neurons in [by_start, by_start + bs).
  //! - Not resets but accumulates activation inputs.
  //! \note X- means input_layer is XBatchLayer.
  void XForwardBPropagateA(const BatchLayerBase &input1_layer,
                           IndexType bx1_start,
                           const BatchLayerBase &input2_layer,
                           IndexType bx2_start,
                           IndexType by_start, IndexType bs) {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == 1);
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input1_layer.CheckAcStates(bx1_start, bs, true));
    assert(input2_layer.CheckAcStates(bx2_start, bs, true));
    assert(CheckAcStates(by_start, bs, false));
#endif

    auto it_input2_activation = input2_layer.activations().begin();
    it_input2_activation += bx2_start; //<! input2_layer.nneurons() == 1.
    IndexType bx1 = bx1_start;
    IndexType by = by_start;
    for (IndexType k = 0; k < bs; k++) {
      XForwardBPropagateA(input1_layer, bx1,
                          *it_input2_activation,
                          by, 1);

      ++it_input2_activation;
      bx1++;
      by++;
    }
  }

  //! Forward propagates
  //! from active input neurons
  //! to all hidden neurons in [by_start, by_start + bs).
  //! - Resets and accumulates activation inputs.
  //! \note Y- means input_layer is YBatchLayer.
  void YForwardPropagateR(const SparseInputYBatchLayer &input_layer,
                          YConnectionBase &conn,
                          IndexType by_start, IndexType bs) {
    assert(conn.ninput() == input_layer.nneurons());
    assert(conn.noutput() == nneurons());
    assert(bs > 0);
    assert(bs == input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckAcStates(by_start, bs, true));
    set_ac_states(by_start, bs, false);
#endif

    conn.BatchForwardPropagate(bs, 0.0f,
                               input_layer.activations(),
                               neurons_.activations, by_start);
  }

  //! Forward propagates
  //! from active input neurons
  //! to all hidden neurons in [by_start, by_start + bs).
  //! - Not resets but accumulates activation inputs.
  //! \note Y- means input_layer is YBatchLayer.
  void YForwardPropagateA(const SparseInputYBatchLayer &input_layer,
                          YConnectionBase &conn,
                          IndexType by_start, IndexType bs) {
    assert(conn.ninput() == input_layer.nneurons());
    assert(conn.noutput() == nneurons());
    assert(bs > 0);
    assert(bs == input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckAcStates(by_start, bs, false));
#endif

    conn.BatchForwardPropagate(bs, 1.0f,
                               input_layer.activations(),
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
    assert(conn.noutput() == nneurons());
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
  //! - Not resets but accumulates error inputs of input layer.
  //! - Accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateAA(BatchLayerBase &input_layer,
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

    input_layer.XBackwardPropagateAAFrom(neurons_.errors, by_start,
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

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs).
  //! - Resets and accumulates error inputs of input layer.
  //! - Not accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateRN(BatchLayerBase &input_layer,
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

    input_layer.XBackwardPropagateRNFrom(neurons_.errors, by_start,
                                         conn,
                                         bx_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs).
  //! - Not resets but accumulates error inputs of input layer.
  //! - Not accumulates gradients of connection.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardPropagateAN(BatchLayerBase &input_layer,
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

    input_layer.XBackwardPropagateANFrom(neurons_.errors, by_start,
                                         conn,
                                         bx_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + noutput)
  //! to all input1 neurons in [bx1_start, bx1_start + noutput).
  //! - Resets and accumulates error inputs of input1_layer.
  //! input1_layer.errorinputs = neurons_.errors() .* input2_layer.activations().
  //! \note X- means input1_layer and input2_layer are both XBatchLayer.
  void XBackwardPropagateR(BatchLayerBase &input1_layer,
                           IndexType bx1_start,
                           const BatchLayerBase &input2_layer,
                           IndexType bx2_start,
                           IndexType by_start, IndexType bs) const {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    // assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == nneurons());
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input2_layer.CheckAcStates(bx2_start, bs, true));
    assert(CheckErStates(by_start, bs, true));
#endif

    input1_layer.XBackwardPropagateRFrom(neurons_.errors, by_start,
                                         input2_layer.activations(), bx2_start, nneurons(),
                                         bx1_start, bs);
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + noutput)
  //! to all input1 neurons in [bx1_start, bx1_start + noutput).
  //! - Not resets but accumulates error inputs of input1_layer.
  //! input1_layer.errorinputs += neurons_.errors() .* input2_layer.activations().
  //! \note X- means input1_layer and input2_layer are both XBatchLayer.
  void XBackwardPropagateA(BatchLayerBase &input1_layer,
                           IndexType bx1_start,
                           const BatchLayerBase &input2_layer,
                           IndexType bx2_start,
                           IndexType by_start, IndexType bs) const {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    // assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == nneurons());
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(input2_layer.CheckAcStates(bx2_start, bs, true));
    assert(CheckErStates(by_start, bs, true));
#endif

    input1_layer.XBackwardPropagateAFrom(neurons_.errors, by_start,
                                         input2_layer.activations(), bx2_start, nneurons(),
                                         bx1_start, bs);
  }

  //! Backward propagates (broadcast connection)
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs)
  //! with a broadcasted scaler connection.
  //! - Resets and accumulates error inputs of input layer.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardBPropagateR(BatchLayerBase &input_layer,
                            IndexType bx_start,
                            WeightType alpha,
                            IndexType by_start, IndexType bs) const {
    assert(input_layer.type() == xbatchlayer);
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(by_start, bs, true));
#endif

    input_layer.XBackwardBPropagateRFrom(neurons_.errors, by_start,
                                         alpha, nneurons(),
                                         bx_start, bs);
  }

  //! Backward propagates (broadcast connection)
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs)
  //! with a broadcasted scaler connection.
  //! - Not resets but accumulates error inputs of input layer.
  //! \note X- means input_layer is XBatchLayer.
  void XBackwardBPropagateA(BatchLayerBase &input_layer,
                            IndexType bx_start,
                            WeightType alpha,
                            IndexType by_start, IndexType bs) const {
    assert(input_layer.type() == xbatchlayer);
    assert(bs > 0);
    assert(bx_start + bs <= input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(by_start, bs, true));
#endif

    input_layer.XBackwardBPropagateAFrom(neurons_.errors, by_start,
                                         alpha, nneurons(),
                                         bx_start, bs);
  }

  //! Backward propagates (broadcast input2 neurons)
  //! from all hidden neurons in [by_start, by_start + noutput)
  //! to all input1 neurons in [bx1_start, bx1_start + noutput).
  //! - Resets and accumulates error inputs of input1_layer.
  //! input1_layer_.errorinputs = neurons_.errors() .* input2_layer.activations().
  //! \note X- means input1_layer and input2_layer are both XBatchLayer.
  void XBackwardBPropagateR(BatchLayerBase &input1_layer,
                            IndexType bx1_start,
                            const BatchLayerBase &input2_layer,
                            IndexType bx2_start,
                            IndexType by_start, IndexType bs) const {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == 1);
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(by_start, bs, true));
#endif

    IndexType noutput = nneurons();
    auto it_input2_activation = input2_layer.activations().begin();
    it_input2_activation += bx2_start; //<! input2_layer.nneurons() == 1.
    IndexType bx1 = bx1_start;
    IndexType by = by_start;
    for (IndexType k = 0; k < bs; k++) {
      input1_layer.XBackwardBPropagateRFrom(neurons_.errors, by,
                                            *it_input2_activation, noutput,
                                            bx1, 1);
      ++it_input2_activation;
      bx1++;
      by++;
    }
  }

  //! Backward propagates (broadcast input2 neurons)
  //! from all hidden neurons in [by_start, by_start + noutput)
  //! to all input1 neurons in [bx1_start, bx1_start + noutput).
  //! - Not resets but accumulates error inputs of input1_layer.
  //! input1_layer_.errorinputs = neurons_.errors() .* input2_layer.activations().
  //! \note X- means input1_layer and input2_layer are both XBatchLayer.
  void XBackwardBPropagateA(BatchLayerBase &input1_layer,
                            IndexType bx1_start,
                            const BatchLayerBase &input2_layer,
                            IndexType bx2_start,
                            IndexType by_start, IndexType bs) const {
    assert(input1_layer.type() == xbatchlayer);
    assert(input2_layer.type() == xbatchlayer);
    assert(input1_layer.nneurons() == nneurons());
    assert(input2_layer.nneurons() == 1);
    assert(bs > 0);
    assert(bx1_start + bs <= input1_layer.batchsize());
    assert(bx2_start + bs <= input2_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(by_start, bs, true));
#endif

    IndexType noutput = nneurons();
    auto it_input2_activation = input2_layer.activations().begin();
    it_input2_activation += bx2_start; //<! input2_layer.nneurons() == 1.
    IndexType bx1 = bx1_start;
    IndexType by = by_start;
    for (IndexType k = 0; k < bs; k++) {
      input1_layer.XBackwardBPropagateAFrom(neurons_.errors, by,
                                            *it_input2_activation, noutput,
                                            bx1, 1);
      ++it_input2_activation;
      bx1++;
      by++;
    }
  }

  //! Backward propagates
  //! from all hidden neurons in [by_start, by_start + bs)
  //! to all input neurons in [bx_start, bx_start + bs)
  //! with a scaler connection.
  //! - Resets and accumulates error inputs of input layer.
  //! - Not accumulates gradients of connection.
  //! \note Y- means input_layer is YBatchLayer.
  void YBackwardPropagateNA(const SparseInputYBatchLayer &input_layer,
                            YConnectionBase &conn,
                            IndexType by_start, IndexType bs) const {
    assert(conn.ninput() == input_layer.nneurons());
    assert(conn.noutput() == nneurons());
    assert(bs == input_layer.batchsize());
    assert(by_start + bs <= batchsize());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(by_start, bs, true));
#endif

    input_layer.XBackwardPropagateNAFrom(neurons_.errors, by_start,
                                         conn);
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
    assert(bs > 0);
    assert(bx_start + bs <= batchsize());
    assert((by_start + bs) * conn.noutput() <= static_cast<IndexType>(output_errors.size()));
    assert(conn.ninput() == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx_start, bs, true));
    set_er_states(bx_start, bs, false);
    assert(CheckAcStates(bx_start, bs, true));
#endif

    conn.BatchBackwardPropagate(bs, 0.0f,
                                output_errors, by_start,
                                neurons_.errors, bx_start);
    conn.BatchAccumulateGradients(bs,
                                  output_errors, by_start,
                                  neurons_.activations, bx_start);
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
    assert(bs > 0);
    assert(bx_start + bs <= batchsize());
    assert((by_start + bs) * conn.noutput() <= static_cast<IndexType>(output_errors.size()));
    assert(conn.ninput() == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx_start, bs, false));
    assert(CheckAcStates(bx_start, bs, true));
#endif

    conn.BatchBackwardPropagate(bs, 1.0f,
                                output_errors, by_start,
                                neurons_.errors, bx_start);
    conn.BatchAccumulateGradients(bs,
                                  output_errors, by_start,
                                  neurons_.activations, bx_start);
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
    assert(bs > 0);
    assert(bx_start + bs <= batchsize());
    assert((by_start + bs) * conn.noutput() <= static_cast<IndexType>(output_errors.size()));
    assert(conn.ninput() == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckAcStates(bx_start, bs, true));
#endif

    conn.BatchAccumulateGradients(bs,
                                  output_errors, by_start,
                                  neurons_.activations, bx_start);
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
    assert(bs > 0);
    assert(bx_start + bs <= batchsize());
    assert((by_start + bs) * conn.noutput() <= static_cast<IndexType>(output_errors.size()));
    assert(conn.ninput() == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx_start, bs, true));
    set_er_states(bx_start, bs, false);
#endif

    conn.BatchBackwardPropagate(bs, 0.0f,
                                output_errors, by_start,
                                neurons_.errors, bx_start);
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
    assert(bs > 0);
    assert(bx_start + bs <= batchsize());
    assert((by_start + bs) * conn.noutput() <= static_cast<IndexType>(output_errors.size()));
    assert(conn.ninput() == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx_start, bs, false));
#endif

    conn.BatchBackwardPropagate(bs, 1.0f,
                                output_errors, by_start,
                                neurons_.errors, bx_start);
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
    assert(nneurons() * (bx2_start + bs) <= static_cast<IndexType>(input_activations.size()));
    assert(nneurons() * (by_start + bs) <= static_cast<IndexType>(output_errors.size()));
    assert(noutput == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx1_start, bs, true));
    set_er_states(bx1_start, bs, false);
#endif

    TIALNN_CBLAS_SBMV(CblasColMajor, CblasUpper,
                      nneurons() * bs, 0,
                      1.0f, input_activations.data() + nneurons() * bx2_start, 1,
                      output_errors.data() + nneurons() * by_start, 1,
                      0.0f, neurons_.errors.data() + nneurons() * bx1_start, 1);
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
    assert(nneurons() * (bx2_start + bs) <= static_cast<IndexType>(input_activations.size()));
    assert(nneurons() * (by_start + bs) <= static_cast<IndexType>(output_errors.size()));
    assert(noutput == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx1_start, bs, false));
#endif

    TIALNN_CBLAS_SBMV(CblasColMajor, CblasUpper,
                      nneurons() * bs, 0,
                      1.0f, input_activations.data() + nneurons() * bx2_start, 1,
                      output_errors.data() + nneurons() * by_start, 1,
                      1.0f, neurons_.errors.data() + nneurons() * bx1_start, 1);
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
    assert(bs > 0);
    assert(bx_start + bs <= batchsize());
    assert((by_start + bs) * noutput <= static_cast<IndexType>(output_errors.size()));
    assert(noutput == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx_start, bs, true));
    set_er_states(bx_start, bs, false);
#endif

    TIALNN_CBLAS_AXPBY(nneurons() * bs,
                       alpha, output_errors.data() + nneurons() * by_start, 1,
                       0.0f, neurons_.errors.data() + nneurons() * bx_start, 1);
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
    assert(bs > 0);
    assert(bx_start + bs <= batchsize());
    assert((by_start + bs) * noutput <= static_cast<IndexType>(output_errors.size()));
    assert(noutput == nneurons());
#ifdef _TIALNN_DEBUG
    assert(CheckErStates(bx_start, bs, false));
#endif

    TIALNN_CBLAS_AXPBY(nneurons() * bs,
                       alpha, output_errors.data() + nneurons() * by_start, 1,
                       1.0f, neurons_.errors.data() + nneurons() * bx_start, 1);
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

    const IndexType n = nneurons() * batchsize();
    neurons_.nneurons = nneurons();
    neurons_.batchsize = batchsize();
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

  //! Neurons.
  Neurons neurons_;

  //! Dummy variable for BatchLayerBase interface.
  std::vector<IndexType> dummy_active_neurons_;
};

} // namespace tialnn

#endif
