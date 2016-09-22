/*!
 * \file attention_softmax_neurons.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_NEURON_ATTENTION_SOFTMAX_NEURONS_H_
#define TIALNN_NEURON_ATTENTION_SOFTMAX_NEURONS_H_

// assert
#include <cassert>
// std::vector
#include <vector>
// std::transform
#include <algorithm>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
#include "../base.h"
// tialnn::exp_f
// tialnn::exp_g
// TIALNN_CBLAS_XXX
#include "../util/numeric.h"

#ifdef _TIALNN_DEBUG
// std::set
#include <set>
#endif

namespace tialnn {

//! Attention softmax neurons in 3-dimension.
//! The normalization is performed over the third dimension.
//! Stores the attention weights.
//! (1, bs, c).
//! \note Currently, only supports nneurons == 1.
struct AttentionSoftmaxNeurons {
  //! Constructor.
  explicit AttentionSoftmaxNeurons() : nneurons(1), atom_batchsize(0), capacity(0) {}
  //! Destructor.
  ~AttentionSoftmaxNeurons() {}

  //! Returns the activation of a neuron.
  ActivationType get_activations(IndexType idx, IndexType k, IndexType t) const {
    assert(idx == 0);
    assert(nneurons == 1);
    assert(k < atom_batchsize);
    assert(t < capacity);
#ifdef _TIALNN_DEBUG
    assert(ac_states.find(atom_batchsize * t + k) != ac_states.end());
#endif

    return activations[atom_batchsize * t + k];
  }
  //! Returns the error of a neuron.
  ErrorType get_errors(IndexType idx, IndexType k, IndexType t) const {
    assert(idx == 0);
    assert(nneurons == 1);
    assert(k < atom_batchsize);
    assert(t < capacity);
#ifdef _TIALNN_DEBUG
    assert(er_states.find(atom_batchsize * t + k) != er_states.end());
#endif

    return errors[atom_batchsize * t + k];
  }

  //! Computes activations of neurons in [:, :, 0:c] according to their inputs.
  //! Normalizes over the third dimension.
  void ComputeActivations(IndexType c) {
    assert(c > 0);
    assert(c <= capacity);
    assert(!activations.empty());
    assert(nneurons == 1);
    assert(atom_batchsize > 0);
    assert(capacity > 0);
    assert(nneurons * atom_batchsize * capacity == activations.size());
#ifdef _TIALNN_DEBUG
    for (IndexType kt = 0; kt < atom_batchsize * c; kt++) {
      assert(ac_states.find(kt) == ac_states.end());
      ac_states.insert(kt);
    }
#endif

    std::vector<ActivationType>::iterator it_ac_start = activations.begin();
    ActivationType *px = activations.data();
    //! For numerical stability, we substract the max batch-by-batch.
    for (IndexType k = 0; k < atom_batchsize; k++) {
      ActivationType max_ac = *it_ac_start;
      std::vector<ActivationType>::iterator it = it_ac_start + atom_batchsize;
      for (IndexType t = 1; t < c; t++) {
        if (max_ac < *it) {
          max_ac = *it;
        }
        it += atom_batchsize;
      }

      TIALNN_CBLAS_AXPY(c, -1.0f, &max_ac, 0, px, atom_batchsize);

      ++it_ac_start;
      ++px;
    }

    std::transform(activations.begin(),
                   activations.begin() + atom_batchsize * c,
                   activations.begin(),
                   exp_f(0));

    px = activations.data();
    for (IndexType k = 0; k < atom_batchsize; k++) {
      ActivationType sum = TIALNN_CBLAS_ASUM(c, px, atom_batchsize);
      TIALNN_CBLAS_SCAL(c, 1.0f / sum, px, atom_batchsize);
      ++px;
    }
  }

  //! Computes activations of neurons in [:, :, 0:c] according to their inputs.
  //! Normalizes over the third dimension.
  void ComputeErrors(IndexType c) {
    assert(c > 0);
    assert(c <= capacity);
    assert(!activations.empty());
    assert(!errors.empty());
    assert(activations.size() == errors.size());
    assert(nneurons == 1);
    assert(atom_batchsize > 0);
    assert(capacity > 0);
#ifdef _TIALNN_DEBUG
    for (IndexType kt = 0; kt < atom_batchsize * c; kt++) {
      assert(ac_states.find(kt) != ac_states.end());
      assert(er_states.find(kt) == er_states.end());
      er_states.insert(kt);
    }
#endif

    const ActivationType *px = activations.data();
    ErrorType *py = errors.data();
    for (IndexType k = 0; k < atom_batchsize; k++) {
      ErrorType sum = TIALNN_CBLAS_DOT(c,
                                       py, atom_batchsize,
                                       px, atom_batchsize);
      TIALNN_CBLAS_AXPY(c,
                        -1.0f, &sum, 0,
                        py, atom_batchsize);

      ++py;
      ++px;
    }

    std::vector<ActivationType>::iterator it_ac_start = activations.begin();
    std::vector<ErrorType>::iterator it_er_start = errors.begin();
    std::transform(it_er_start,
                   it_er_start + atom_batchsize * c,
                   it_ac_start,
                   it_er_start,
                   exp_g());
  }

#ifdef _TIALNN_DEBUG
  //! The state of the neuron activations for each batch element.
  //! not in ac_states - in weighted sum state;
  //! in ac_states     - in activation state.
  std::set<IndexType> ac_states;
  //! The state of the neuron errors for each batch element.
  //! not in er_states - in weighted sum state;
  //! in er_states     - in error state.
  std::set<IndexType> er_states;
#endif


  //! Number of neurons.
  const IndexType nneurons;
  //! Atom batch size.
  IndexType atom_batchsize;
  //! Capacity.
  IndexType capacity;

  //! The activations of the neurons.
  std::vector<ActivationType> activations;
  //! The errors of the neurons.
  std::vector<ErrorType> errors;
};

} // namespace xmemnet

#endif
