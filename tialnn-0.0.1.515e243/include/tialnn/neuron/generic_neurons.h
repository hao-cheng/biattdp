/*!
 * \file generic_neurons.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_NEURON_GENERIC_NEURONS_H_
#define TIALNN_NEURON_GENERIC_NEURONS_H_

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

#ifdef _TIALNN_DEBUG
// std::set
#include <set>
#endif

namespace tialnn {

//! Generic neurons.
//! \note We use struct for neurons to allow public access to the value.
//!       The derived classes can have private member if that makes sense.
template <class Func_f, class Func_g>
struct GenericNeurons {
  //! Constructor.
  explicit GenericNeurons() : nneurons(0), batchsize(0) {}
  //! Destructor.
  ~GenericNeurons() {}

  //! Returns the activation of a neuron.
  ActivationType get_activations(IndexType idx, IndexType k) const {
    assert(idx < nneurons);
    assert(k < batchsize);
#ifdef _TIALNN_DEBUG
    assert(ac_states.find(k) != ac_states.end());
#endif

    return activations[nneurons * k + idx];
  }
  //! Returns the error of a neuron.
  ErrorType get_errors(IndexType idx, IndexType k) const {
    assert(idx < nneurons);
    assert(k < batchsize);
#ifdef _TIALNN_DEBUG
    assert(er_states.find(k) != er_states.end());
#endif

    return errors[nneurons * k + idx];
  }

  //! Computes all neuron activations according to their inputs.
  void ComputeActivations() {
    assert(!activations.empty());
    assert(nneurons > 0);
    assert(batchsize > 0);
    assert(nneurons * batchsize == activations.size());
#ifdef _TIALNN_DEBUG
    for (IndexType k = 0; k < batchsize; k++) {
      assert(ac_states.find(k) == ac_states.end());
      ac_states.insert(k);
    }
#endif

    std::vector<ActivationType>::iterator it_ac_start = activations.begin();
    std::transform(it_ac_start,
                   activations.end(),
                   it_ac_start,
                   Func_f());
  }

  //! Computes all neuron errors according to their inputs.
  void ComputeErrors() {
    assert(!activations.empty());
    assert(!errors.empty());
    assert(activations.size() == errors.size());
    assert(nneurons > 0);
    assert(batchsize > 0);
    assert(nneurons * batchsize == errors.size());
#ifdef _TIALNN_DEBUG
    for (IndexType k = 0; k < batchsize; k++) {
      assert(ac_states.find(k) != ac_states.end());
      assert(er_states.find(k) == er_states.end());
      er_states.insert(k);
    }
#endif

    std::vector<ActivationType>::iterator it_ac_start = activations.begin();
    std::vector<ErrorType>::iterator it_er_start = errors.begin();
    std::transform(it_er_start,
                   errors.end(),
                   it_ac_start,
                   it_er_start,
                   Func_g());
  }

  //! Computes activations of neurons in [b_start, b_start + bs) according to their inputs.
  void ComputeActivations(IndexType b_start, IndexType bs) {
    assert(!activations.empty());
    assert(nneurons > 0);
    assert(batchsize > 0);
    assert(nneurons * batchsize == activations.size());
    assert(bs > 0);
    assert(b_start + bs <= batchsize);
#ifdef _TIALNN_DEBUG
    for (IndexType k = b_start; k < b_start + bs; k++) {
      assert(ac_states.find(k) == ac_states.end());
      ac_states.insert(k);
    }
#endif

    std::vector<ActivationType>::iterator it_ac_start = activations.begin() + nneurons * b_start;
    std::transform(it_ac_start,
                   it_ac_start + nneurons * bs,
                   it_ac_start,
                   Func_f());
  }

  //! Computes errors of neurons in [b_start, b_start + bs) according to their inputs.
  void ComputeErrors(IndexType b_start, IndexType bs) {
    assert(!activations.empty());
    assert(!errors.empty());
    assert(activations.size() == errors.size());
    assert(nneurons > 0);
    assert(batchsize > 0);
    assert(nneurons * batchsize == errors.size());
    assert(bs > 0);
    assert(b_start + bs <= batchsize);
#ifdef _TIALNN_DEBUG
    for (IndexType k = b_start; k < b_start + bs; k++) {
      assert(ac_states.find(k) != ac_states.end());
      assert(er_states.find(k) == er_states.end());
      er_states.insert(k);
    }
#endif

    IndexType offset = nneurons * b_start;
    std::vector<ActivationType>::iterator it_ac_start = activations.begin() + offset;
    std::vector<ErrorType>::iterator it_er_start = errors.begin() + offset;
    std::transform(it_er_start,
                   it_er_start + nneurons * bs,
                   it_ac_start,
                   it_er_start,
                   Func_g());
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
  IndexType nneurons;
  //! Batch size.
  IndexType batchsize;

  //! The activations of the neurons.
  std::vector<ActivationType> activations;
  //! The errors of the neurons.
  std::vector<ErrorType> errors;
};

} // namespace tialnn

#endif