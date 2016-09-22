/*!
 * \file sparse_input_ybatchlayer.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_LAYER_SPARSE_INPUT_YBATCHLAYER_H_
#define TIALNN_LAYER_SPARSE_INPUT_YBATCHLAYER_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::vector
#include <vector>
// std::unordered_map
#include <unordered_map>
// std::fill
#include <algorithm>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
#include "../base.h"
// tialnn::YConnectionBase
#include "../connection/yconnection_base.h"

namespace tialnn {

//! Sparse input y-batchLayer.
//! - It has similar interface of YBatchLayer but is not a derived class.
//! - It only has activations as an unordered_map.
class SparseInputYBatchLayer {
 public:
  //! Constructor.
  explicit SparseInputYBatchLayer() : nneurons_(0), batchsize_(0) {}
  //! Destructor.
  virtual ~SparseInputYBatchLayer() {}

  //! Returns the number of neurons.
  IndexType nneurons() const { return nneurons_; }
  //! Returns the batch size.
  IndexType batchsize() const { return batchsize_; }

  //! Returns the activations of all neurons.
  const std::unordered_map<IndexType, ActivationType>& activations() const {
    return activations_;
  }
  //! Returns the activation of a neuron.
  ActivationType activations(IndexType idx, IndexType k) const {
    assert(idx < nneurons_);
    assert(k < batchsize_);
    std::unordered_map<IndexType, ActivationType>::const_iterator it = activations_.find(nneurons_ * k + idx);
    if (it != activations_.end()) {
      return it->second;
    } else {
      return 0;
    }
  }

  //! Sets the number of neurons and batch size.
  //! Automatically calls AllocateLayer().
  void set_nneurons_batchsize(IndexType n, IndexType b) {
    assert(nneurons_ == 0);
    assert(batchsize_ == 0);

    nneurons_ = n;
    batchsize_ = b;
    AllocateLayer();
  }
  //! Sets activation of a neuron.
  void set_activations(IndexType idx, IndexType k, ActivationType val) {
    assert(idx < nneurons_);
    assert(k < batchsize_);
    if (val == 0) {
      std::unordered_map<IndexType, ActivationType>::const_iterator it = activations_.find(nneurons_ * k + idx);
      if (it != activations_.end()) {
        activations_.erase(it);
      }
    } else {
      activations_[nneurons_ * k + idx] = val;
    }
  }
  //! Accumulates activation of a neuron.
  void accumulate_activations(IndexType idx, IndexType k, ActivationType val) {
    assert(idx < nneurons_);
    assert(k < batchsize_);
    
    std::unordered_map<IndexType, ActivationType>::iterator it = activations_.find(nneurons_ * k + idx);
    if (it == activations_.end()) {
      activations_[nneurons_ * k + idx] = val;
    } else {
      if (it->second == -val) {
        activations_.erase(it);
      } else {
        it->second += val;
      }
    }
  }

  //! Sets all activations to value.
  void SetActivationsToValue(ActivationType val) {
    if (val == 0) {
      activations_.clear();
    } else {
      for (IndexType i = 0; i < nneurons_ * batchsize_; i++) {
        activations_[i] = val;
      }
    }
  }

  //! Backward propagates
  //! from all output neurons in [by_start, by_start + batchsize())
  //! to all hidden neurons.
  //! - Only accumulates gradients of connection.
  //! \note X- means output_layer is XBatchLayer.
  void XBackwardPropagateNAFrom(const std::vector<ErrorType> &output_errors,
                                IndexType by_start,
                                YConnectionBase &conn) const {
    assert(conn.ninput() == nneurons_);
    assert((by_start + batchsize_) * conn.noutput() <= static_cast<IndexType>(output_errors.size()));

    conn.BatchAccumulateGradients(batchsize(),
                                  output_errors, by_start,
                                  activations_);
  }

 private:
  //! Allocates the layer.
  void AllocateLayer() {
    if (nneurons() == 0) {
      TIALNN_ERR("nneurons_ should be greater than 0!");
      std::exit(EXIT_FAILURE);
    }
    if (batchsize() == 0) {
      TIALNN_ERR("batchsize_ should be greater than 0!");
      std::exit(EXIT_FAILURE);
    }

    assert(activations_.empty());
  };

  //! Number of neurons.
  IndexType nneurons_;
  //! Batch size.
  IndexType batchsize_;

  //! Activations.
  std::unordered_map<IndexType, ActivationType> activations_;
};

} // namespace tialnn

#endif
