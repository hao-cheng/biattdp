/*!
 * \file cmemnet_attention_unit.h
 *
 * 
 * \version 0.0.6
 */

#ifndef CMEMNET_CMEMNET_ATTENTION_UNIT_H_
#define CMEMNET_CMEMNET_ATTENTION_UNIT_H_

// std::vector
#include <vector>
// tialnn::IndexType
// tialnn::ErrorType
#include <tialnn/base.h>
// tialnn::tanh_f
// tialnn::tanh_g
#include <tialnn/util/numeric.h>
// tialnn::GenericNeurons
#include <tialnn/neuron/generic_neurons.h>
// tialnn::IdentityNeurons
#include <tialnn/neuron/identity_neurons.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::AttentionSoftmaxXBatchLayer
#include <tialnn/layer/attention_softmax_xbatchlayer.h>

namespace cmemnet {

//! Single directional Attention unit for CMemNet.
struct CMemNetAttentionUnit {
  //! Constructor.
  CMemNetAttentionUnit() : capacity(0), size(0), batchsize(0),
    ptr_attention_conn_hidden0_hidden1(nullptr),
    ptr_attention_conn_bias_hidden1(nullptr),
    ptr_bias_layer(nullptr) {}
  //! Destructor.
  ~CMemNetAttentionUnit() {}

  //! Allocates the unit.
  //! \param c        Capacity of the attention unit.
  void AllocateUnit(tialnn::IndexType c,
                    tialnn::IndexType bs,
                    tialnn::IndexType nh) {
    assert(capacity == 0);
    assert(batchsize == 0);
    assert(ptr_attention_conn_hidden0_hidden1);
    assert(ptr_attention_conn_bias_hidden1);
    assert(ptr_bias_layer);

    capacity = c;
    size = 0;
    batchsize = bs;

    hidden0_layer.set_nneurons_batchsize(nh, bs * c);
    hidden1_layer.set_atom_batchsize(bs);
    hidden1_layer.set_capacity(c);
    hidden1_layer.set_nneurons_batchsize(1, bs * c);

    //! \note we reserve the memory for hidden1_label_errors.
    hidden1_label_errors.resize(bs * c);

    l2rhidden_layer.set_nneurons_batchsize(nh, bs);
    r2lhidden_layer.set_nneurons_batchsize(nh, bs);
  }

  //! Capacity of the unit.
  tialnn::IndexType capacity;
  //! Current size of the unit.
  //! It should always be smaller than capacity.
  tialnn::IndexType size;
  //! Batch size.
  tialnn::IndexType batchsize;

  //! Stores tanh(encoderatom.memin_layer + decdoeratom.memnin_layer).
  //! (nh, bs * c)
  tialnn::XBatchLayer<tialnn::GenericNeurons<tialnn::tanh_f, tialnn::tanh_g>> hidden0_layer;
  //! Attention weight for each encoder memout layers.
  //! stores softmax( v.T * [atom.hidden0_layer for atom in attentionatoms] ).
  //! (1, bs * c) or (bs, c)
  tialnn::AttentionSoftmaxXBatchLayer hidden1_layer;
  //! Errors for hidden1 layer from attention labels.
  //! (1, bs * c) or (bs, c)
  std::vector<tialnn::ErrorType> hidden1_label_errors;

  //! Attention weighted sum of encoder l2r memout layers.
  //! (nh, bs)
  tialnn::XBatchLayer<tialnn::IdentityNeurons> l2rhidden_layer;
  //! Attention weighted sum of encoder r2l memout layers.
  //! (nh, bs)
  tialnn::XBatchLayer<tialnn::IdentityNeurons> r2lhidden_layer;

  //! Pointer to connection: attention hidden0 -> attention hidden1.
  tialnn::XConnectionBase *ptr_attention_conn_hidden0_hidden1;
  //! Pointer to connection: attention bias -> attention hidden1.
  //! \note since attention_hidden1 has a single neuron, bias acts as the globalbias.
  tialnn::XConnectionBase *ptr_attention_conn_bias_hidden1;

  //! Pointer to the bias layer (shared by the whole network).
  const tialnn::XBatchLayer<tialnn::IdentityNeurons> *ptr_bias_layer;
};

} // namespace cmemnet

#endif
