/*!
 * \file cmemnet_projection_unit.h
 *
 * 
 * \version 0.0.6
 */

#ifndef CMEMNET_CMEMNET_PROJECTION_UNIT_H_
#define CMEMNET_CMEMNET_PROJECTION_UNIT_H_

// assert
#include <cassert>
// std::vector
#include <vector>
// tialnn::IndexType
#include <tialnn/base.h>
// tialnn::leaky_relu_f
// tialnn::leaky_relu_g
#include <tialnn/util/numeric.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::GenericNeurons
#include <tialnn/neuron/generic_neurons.h>
// tialnn::XConnectionBase
#include <tialnn/connection/xconnection_base.h>
// tialnn::XConnectionInputMajor
#include <tialnn/connection/xconnection_inputmajor.h>
// tialnn::XConnectionOutputShared
#include <tialnn/connection/xconnection_outputshared.h>

namespace cmemnet {

//! Projection unit for CMemNet.
struct CMemNetProjectionUnit {
  //! Constructor
  CMemNetProjectionUnit() : capacity(0), size(0), batchsize(0),
    ptr_conn_composition_projection(nullptr),
    ptr_conn_bias_projection(nullptr),
    ptr_conn_globalbias_projection(nullptr),
    ptr_bias_layer(nullptr) {}
  //! Destructor
  ~CMemNetProjectionUnit() {
    assert(!ptr_conn_composition_projection);
    assert(!ptr_conn_bias_projection);
    assert(!ptr_conn_globalbias_projection);
  }

  //! Allocates the unit
  template <class GradientSolver>
  void AllocateUnit(tialnn::IndexType c,
                    tialnn::IndexType bs,
                    tialnn::IndexType ncomp,
                    tialnn::IndexType nproj) {
    assert(capacity == 0);
    assert(batchsize == 0);

    assert(!ptr_conn_composition_projection);
    assert(!ptr_conn_bias_projection);
    assert(!ptr_conn_globalbias_projection);

    assert(ptr_bias_layer);

    capacity = c;
    size = 0;
    batchsize = bs;

    ptr_conn_composition_projection = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_bias_projection = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_globalbias_projection = new tialnn::XConnectionOutputShared<GradientSolver>();

    ptr_conn_composition_projection->set_dims(ncomp, nproj);
    ptr_conn_bias_projection->set_dims(1, nproj);
    ptr_conn_globalbias_projection->set_dims(1, nproj);

    projection_layer.set_nneurons_batchsize(nproj, bs * c);
  }

  //! Destroys the unit.
  void DestroyUnit() {
    assert(ptr_conn_composition_projection);
    assert(ptr_conn_bias_projection);
    assert(ptr_conn_globalbias_projection);

    delete ptr_conn_composition_projection;
    delete ptr_conn_bias_projection;
    delete ptr_conn_globalbias_projection;

    ptr_conn_composition_projection = nullptr;
    ptr_conn_bias_projection = nullptr;
    ptr_conn_globalbias_projection = nullptr;
  }

  //! Capacity of the unit.
  tialnn::IndexType capacity;
  //! Current size of the unit.
  //! It should always be no more than capacity.
  tialnn::IndexType size;
  //! Batch size.
  tialnn::IndexType batchsize;

  //! Connection: composition -> projection
  tialnn::XConnectionBase *ptr_conn_composition_projection;
  //! Neuron-depedent bias for the projection layer.
  tialnn::XConnectionBase *ptr_conn_bias_projection;
  //! Neuron-indepedent bias for the projection layer.
  tialnn::XConnectionBase *ptr_conn_globalbias_projection;

  //! Pointer to the bias layer (shared by the whole network). 
  const tialnn::XBatchLayer<tialnn::IdentityNeurons> *ptr_bias_layer;

  //! Projection layer.
  //! (nh, bs * c)
  tialnn::XBatchLayer<tialnn::GenericNeurons<tialnn::leaky_relu_f, tialnn::leaky_relu_g>> projection_layer;
};

} // namespace cmemnet

#endif
