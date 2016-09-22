/*!
 * \file cmemnet_encoder_unit.h
 *
 * 
 * \version 0.0.6
 */

#ifndef CMEMNET_CMEMNET_ENCODER_UNIT_H_
#define CMEMNET_CMEMNET_ENCODER_UNIT_H_

// assert
#include <cassert>
// std::vector
#include <vector>
// tialnn::IndexType
#include <tialnn/base.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::IdentityNeurons
#include <tialnn/neuron/identity_neurons.h>
// tialnn::GenericNeurons
#include <tialnn/neuron/generic_neurons.h>
// tialnn::XConnectionBase
#include <tialnn/connection/xconnection_base.h>
// tialnn::XConnectionInputMajor
#include <tialnn/connection/xconnection_inputmajor.h>
// tialnn::XConnectionOutputShared
#include <tialnn/connection/xconnection_outputshared.h>
// tialnn::GatedRecurrentXUnit
#include <tialnn/unit/gated_recurrent_xunit.h>
// tialnn::leaky_relu_f
// tialnn::leaky_relu_g
#include <tialnn/util/numeric.h>

namespace cmemnet {

struct CMemNetEncoderSubUnit {
  //! Constructor.
  CMemNetEncoderSubUnit() : capacity(0), size(0), batchsize(0),
    ptr_conn_projection_memoutreset(nullptr),
    ptr_conn_projection_memoutupdate(nullptr),
    ptr_conn_projection_memouthtilde(nullptr),
    ptr_conn_memout_memin(nullptr),
    ptr_memout_conn_hprev_reset(nullptr),
    ptr_memout_conn_hprev_update(nullptr),
    ptr_memout_conn_hprev_htilde(nullptr),
    ptr_memout_conn_bias_reset(nullptr),
    ptr_memout_conn_bias_update(nullptr),
    ptr_memout_conn_bias_htilde(nullptr),
    ptr_memout_conn_globalbias_reset(nullptr),
    ptr_memout_conn_globalbias_update(nullptr),
    ptr_memout_conn_globalbias_htilde(nullptr),
    ptr_init_hidden_layer(nullptr), ptr_bias_layer(nullptr), ptr_negones_layer(nullptr) {}
  //! Destructor.
  ~CMemNetEncoderSubUnit() {
    assert(!ptr_conn_projection_memoutreset);
    assert(!ptr_conn_projection_memoutupdate);
    assert(!ptr_conn_projection_memouthtilde);

    assert(!ptr_conn_memout_memin);

    assert(!ptr_memout_conn_hprev_reset);
    assert(!ptr_memout_conn_hprev_update);
    assert(!ptr_memout_conn_hprev_htilde);
    assert(!ptr_memout_conn_bias_reset);
    assert(!ptr_memout_conn_bias_update);
    assert(!ptr_memout_conn_bias_htilde);
    assert(!ptr_memout_conn_globalbias_reset);
    assert(!ptr_memout_conn_globalbias_update);
    assert(!ptr_memout_conn_globalbias_htilde);
  }

  //! Allocates the unit.
  template <class GradientSolver>
  void AllocateUnit(tialnn::IndexType c,
                    tialnn::IndexType bs,
                    tialnn::IndexType nproj,
                    tialnn::IndexType nh) {
    assert(capacity == 0);
    assert(batchsize == 0);

    assert(!ptr_conn_projection_memoutreset);
    assert(!ptr_conn_projection_memoutupdate);
    assert(!ptr_conn_projection_memouthtilde);

    assert(!ptr_conn_memout_memin);

    assert(!ptr_memout_conn_hprev_reset);
    assert(!ptr_memout_conn_hprev_update);
    assert(!ptr_memout_conn_hprev_htilde);
    assert(!ptr_memout_conn_bias_reset);
    assert(!ptr_memout_conn_bias_update);
    assert(!ptr_memout_conn_bias_htilde);
    assert(!ptr_memout_conn_globalbias_reset);
    assert(!ptr_memout_conn_globalbias_update);
    assert(!ptr_memout_conn_globalbias_htilde);

    assert(ptr_init_hidden_layer);
    assert(ptr_bias_layer);
    assert(ptr_negones_layer);

    capacity = c;
    size = 0;
    batchsize = bs;

    ptr_conn_projection_memoutreset = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_projection_memoutupdate = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_projection_memouthtilde = new tialnn::XConnectionInputMajor<GradientSolver>();

    ptr_conn_memout_memin = new tialnn::XConnectionInputMajor<GradientSolver>();

    ptr_memout_conn_hprev_reset = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_memout_conn_hprev_update = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_memout_conn_hprev_htilde = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_memout_conn_bias_reset = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_memout_conn_bias_update = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_memout_conn_bias_htilde = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_memout_conn_globalbias_reset = new tialnn::XConnectionOutputShared<GradientSolver>();
    ptr_memout_conn_globalbias_update = new tialnn::XConnectionOutputShared<GradientSolver>();
    ptr_memout_conn_globalbias_htilde = new tialnn::XConnectionOutputShared<GradientSolver>();

    ptr_conn_projection_memoutreset->set_dims(nproj, nh);
    ptr_conn_projection_memoutupdate->set_dims(nproj, nh);
    ptr_conn_projection_memouthtilde->set_dims(nproj, nh);

    ptr_conn_memout_memin->set_dims(nh, nh);

    ptr_memout_conn_hprev_reset->set_dims(nh, nh);
    ptr_memout_conn_hprev_update->set_dims(nh, nh);
    ptr_memout_conn_hprev_htilde->set_dims(nh, nh);
    ptr_memout_conn_bias_reset->set_dims(1, nh);
    ptr_memout_conn_bias_update->set_dims(1, nh);
    ptr_memout_conn_bias_htilde->set_dims(1, nh);
    ptr_memout_conn_globalbias_reset->set_dims(1, nh);
    ptr_memout_conn_globalbias_update->set_dims(1, nh);
    ptr_memout_conn_globalbias_htilde->set_dims(1, nh);

    memout_layer.set_atom_batchsize(bs);
    memout_layer.set_capacity(c);
    memout_layer.set_ptr_negones(ptr_negones_layer);
    memout_layer.set_ptr_conn_hprev_reset(ptr_memout_conn_hprev_reset);
    memout_layer.set_ptr_conn_hprev_update(ptr_memout_conn_hprev_update);
    memout_layer.set_ptr_conn_hprev_htilde(ptr_memout_conn_hprev_htilde);
    memout_layer.set_ptr_bias_layer(ptr_bias_layer);
    memout_layer.set_ptr_conn_bias_reset(ptr_memout_conn_bias_reset);
    memout_layer.set_ptr_conn_bias_update(ptr_memout_conn_bias_update);
    memout_layer.set_ptr_conn_bias_htilde(ptr_memout_conn_bias_htilde);
    memout_layer.set_ptr_conn_globalbias_reset(ptr_memout_conn_globalbias_reset);
    memout_layer.set_ptr_conn_globalbias_update(ptr_memout_conn_globalbias_update);
    memout_layer.set_ptr_conn_globalbias_htilde(ptr_memout_conn_globalbias_htilde);
    memout_layer.set_nneurons_batchsize(nh, bs * c);

    memin_layer.set_nneurons_batchsize(nh, bs * c);
  }

  //! Destroys the unit.
  void DestroyUnit() {
    assert(ptr_conn_projection_memoutreset);
    assert(ptr_conn_projection_memoutupdate);
    assert(ptr_conn_projection_memouthtilde);

    assert(ptr_conn_memout_memin);

    assert(ptr_memout_conn_hprev_reset);
    assert(ptr_memout_conn_hprev_update);
    assert(ptr_memout_conn_hprev_htilde);
    assert(ptr_memout_conn_bias_reset);
    assert(ptr_memout_conn_bias_update);
    assert(ptr_memout_conn_bias_htilde);
    assert(ptr_memout_conn_globalbias_reset);
    assert(ptr_memout_conn_globalbias_update);
    assert(ptr_memout_conn_globalbias_htilde);

    delete ptr_conn_projection_memoutreset;
    delete ptr_conn_projection_memoutupdate;
    delete ptr_conn_projection_memouthtilde;

    delete ptr_conn_memout_memin;

    delete ptr_memout_conn_hprev_reset;
    delete ptr_memout_conn_hprev_update;
    delete ptr_memout_conn_hprev_htilde;
    delete ptr_memout_conn_bias_reset;
    delete ptr_memout_conn_bias_update;
    delete ptr_memout_conn_bias_htilde;
    delete ptr_memout_conn_globalbias_reset;
    delete ptr_memout_conn_globalbias_update;
    delete ptr_memout_conn_globalbias_htilde;

    ptr_conn_projection_memoutreset = nullptr;
    ptr_conn_projection_memoutupdate = nullptr;
    ptr_conn_projection_memouthtilde = nullptr;

    ptr_conn_memout_memin = nullptr;

    ptr_memout_conn_hprev_reset = nullptr;
    ptr_memout_conn_hprev_update = nullptr;
    ptr_memout_conn_hprev_htilde = nullptr;
    ptr_memout_conn_bias_reset = nullptr;
    ptr_memout_conn_bias_update = nullptr;
    ptr_memout_conn_bias_htilde = nullptr;
    ptr_memout_conn_globalbias_reset = nullptr;
    ptr_memout_conn_globalbias_update = nullptr;
    ptr_memout_conn_globalbias_htilde = nullptr;
  }

  //! Capacity of the unit.
  tialnn::IndexType capacity;
  //! Current size of the unit.
  //! It should always be smaller than capacity.
  tialnn::IndexType size;
  //! Batch size.
  tialnn::IndexType batchsize;

  //! Connection: projection -> memout reset gate.
  tialnn::XConnectionBase *ptr_conn_projection_memoutreset;
  //! Connection: projection -> memout update gate.
  tialnn::XConnectionBase *ptr_conn_projection_memoutupdate;
  //! Connection: projection -> memout htilde.
  tialnn::XConnectionBase *ptr_conn_projection_memouthtilde;

  //! Connection: memout hidden -> memin.
  tialnn::XConnectionBase *ptr_conn_memout_memin;

  //! Connection: memout hprev -> memout reset gate.
  tialnn::XConnectionBase *ptr_memout_conn_hprev_reset;
  //! Connection: memout hprev -> memout update gate.
  tialnn::XConnectionBase *ptr_memout_conn_hprev_update;
  //! Connection: memout hprev -> memout htilde.
  tialnn::XConnectionBase *ptr_memout_conn_hprev_htilde;
  //! Connection: memout bias -> memout reset gate.
  tialnn::XConnectionBase *ptr_memout_conn_bias_reset;
  //! Connection: memout bias -> memout update gate.
  tialnn::XConnectionBase *ptr_memout_conn_bias_update;
  //! Connection: memout bias -> memout htilde.
  tialnn::XConnectionBase *ptr_memout_conn_bias_htilde;
  //! Connection: memout globalbias -> memout reset gate.
  tialnn::XConnectionBase *ptr_memout_conn_globalbias_reset;
  //! Connection: memout globalbias -> memout update gate.
  tialnn::XConnectionBase *ptr_memout_conn_globalbias_update;
  //! Connection: memout globalbias -> memout htilde.
  tialnn::XConnectionBase *ptr_memout_conn_globalbias_htilde;

  //! Output memory layer.
  //! Gated recurrent x-unit.
  //! (nh, bs * c)
  tialnn::GatedRecurrentXUnit<tialnn::GenericNeurons<tialnn::leaky_relu_f, tialnn::leaky_relu_g>> memout_layer;
  //! Input memory layer.
  //! (nh, bs * c)
  tialnn::XBatchLayer<tialnn::IdentityNeurons> memin_layer;

  //! Initial memout layer.
  //! Errors will be set but never used.
  const tialnn::XBatchLayer<tialnn::IdentityNeurons> *ptr_init_hidden_layer;

  //! Pointer to the bias layer (shared by the whole network).
  const tialnn::XBatchLayer<tialnn::IdentityNeurons> *ptr_bias_layer;

  //! Pointer to auxilary layer of all -1s.
  //! Used for memout.
  //! \note It points to a constant layer.
  const tialnn::XBatchLayer<tialnn::IdentityNeurons> *ptr_negones_layer;
};


//! Encoder unit for CMemNet.
struct CMemNetEncoderUnit {
  //! Constructor.
  CMemNetEncoderUnit() : capacity(0), size(0), batchsize(0) {}
  //! Destructor.
  ~CMemNetEncoderUnit() {}

  //! Allocates the unit.
  template <class GradientSolver>
  void AllocateUnit(tialnn::IndexType c,
                    tialnn::IndexType bs,
                    tialnn::IndexType nproj,
                    tialnn::IndexType nh) {
    assert(capacity == 0);
    assert(batchsize == 0);

    capacity = c;
    size = 0;
    batchsize = bs;

    l2r.AllocateUnit<GradientSolver>(c, bs, nproj, nh);
    r2l.AllocateUnit<GradientSolver>(c, bs, nproj, nh);
  }

  //! Destroys the unit.
  void DestroyUnit() {
    l2r.DestroyUnit();
    r2l.DestroyUnit();
  }

  //! Capacity of the unit.
  tialnn::IndexType capacity;
  //! Current size of the unit.
  //! It should always be smaller than capacity.
  tialnn::IndexType size;
  //! Batch size.
  tialnn::IndexType batchsize;

  //! Left-to-right encoder unit.
  CMemNetEncoderSubUnit l2r;
  //! Right-to-left encoder unit.
  CMemNetEncoderSubUnit r2l;
};

} // namespace cmemnet

#endif
