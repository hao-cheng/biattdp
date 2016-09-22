/*!
 * \file cmemnet_input_unit.h
 *
 * 
 * \version 0.0.6
 */

#ifndef CMEMNET_CMEMNET_INPUT_UNIT_H_
#define CMEMNET_CMEMNET_INPUT_UNIT_H_

// assert
#include <cassert>
// std::vector
#include <vector>
// tialnn::IndexType
// tialnn::ErrorType
#include <tialnn/base.h>
// tialnn::SparseInputYBatchLayer
#include <tialnn/layer/sparse_input_ybatchlayer.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::IdentityNeurons
#include <tialnn/neuron/identity_neurons.h>
// tialnn::YConnectionBase
#include <tialnn/connection/yconnection_base.h>
// tialnn::YConnectionInputMajor
#include <tialnn/connection/yconnection_inputmajor.h>
// tialnn::XConnectionInputMajor
#include <tialnn/connection/xconnection_inputmajor.h>
// tialnn::XConnectionOutputShared
#include <tialnn/connection/xconnection_outputshared.h>

namespace cmemnet {

struct CMemNetInputAtom {
  //! Sparse input layer.
  //! (nin, bs)
  tialnn::SparseInputYBatchLayer input_layer;
};

//! Input unit for CMemNet.
struct CMemNetInputUnit {
  //! Constructor.
  CMemNetInputUnit() : capacity(0), size(0), batchsize(0),
    ptr_conn_input_composition(nullptr),
    ptr_conn_bias_composition(nullptr),
    ptr_conn_globalbias_composition(nullptr),
    ptr_bias_layer(nullptr) {}
  //! Destructor.
  ~CMemNetInputUnit() {
    assert(!ptr_conn_input_composition);
    assert(!ptr_conn_bias_composition);
    assert(!ptr_conn_globalbias_composition);
  }

  //! Allocates the unit.
  template <class GradientSolver>
  void AllocateUnit(tialnn::IndexType c,
                    tialnn::IndexType bs,
                    tialnn::IndexType nin,
                    tialnn::IndexType ncomp) {
    assert(capacity == 0);
    assert(batchsize == 0);

    assert(!ptr_conn_input_composition);
    assert(!ptr_conn_bias_composition);
    assert(!ptr_conn_globalbias_composition);

    assert(ptr_bias_layer);

    capacity = c;
    size = 0;
    batchsize = bs;

    ptr_conn_input_composition = new tialnn::YConnectionInputMajor<GradientSolver>();
    ptr_conn_bias_composition = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_globalbias_composition = new tialnn::XConnectionOutputShared<GradientSolver>();

    ptr_conn_input_composition->set_dims(nin, ncomp);
    ptr_conn_bias_composition->set_dims(1, ncomp);
    ptr_conn_globalbias_composition->set_dims(1, ncomp);

    composition_layer.set_nneurons_batchsize(ncomp, bs * c);

    inputatoms.resize(c);
    for (auto &atom : inputatoms) {
      atom.input_layer.set_nneurons_batchsize(nin, bs);
    }
  }

  //! Destroys the unit.
  void DestroyUnit() {
    assert(ptr_conn_input_composition);
    assert(ptr_conn_bias_composition);
    assert(ptr_conn_globalbias_composition);

    delete ptr_conn_input_composition;
    delete ptr_conn_bias_composition;
    delete ptr_conn_globalbias_composition;

    ptr_conn_input_composition = nullptr;
    ptr_conn_bias_composition = nullptr;
    ptr_conn_globalbias_composition = nullptr;
  }

  //! Capacity of the unit.
  tialnn::IndexType capacity;
  //! Current size of the unit.
  //! It should always be no more than capacity.
  tialnn::IndexType size;
  //! Batch size.
  tialnn::IndexType batchsize;

  //! Connection: input -> composition
  tialnn::YConnectionBase *ptr_conn_input_composition;
  //! Neuron-depedent bias for the composition layer.
  tialnn::XConnectionBase *ptr_conn_bias_composition;
  //! Neuron-indepedent bias for the composition layer.
  tialnn::XConnectionBase *ptr_conn_globalbias_composition;

  //! Pointer to the bias layer (shared by the whole network). 
  const tialnn::XBatchLayer<tialnn::IdentityNeurons> *ptr_bias_layer;

  //! Composition layer.
  //! (ncomp, bs * c)
  tialnn::XBatchLayer<tialnn::IdentityNeurons> composition_layer;

  //! Input atoms.
  std::vector<CMemNetInputAtom> inputatoms;
};

} // namespace cmemnet

#endif
