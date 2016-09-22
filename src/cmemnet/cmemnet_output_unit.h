/*!
 * \file cmemnet_output_unit.h
 *
 * 
 * \version 0.0.6
 */

#ifndef CMEMNET_CMEMNET_OUTPUT_UNIT_H_
#define CMEMNET_CMEMNET_OUTPUT_UNIT_H_

// assert
#include <cassert>
// std::vector
#include <vector>
// tialnn::IndexType
// tialnn::ErrorType
#include <tialnn/base.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::IdentityNeurons
#include <tialnn/neuron/identity_neurons.h>
// tialnn::SoftmaxNeurons
#include <tialnn/neuron/softmax_neurons.h>
// tialnn::XConnectionBase
#include <tialnn/connection/xconnection_base.h>
// tialnn::XConnectionInputMajor
#include <tialnn/connection/xconnection_inputmajor.h>
// tialnn::XConnectionOutputShared
#include <tialnn/connection/xconnection_outputshared.h>

namespace cmemnet {

//! Output unit for CMemNet.
struct CMemNetOutputUnit {
  //! Constructor.
  CMemNetOutputUnit() : capacity(0), size(0), batchsize(0),
    ptr_conn_l2ratt_l2rhidden_output(nullptr), ptr_conn_l2ratt_r2lhidden_output(nullptr),
    ptr_conn_r2latt_l2rhidden_output(nullptr), ptr_conn_r2latt_r2lhidden_output(nullptr),
    ptr_conn_decoderl2rmemout_output(nullptr), ptr_conn_decoderr2lmemout_output(nullptr),
    ptr_conn_bias_output(nullptr),
    ptr_conn_globalbias_output(nullptr),
    ptr_bias_layer(nullptr) {}
  //! Destructor.
  ~CMemNetOutputUnit() {
    assert(!ptr_conn_l2ratt_l2rhidden_output);
    assert(!ptr_conn_l2ratt_r2lhidden_output);
    assert(!ptr_conn_r2latt_l2rhidden_output);
    assert(!ptr_conn_r2latt_r2lhidden_output);
    assert(!ptr_conn_decoderl2rmemout_output);
    assert(!ptr_conn_decoderr2lmemout_output);
    assert(!ptr_conn_bias_output);
    assert(!ptr_conn_globalbias_output);
  }

  //! Allocates the unit.
  template <class GradientSolver>
  void AllocateUnit(tialnn::IndexType c,
                    tialnn::IndexType bs,
                    tialnn::IndexType nh,
                    tialnn::IndexType nout) {
    assert(capacity == 0);
    assert(batchsize == 0);

    assert(!ptr_conn_l2ratt_l2rhidden_output);
    assert(!ptr_conn_l2ratt_r2lhidden_output);
    assert(!ptr_conn_r2latt_l2rhidden_output);
    assert(!ptr_conn_r2latt_r2lhidden_output);
    assert(!ptr_conn_decoderl2rmemout_output);
    assert(!ptr_conn_decoderr2lmemout_output);
    assert(!ptr_conn_bias_output);
    assert(!ptr_conn_globalbias_output);

    assert(ptr_bias_layer);

    capacity = c;
    size = 0;
    batchsize = bs;

    ptr_conn_l2ratt_l2rhidden_output = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_l2ratt_r2lhidden_output = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_r2latt_l2rhidden_output = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_r2latt_r2lhidden_output = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_decoderl2rmemout_output = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_decoderr2lmemout_output = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_bias_output = new tialnn::XConnectionInputMajor<GradientSolver>();
    ptr_conn_globalbias_output = new tialnn::XConnectionOutputShared<GradientSolver>();

    ptr_conn_l2ratt_l2rhidden_output->set_dims(nh, nout);
    ptr_conn_l2ratt_r2lhidden_output->set_dims(nh, nout);
    ptr_conn_r2latt_l2rhidden_output->set_dims(nh, nout);
    ptr_conn_r2latt_r2lhidden_output->set_dims(nh, nout);
    ptr_conn_decoderl2rmemout_output->set_dims(nh, nout);
    ptr_conn_decoderr2lmemout_output->set_dims(nh, nout);
    ptr_conn_bias_output->set_dims(1, nout);
    ptr_conn_globalbias_output->set_dims(1, nout);

    output_layer.set_nneurons_batchsize(nout, bs * c);
    output_errors.resize(nout * bs * c);
  }

  //! Destroys the unit.
  void DestroyUnit() {
    assert(ptr_conn_l2ratt_l2rhidden_output);
    assert(ptr_conn_l2ratt_r2lhidden_output);
    assert(ptr_conn_r2latt_l2rhidden_output);
    assert(ptr_conn_r2latt_r2lhidden_output);
    assert(ptr_conn_decoderl2rmemout_output);
    assert(ptr_conn_decoderr2lmemout_output);
    assert(ptr_conn_bias_output);
    assert(ptr_conn_globalbias_output);

    delete ptr_conn_l2ratt_l2rhidden_output;
    delete ptr_conn_l2ratt_r2lhidden_output;
    delete ptr_conn_r2latt_l2rhidden_output;
    delete ptr_conn_r2latt_r2lhidden_output;
    delete ptr_conn_decoderl2rmemout_output;
    delete ptr_conn_decoderr2lmemout_output;
    delete ptr_conn_bias_output;
    delete ptr_conn_globalbias_output;

    ptr_conn_l2ratt_l2rhidden_output = nullptr;
    ptr_conn_l2ratt_r2lhidden_output = nullptr;
    ptr_conn_r2latt_l2rhidden_output = nullptr;
    ptr_conn_r2latt_r2lhidden_output = nullptr;
    ptr_conn_decoderl2rmemout_output = nullptr;
    ptr_conn_decoderr2lmemout_output = nullptr;
    ptr_conn_bias_output = nullptr;
    ptr_conn_globalbias_output = nullptr;
  }

  //! Capacity of the unit.
  tialnn::IndexType capacity;
  //! Current size of the unit.
  //! It should always be smaller than capacity.
  tialnn::IndexType size;
  //! Batch size.
  tialnn::IndexType batchsize;

  //! Connection: attention hidden -> output.
  tialnn::XConnectionBase *ptr_conn_l2ratt_l2rhidden_output;
  tialnn::XConnectionBase *ptr_conn_l2ratt_r2lhidden_output;
  tialnn::XConnectionBase *ptr_conn_r2latt_l2rhidden_output;
  tialnn::XConnectionBase *ptr_conn_r2latt_r2lhidden_output;
  //! Connection: decoder memout -> output.
  tialnn::XConnectionBase *ptr_conn_decoderl2rmemout_output;
  tialnn::XConnectionBase *ptr_conn_decoderr2lmemout_output;
  //! Neuron-depedent bias for the output layer.
  tialnn::XConnectionBase *ptr_conn_bias_output;
  //! Neuron-indepedent bias for the output layer.
  tialnn::XConnectionBase *ptr_conn_globalbias_output;

  //! Pointer to the bias layer (shared by the whole network). 
  const tialnn::XBatchLayer<tialnn::IdentityNeurons> *ptr_bias_layer;

  //! Output layer.
  //! (nout, bs * c)
  tialnn::XBatchLayer<tialnn::SoftmaxNeurons> output_layer;
  //! Errors for output layer.
  //! (nout, bs * c)
  std::vector<tialnn::ErrorType> output_errors;
};

} // namespace cmemnet

#endif
