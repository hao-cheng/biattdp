/*!
 * \file cmemnet_dparser.cpp
 *
 * 
 * \version 0.0.6
 */

#ifdef __GNUC__
//! Disable this warning for BOOST.
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "cmemnet_dparser.h"

// assert
#include <cassert>
// std::log
#include <cmath>
// std::cout
// std::cerr
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// std::vector
#include <vector>
// std::distance
#include <iterator>
// std::copy
// std::fill
// std::transform
// std::max_element
#include <algorithm>
// boost::adjacenty_list
// boost::listS
// boost::vecS
// boost::directedS
// boost::add_edge
#include <boost/graph/adjacency_list.hpp>
// BOOST_FOREACH
#include <boost/foreach.hpp>
// edmonds_optimum_branching
#include <edmonds/edmonds_optimum_branching.hpp>
// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
// tialnn::WeightType
#include <tialnn/base.h>
// TIALNN_CBLAS_XXX
#include <tialnn/util/numeric.h>
// tialnn::write_xxx
// tialnn::read_xxx
#include <tialnn/util/futil.h>
// tialnn::VanillaGrad
#include <tialnn/gradient/vanilla_grad.h>
// tialnn::AdaGrad
#include <tialnn/gradient/ada_grad.h>
// tialnn::AdaDelta
#include <tialnn/gradient/ada_delta.h>
// tialnn::Adam
#include <tialnn/gradient/adam.h>
// tialnn::MikolovInitializer
#include <tialnn/initializer/mikolov_initializer.h>
// tialnn::UniformInitializer
#include <tialnn/initializer/uniform_initializer.h>
// tialnn::GaussianInitializer
#include <tialnn/initializer/gaussian_initializer.h>
// memnet::DataPointToken
#include "../memnet/datapoint_token.h"
// memnet::DataPointBatch
#include "../memnet/datapoint_batch.h"
// cmemnet::CMemNetForwardPropagate
// cmemnet:CMemNetBackwardPropagate
#include "cmemnet_utils.h"

using std::log;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::distance;
using std::copy;
using std::fill;
using std::transform;
using std::max_element;
using tialnn::IndexType;
using tialnn::ActivationType;
using tialnn::ErrorType;
using tialnn::WeightType;
using tialnn::read_single;
using tialnn::write_single;
using tialnn::VanillaGrad;
using tialnn::AdaGrad;
using tialnn::AdaDelta;
using tialnn::Adam;
using tialnn::MikolovInitializer;
using tialnn::UniformInitializer;
using tialnn::GaussianInitializer;
using memnet::DataPointToken;
using memnet::DataPointBatch;

namespace cmemnet {

//! NOTE:
//! 1. Here, we also sets the layers with fixed activations.
//!    - bias_batchlayer_
//!    - encoder_unit_.init_batchlayer
//!    - decoder_unit_.init_batchlayer
void CMemNetDParser::AllocateModel() {
  assert(modelopts().max_sequence_length > 1);
  assert(modelopts().batch_size > 0);
  assert(num_composition_neurons_ > 0);
  assert(num_projection_neurons_ > 0);
  assert(num_hidden_neurons_ > 0);
  assert(word_vocab().size() > 0);
  assert(dependency_vocab().size() > 0);

  //! Allocate layers.
  negones_layer_.set_nneurons_batchsize(num_hidden_neurons_, modelopts().batch_size);
  negones_layer_.SetActivationsToValue(-1.0f);

  bias_layer_.set_nneurons_batchsize(1, modelopts().batch_size * modelopts().max_sequence_length);
  bias_layer_.SetActivationsToValue(1.0f);

  init_hidden_layer_.set_nneurons_batchsize(num_hidden_neurons_, modelopts().batch_size);
  init_hidden_layer_.SetActivationsToValue(0.1f);

  //===============================
  IndexType ninput = word_vocab().size();
  ninput += lemma_vocab().size();
  ninput += cpos_vocab().size();
  ninput += pos_vocab().size();
  ninput += feature_vocab().size();

  input_unit_.ptr_bias_layer = &bias_layer_;

  if (modelopts().solver == modelopts().vanilla) {
    input_unit_.AllocateUnit<VanillaGrad>(modelopts().max_sequence_length,
                                          modelopts().batch_size,
                                          ninput,
                                          num_composition_neurons_);
  } else if (modelopts().solver == modelopts().adagrad) {
    input_unit_.AllocateUnit<AdaGrad>(modelopts().max_sequence_length,
                                      modelopts().batch_size,
                                      ninput,
                                      num_composition_neurons_);
  } else if (modelopts().solver == modelopts().adadelta) {
    input_unit_.AllocateUnit<AdaDelta>(modelopts().max_sequence_length,
                                       modelopts().batch_size,
                                       ninput,
                                       num_composition_neurons_);
  } else if (modelopts().solver == modelopts().adam) {
    input_unit_.AllocateUnit<Adam>(modelopts().max_sequence_length,
                                   modelopts().batch_size,
                                   ninput,
                                   num_composition_neurons_);
  }

  RegisterConnection(input_unit_.ptr_conn_input_composition);
  RegisterConnection(input_unit_.ptr_conn_bias_composition);
  RegisterConnection(input_unit_.ptr_conn_globalbias_composition);
  //===============================

  //===============================
  projection_unit_.ptr_bias_layer = &bias_layer_;

  if (modelopts().solver == modelopts().vanilla) {
    projection_unit_.AllocateUnit<VanillaGrad>(modelopts().max_sequence_length,
                                               modelopts().batch_size,
                                               num_composition_neurons_,
                                               num_projection_neurons_);
  } else if (modelopts().solver == modelopts().adagrad) {
    projection_unit_.AllocateUnit<AdaGrad>(modelopts().max_sequence_length,
                                           modelopts().batch_size,
                                           num_composition_neurons_,
                                           num_projection_neurons_);
  } else if (modelopts().solver == modelopts().adadelta) {
    projection_unit_.AllocateUnit<AdaDelta>(modelopts().max_sequence_length,
                                            modelopts().batch_size,
                                            num_composition_neurons_,
                                            num_projection_neurons_);
  } else if (modelopts().solver == modelopts().adam) {
    projection_unit_.AllocateUnit<Adam>(modelopts().max_sequence_length,
                                        modelopts().batch_size,
                                        num_composition_neurons_,
                                        num_projection_neurons_);
  }

  RegisterConnection(projection_unit_.ptr_conn_composition_projection);
  RegisterConnection(projection_unit_.ptr_conn_bias_projection);
  RegisterConnection(projection_unit_.ptr_conn_globalbias_projection);
  //===============================

  //===============================
  encoder_unit_.l2r.ptr_init_hidden_layer = &init_hidden_layer_;
  encoder_unit_.l2r.ptr_bias_layer = &bias_layer_;
  encoder_unit_.l2r.ptr_negones_layer = &negones_layer_;

  encoder_unit_.r2l.ptr_init_hidden_layer = &init_hidden_layer_;
  encoder_unit_.r2l.ptr_bias_layer = &bias_layer_;
  encoder_unit_.r2l.ptr_negones_layer = &negones_layer_;

  if (modelopts().solver == modelopts().vanilla) {
    encoder_unit_.AllocateUnit<VanillaGrad>(modelopts().max_sequence_length,
                                            modelopts().batch_size,
                                            num_projection_neurons_,
                                            num_hidden_neurons_);
  } else if (modelopts().solver == modelopts().adagrad) {
    encoder_unit_.AllocateUnit<AdaGrad>(modelopts().max_sequence_length,
                                        modelopts().batch_size,
                                        num_projection_neurons_,
                                        num_hidden_neurons_);
  } else if (modelopts().solver == modelopts().adadelta) {
    encoder_unit_.AllocateUnit<AdaDelta>(modelopts().max_sequence_length,
                                         modelopts().batch_size,
                                         num_projection_neurons_,
                                         num_hidden_neurons_);
  } else if (modelopts().solver == modelopts().adam) {
    encoder_unit_.AllocateUnit<Adam>(modelopts().max_sequence_length,
                                     modelopts().batch_size,
                                     num_projection_neurons_,
                                     num_hidden_neurons_);
  }

  RegisterConnection(encoder_unit_.l2r.ptr_conn_projection_memoutreset);
  RegisterConnection(encoder_unit_.l2r.ptr_conn_projection_memoutupdate);
  RegisterConnection(encoder_unit_.l2r.ptr_conn_projection_memouthtilde);
  RegisterConnection(encoder_unit_.l2r.ptr_conn_memout_memin);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_hprev_reset);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_hprev_update);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_hprev_htilde);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_bias_reset);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_bias_update);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_bias_htilde);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_globalbias_reset);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_globalbias_update);
  RegisterConnection(encoder_unit_.l2r.ptr_memout_conn_globalbias_htilde);

  RegisterConnection(encoder_unit_.r2l.ptr_conn_projection_memoutreset);
  RegisterConnection(encoder_unit_.r2l.ptr_conn_projection_memoutupdate);
  RegisterConnection(encoder_unit_.r2l.ptr_conn_projection_memouthtilde);
  RegisterConnection(encoder_unit_.r2l.ptr_conn_memout_memin);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_hprev_reset);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_hprev_update);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_hprev_htilde);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_bias_reset);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_bias_update);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_bias_htilde);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_globalbias_reset);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_globalbias_update);
  RegisterConnection(encoder_unit_.r2l.ptr_memout_conn_globalbias_htilde);
  //===============================

  //===============================
  decoder_unit_.l2r.ptr_init_hidden_layer = &init_hidden_layer_;
  decoder_unit_.l2r.ptr_bias_layer = &bias_layer_;
  decoder_unit_.l2r.ptr_negones_layer = &negones_layer_;

  decoder_unit_.r2l.ptr_init_hidden_layer = &init_hidden_layer_;
  decoder_unit_.r2l.ptr_bias_layer = &bias_layer_;
  decoder_unit_.r2l.ptr_negones_layer = &negones_layer_;

  if (modelopts().solver == modelopts().vanilla) {
    decoder_unit_.AllocateUnit<VanillaGrad>(modelopts().max_sequence_length,
                                            modelopts().max_sequence_length,
                                            modelopts().batch_size,
                                            num_projection_neurons_,
                                            num_hidden_neurons_);
  } else if (modelopts().solver == modelopts().adagrad) {
    decoder_unit_.AllocateUnit<AdaGrad>(modelopts().max_sequence_length,
                                        modelopts().max_sequence_length,
                                        modelopts().batch_size,
                                        num_projection_neurons_,
                                        num_hidden_neurons_);
  } else if (modelopts().solver == modelopts().adadelta) {
    decoder_unit_.AllocateUnit<AdaDelta>(modelopts().max_sequence_length,
                                         modelopts().max_sequence_length,
                                         modelopts().batch_size,
                                         num_projection_neurons_,
                                         num_hidden_neurons_);
  } else if (modelopts().solver == modelopts().adam) {
    decoder_unit_.AllocateUnit<Adam>(modelopts().max_sequence_length,
                                     modelopts().max_sequence_length,
                                     modelopts().batch_size,
                                     num_projection_neurons_,
                                     num_hidden_neurons_);
  }

  RegisterConnection(decoder_unit_.l2r.ptr_conn_projection_memoutreset);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_projection_memoutupdate);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_projection_memouthtilde);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_attl2rhidden_memoutreset);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_attl2rhidden_memoutupdate);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_attl2rhidden_memouthtilde);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_attr2lhidden_memoutreset);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_attr2lhidden_memoutupdate);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_attr2lhidden_memouthtilde);
  RegisterConnection(decoder_unit_.l2r.ptr_conn_memout_memin);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_hprev_reset);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_hprev_update);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_hprev_htilde);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_bias_reset);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_bias_update);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_bias_htilde);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_globalbias_reset);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_globalbias_update);
  RegisterConnection(decoder_unit_.l2r.ptr_memout_conn_globalbias_htilde);
  RegisterConnection(decoder_unit_.l2r.ptr_attention_conn_hidden0_hidden1);
  RegisterConnection(decoder_unit_.l2r.ptr_attention_conn_bias_hidden1);

  RegisterConnection(decoder_unit_.r2l.ptr_conn_projection_memoutreset);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_projection_memoutupdate);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_projection_memouthtilde);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_attl2rhidden_memoutreset);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_attl2rhidden_memoutupdate);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_attl2rhidden_memouthtilde);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_attr2lhidden_memoutreset);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_attr2lhidden_memoutupdate);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_attr2lhidden_memouthtilde);
  RegisterConnection(decoder_unit_.r2l.ptr_conn_memout_memin);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_hprev_reset);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_hprev_update);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_hprev_htilde);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_bias_reset);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_bias_update);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_bias_htilde);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_globalbias_reset);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_globalbias_update);
  RegisterConnection(decoder_unit_.r2l.ptr_memout_conn_globalbias_htilde);
  RegisterConnection(decoder_unit_.r2l.ptr_attention_conn_hidden0_hidden1);
  RegisterConnection(decoder_unit_.r2l.ptr_attention_conn_bias_hidden1);
  //===============================

  //===============================
  output_unit_.ptr_bias_layer = &bias_layer_;

  if (modelopts().solver == modelopts().vanilla) {
    output_unit_.AllocateUnit<VanillaGrad>(modelopts().max_sequence_length - 1, //<! \note Does not include <root>.
                                           modelopts().batch_size,
                                           num_hidden_neurons_,
                                           dependency_vocab().size());
  } else if (modelopts().solver == modelopts().adagrad) {
    output_unit_.AllocateUnit<AdaGrad>(modelopts().max_sequence_length - 1, //<! \note Does not include <root>.
                                       modelopts().batch_size,
                                       num_hidden_neurons_,
                                       dependency_vocab().size());
  } else if (modelopts().solver == modelopts().adadelta) {
    output_unit_.AllocateUnit<AdaDelta>(modelopts().max_sequence_length - 1, //<! \note Does not include <root>.
                                        modelopts().batch_size,
                                        num_hidden_neurons_,
                                        dependency_vocab().size());
  } else if (modelopts().solver == modelopts().adam) {
    output_unit_.AllocateUnit<Adam>(modelopts().max_sequence_length - 1, //<! \note Does not include <root>.
                                    modelopts().batch_size,
                                    num_hidden_neurons_,
                                    dependency_vocab().size());
  }

  RegisterConnection(output_unit_.ptr_conn_l2ratt_l2rhidden_output);
  RegisterConnection(output_unit_.ptr_conn_l2ratt_r2lhidden_output);
  RegisterConnection(output_unit_.ptr_conn_r2latt_l2rhidden_output);
  RegisterConnection(output_unit_.ptr_conn_r2latt_r2lhidden_output);
  RegisterConnection(output_unit_.ptr_conn_decoderl2rmemout_output);
  RegisterConnection(output_unit_.ptr_conn_decoderr2lmemout_output);
  RegisterConnection(output_unit_.ptr_conn_bias_output);
  RegisterConnection(output_unit_.ptr_conn_globalbias_output);
  //===============================
}

void CMemNetDParser::DestroyModel() {
  input_unit_.DestroyUnit();
  projection_unit_.DestroyUnit();
  encoder_unit_.DestroyUnit();
  decoder_unit_.DestroyUnit();
  output_unit_.DestroyUnit();
}

void CMemNetDParser::InitializeNeuralNet() {
  //! Reset activations.
  ResetActivations();

  if (modelopts().initializer == modelopts().mikolov) {
    MikolovInitializer initializer;
    InitializeWeights(initializer, rng_engine_);
  } else if (modelopts().initializer == modelopts().gaussian) {
    GaussianInitializer initializer(modelopts().initializer_std);
    InitializeWeights(initializer, rng_engine_);
  } else if (modelopts().initializer == modelopts().xarvier) {
    //! \note We use scaled uniform initialization.
    //! \see http://deeplearning.net/tutorial/mlp.html#weight-initialization.
    UniformInitializer initializer;
    InitializeWeights(initializer, rng_engine_);

    //! \note Needs to manually initialize connections with sigmoid output layer.
    //===============================
    initializer.XavierSetScale(encoder_unit_.l2r.ptr_conn_projection_memoutreset, true);
    encoder_unit_.l2r.ptr_conn_projection_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(encoder_unit_.l2r.ptr_conn_projection_memoutupdate, true);
    encoder_unit_.l2r.ptr_conn_projection_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(encoder_unit_.l2r.ptr_memout_conn_hprev_reset, true);
    encoder_unit_.l2r.ptr_memout_conn_hprev_reset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(encoder_unit_.l2r.ptr_memout_conn_hprev_update, true);
    encoder_unit_.l2r.ptr_memout_conn_hprev_update->RandomlyInitialize(initializer, rng_engine_);


    initializer.XavierSetScale(encoder_unit_.r2l.ptr_conn_projection_memoutreset, true);
    encoder_unit_.r2l.ptr_conn_projection_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(encoder_unit_.r2l.ptr_conn_projection_memoutupdate, true);
    encoder_unit_.r2l.ptr_conn_projection_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(encoder_unit_.r2l.ptr_memout_conn_hprev_reset, true);
    encoder_unit_.r2l.ptr_memout_conn_hprev_reset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(encoder_unit_.r2l.ptr_memout_conn_hprev_update, true);
    encoder_unit_.r2l.ptr_memout_conn_hprev_update->RandomlyInitialize(initializer, rng_engine_);
    //===============================

    //===============================
    initializer.XavierSetScale(decoder_unit_.l2r.ptr_conn_projection_memoutreset, true);
    decoder_unit_.l2r.ptr_conn_projection_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.l2r.ptr_conn_projection_memoutupdate, true);
    decoder_unit_.l2r.ptr_conn_projection_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.l2r.ptr_conn_attl2rhidden_memoutreset, true);
    decoder_unit_.l2r.ptr_conn_attl2rhidden_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.l2r.ptr_conn_attl2rhidden_memoutupdate, true);
    decoder_unit_.l2r.ptr_conn_attl2rhidden_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.l2r.ptr_conn_attr2lhidden_memoutreset, true);
    decoder_unit_.l2r.ptr_conn_attr2lhidden_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.l2r.ptr_conn_attr2lhidden_memoutupdate, true);
    decoder_unit_.l2r.ptr_conn_attr2lhidden_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.l2r.ptr_memout_conn_hprev_reset, true);
    decoder_unit_.l2r.ptr_memout_conn_hprev_reset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.l2r.ptr_memout_conn_hprev_update, true);
    decoder_unit_.l2r.ptr_memout_conn_hprev_update->RandomlyInitialize(initializer, rng_engine_);


    initializer.XavierSetScale(decoder_unit_.r2l.ptr_conn_projection_memoutreset, true);
    decoder_unit_.r2l.ptr_conn_projection_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.r2l.ptr_conn_projection_memoutupdate, true);
    decoder_unit_.r2l.ptr_conn_projection_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.r2l.ptr_conn_attl2rhidden_memoutreset, true);
    decoder_unit_.r2l.ptr_conn_attl2rhidden_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.r2l.ptr_conn_attl2rhidden_memoutupdate, true);
    decoder_unit_.r2l.ptr_conn_attl2rhidden_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.r2l.ptr_conn_attr2lhidden_memoutreset, true);
    decoder_unit_.r2l.ptr_conn_attr2lhidden_memoutreset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.r2l.ptr_conn_attr2lhidden_memoutupdate, true);
    decoder_unit_.r2l.ptr_conn_attr2lhidden_memoutupdate->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.r2l.ptr_memout_conn_hprev_reset, true);
    decoder_unit_.r2l.ptr_memout_conn_hprev_reset->RandomlyInitialize(initializer, rng_engine_);

    initializer.XavierSetScale(decoder_unit_.r2l.ptr_memout_conn_hprev_update, true);
    decoder_unit_.r2l.ptr_memout_conn_hprev_update->RandomlyInitialize(initializer, rng_engine_);
    //===============================
  }

  //! \note The conn_bias_xxx and conn_globalbias_xxx are initialized as 0s.
  input_unit_.ptr_conn_bias_composition->ResetConnection();
  input_unit_.ptr_conn_globalbias_composition->ResetConnection();

  projection_unit_.ptr_conn_bias_projection->ResetConnection();
  projection_unit_.ptr_conn_globalbias_projection->ResetConnection();

  encoder_unit_.l2r.ptr_memout_conn_bias_reset->ResetConnection();
  encoder_unit_.l2r.ptr_memout_conn_bias_update->ResetConnection();
  encoder_unit_.l2r.ptr_memout_conn_bias_htilde->ResetConnection();
  encoder_unit_.l2r.ptr_memout_conn_globalbias_reset->ResetConnection();
  encoder_unit_.l2r.ptr_memout_conn_globalbias_update->ResetConnection();
  encoder_unit_.l2r.ptr_memout_conn_globalbias_htilde->ResetConnection();

  encoder_unit_.r2l.ptr_memout_conn_bias_reset->ResetConnection();
  encoder_unit_.r2l.ptr_memout_conn_bias_update->ResetConnection();
  encoder_unit_.r2l.ptr_memout_conn_bias_htilde->ResetConnection();
  encoder_unit_.r2l.ptr_memout_conn_globalbias_reset->ResetConnection();
  encoder_unit_.r2l.ptr_memout_conn_globalbias_update->ResetConnection();
  encoder_unit_.r2l.ptr_memout_conn_globalbias_htilde->ResetConnection();

  decoder_unit_.l2r.ptr_memout_conn_bias_reset->ResetConnection();
  decoder_unit_.l2r.ptr_memout_conn_bias_update->ResetConnection();
  decoder_unit_.l2r.ptr_memout_conn_bias_htilde->ResetConnection();
  decoder_unit_.l2r.ptr_memout_conn_globalbias_reset->ResetConnection();
  decoder_unit_.l2r.ptr_memout_conn_globalbias_update->ResetConnection();
  decoder_unit_.l2r.ptr_memout_conn_globalbias_htilde->ResetConnection();
  decoder_unit_.l2r.ptr_attention_conn_bias_hidden1->ResetConnection();

  decoder_unit_.r2l.ptr_memout_conn_bias_reset->ResetConnection();
  decoder_unit_.r2l.ptr_memout_conn_bias_update->ResetConnection();
  decoder_unit_.r2l.ptr_memout_conn_bias_htilde->ResetConnection();
  decoder_unit_.r2l.ptr_memout_conn_globalbias_reset->ResetConnection();
  decoder_unit_.r2l.ptr_memout_conn_globalbias_update->ResetConnection();
  decoder_unit_.r2l.ptr_memout_conn_globalbias_htilde->ResetConnection();
  decoder_unit_.r2l.ptr_attention_conn_bias_hidden1->ResetConnection();

  output_unit_.ptr_conn_bias_output->ResetConnection();
  output_unit_.ptr_conn_globalbias_output->ResetConnection();
}

void CMemNetDParser::SetWordEmbeddings(IndexType widx, const vector<WeightType> &embd) {
  if (static_cast<IndexType>(embd.size()) != num_composition_neurons_) {
    cerr << "Embedding dimension mismatch: " << word_vocab().type4idx(widx) << endl;
    exit(EXIT_FAILURE);
  }

  IndexType d = 0;
  for (auto val : embd) {
    input_unit_.ptr_conn_input_composition->set_weights(widx, d, val);
    d++;
  }
}

void CMemNetDParser::ResetActivations() {
  //! \note In current design, we do not need to reset activations manually.
}

void CMemNetDParser::SetActivationsForInputLayers(const memnet::DataPointBatch *ptr_databatch, 
                                                  const vector<IndexType> &idx_datapoints) {
  assert(static_cast<IndexType>(idx_datapoints.size()) <= modelopts().batch_size);

  input_unit_.size = ptr_databatch->sequence_length();
  assert(input_unit_.size <= input_unit_.capacity);

  IndexType t = 0;
  IndexType offset_lemma = word_vocab().size();
  IndexType offset_cpos = offset_lemma + lemma_vocab().size();
  IndexType offset_pos = offset_cpos + cpos_vocab().size();
  IndexType offset_feat = offset_pos + pos_vocab().size();
  for (auto &inputatom : input_unit_.inputatoms) {
    if (t == ptr_databatch->sequence_length()) {
      break;
    }
    inputatom.input_layer.SetActivationsToValue(0.0f);
    IndexType k = 0; //<! Index in batch.
    for (IndexType idx : idx_datapoints) {
      const memnet::DataPointToken &tok = ptr_databatch->data(idx, t);
      //! \note If this is slow, directly sets projection layer.
      inputatom.input_layer.set_activations(tok.form, k, 1.0f);
      if (!lemma_vocab().empty()) {
        inputatom.input_layer.set_activations(offset_lemma + tok.lemma, k, 1.0f);
      }
      if (!cpos_vocab().empty()) {
        inputatom.input_layer.set_activations(offset_cpos + tok.cpos_tag, k, 1.0f);
      }
      if (!pos_vocab().empty()) {
        inputatom.input_layer.set_activations(offset_pos + tok.pos_tag, k, 1.0f);
      }
      if (!feature_vocab().empty()) {
        for (auto &feat : tok.feats) {
          //! \note For feature, we accumulate activations rather than set activations.
          inputatom.input_layer.accumulate_activations(offset_feat + feat, k, 1.0f);
        }
      }

      k++;
    }
    t++;
  }
}

double CMemNetDParser::SetErrorsForOutputLayers(const memnet::DataPointBatch *ptr_databatch,
                                                const vector<IndexType> &idx_datapoints,
                                                float multitask_weight_arclabel) {
  assert(decoder_unit_.size == ptr_databatch->sequence_length());
  assert(output_unit_.size == ptr_databatch->sequence_length() - 1);

  double logp = 0.0f;
  const IndexType bs = modelopts().batch_size;
  const ActivationType one_over_bs = 1.0f / bs;
  const ActivationType gamma_over_bs = multitask_weight_arclabel / bs;
  IndexType dependency_vocab_size = dependency_vocab().size();

#ifdef _TIALNN_DEBUG
  assert(output_unit_.output_layer.CheckAcStates(0, bs * output_unit_.size, true));
#endif

  IndexType bout_offset = 0;
  auto it_output_activations = output_unit_.output_layer.activations().begin();
  auto it_next_output_activations = it_output_activations;
  auto it_output_errors = output_unit_.output_errors.begin();
  auto it_next_output_errors = it_output_errors;

  auto it_l2r_decoderatom = decoder_unit_.l2r.decoderatoms.begin();
  auto it_r2l_decoderatom = decoder_unit_.r2l.decoderatoms.begin();
  ++it_l2r_decoderatom; //<! \note: skip <root>.
  ++it_r2l_decoderatom; //<! \note: skip <root>.
  for (IndexType t = 1; t < ptr_databatch->sequence_length(); t++) {
#ifdef _TIALNN_DEBUG
    assert(it_l2r_decoderatom->attention_unit.hidden1_layer.CheckAcStates(0, bs * it_l2r_decoderatom->attention_unit.size, true));
    assert(it_r2l_decoderatom->attention_unit.hidden1_layer.CheckAcStates(0, bs * it_r2l_decoderatom->attention_unit.size, true));
#endif

    assert(it_l2r_decoderatom->attention_unit.batchsize == bs);
    assert(it_r2l_decoderatom->attention_unit.batchsize == bs);
    const ActivationType *ptr_l2r_hidden1_ac = it_l2r_decoderatom->attention_unit.hidden1_layer.activations().data();
    const ActivationType *ptr_r2l_hidden1_ac = it_r2l_decoderatom->attention_unit.hidden1_layer.activations().data();
    ErrorType *ptr_l2r_hidden1_label_er = it_l2r_decoderatom->attention_unit.hidden1_label_errors.data();
    ErrorType *ptr_r2l_hidden1_label_er = it_r2l_decoderatom->attention_unit.hidden1_label_errors.data();

    IndexType k = 0; //<! Index in batch.
    for (IndexType idx : idx_datapoints) {
      it_output_activations = it_next_output_activations;
      it_next_output_activations += dependency_vocab_size;
      it_output_errors = it_next_output_errors;
      it_next_output_errors += dependency_vocab_size;

      const memnet::DataPointToken &tok = ptr_databatch->data(idx, t);
      if (tok.form == word_vocab().blank_idx()) {
        //! \note Sets errors to 0 for <blank>.
        fill(it_output_errors, it_next_output_errors, 0.0f);
        //! \note We need to use cblas_dscal here because of the stepsize bs.
        TIALNN_CBLAS_SCAL(it_l2r_decoderatom->attention_unit.size, 0.0f, ptr_l2r_hidden1_label_er, bs);
        TIALNN_CBLAS_SCAL(it_r2l_decoderatom->attention_unit.size, 0.0f, ptr_r2l_hidden1_label_er, bs);
      } else {
        TIALNN_CBLAS_AXPBY(dependency_vocab_size,
                           -gamma_over_bs, &(*it_output_activations), 1,
                           0.0f, &(*it_output_errors), 1);
        *(it_output_errors + tok.dependency) += gamma_over_bs;
        logp += multitask_weight_arclabel * log(output_unit_.output_layer.activations(tok.dependency, k + bout_offset));

        //! \note We need to use cblas_daxpby here because of the stepsize bs.
        TIALNN_CBLAS_AXPBY(it_l2r_decoderatom->attention_unit.size, -one_over_bs,
                           ptr_l2r_hidden1_ac, bs,
                           0.0f, ptr_l2r_hidden1_label_er, bs);
        *(ptr_l2r_hidden1_label_er + bs * tok.headword_position) += one_over_bs;
        logp += log(it_l2r_decoderatom->attention_unit.hidden1_layer.activations(k, tok.headword_position));

        TIALNN_CBLAS_AXPBY(it_r2l_decoderatom->attention_unit.size, -one_over_bs,
                           ptr_r2l_hidden1_ac, bs,
                           0.0f, ptr_r2l_hidden1_label_er, bs);
        *(ptr_r2l_hidden1_label_er + bs * tok.headword_position) += one_over_bs;
        logp += log(it_r2l_decoderatom->attention_unit.hidden1_layer.activations(k, tok.headword_position));
      }

      ++ptr_l2r_hidden1_ac;
      ++ptr_l2r_hidden1_label_er;
      ++ptr_r2l_hidden1_ac;
      ++ptr_r2l_hidden1_label_er;

      k++;
    }

    //! \note Also needs to set errors to 0 for rest elements in this batch.
    //        Make sure [0, bs_x_outsz) and [0, bs_x_encsz) are all set.
    assert(output_unit_.batchsize == bs);
    while (k < bs) {
      //! \note Still needs to update the iterator.
      it_output_activations = it_next_output_activations;
      it_next_output_activations += dependency_vocab_size;

      it_output_errors = it_next_output_errors;
      it_next_output_errors += dependency_vocab_size;
      fill(it_output_errors, it_next_output_errors, 0.0f);

      //! \note We need to use cblas_dscal here because of the stepsize bs.
      TIALNN_CBLAS_SCAL(it_l2r_decoderatom->attention_unit.size, 0.0f, ptr_l2r_hidden1_label_er, bs);
      TIALNN_CBLAS_SCAL(it_r2l_decoderatom->attention_unit.size, 0.0f, ptr_r2l_hidden1_label_er, bs);

      //! \note No need to ++ptr_hidden1_ac because we assign it at the beginning of the loop.
      ++ptr_l2r_hidden1_label_er;
      ++ptr_r2l_hidden1_label_er;
      k++;
    }

    bout_offset += bs;
    ++it_l2r_decoderatom;
    ++it_r2l_decoderatom;
  }

  //! \note We only sets the errors[0:nh, 0:bs, 0:sz].
  //!       errors[:, :, sz:] are invalid.

  return logp;
}

//! Almost identitcal to SetErrorsForOutputLayers.
double CMemNetDParser::GetLogProb(const memnet::DataPointBatch *ptr_databatch,
                                  const vector<IndexType> &idx_datapoints,
                                  float multitask_weight_arclabel) {
  assert(decoder_unit_.size == ptr_databatch->sequence_length());
  assert(output_unit_.size == ptr_databatch->sequence_length() - 1);

  double logp = 0.0f;
  const IndexType bs = modelopts().batch_size;

  IndexType bout_offset = 0;
  auto it_l2r_decoderatom = decoder_unit_.l2r.decoderatoms.begin();
  auto it_r2l_decoderatom = decoder_unit_.r2l.decoderatoms.begin();
  ++it_l2r_decoderatom; //! \note: skip <root>.
  ++it_r2l_decoderatom; //! \note: skip <root>.
  for (IndexType t = 1; t < ptr_databatch->sequence_length(); t++) {
    IndexType k = 0; //<! Index in batch.
    for (IndexType idx : idx_datapoints) {
      const memnet::DataPointToken &tok = ptr_databatch->data(idx, t);
      //! \note Skips the blank tokens.
      if (tok.form != word_vocab().blank_idx()) {
        logp += multitask_weight_arclabel * log(output_unit_.output_layer.activations(tok.dependency, k + bout_offset));

        logp += log(it_l2r_decoderatom->attention_unit.hidden1_layer.activations(k, tok.headword_position));
        logp += log(it_r2l_decoderatom->attention_unit.hidden1_layer.activations(k, tok.headword_position));
      }

      k++;
    }

    bout_offset += bs;
    ++it_l2r_decoderatom;
    ++it_r2l_decoderatom;
  }

  return logp;
}

void CMemNetDParser::ForwardPropagate() {
  CMemNetForwardPropagate(input_unit_);
  CMemNetForwardPropagate(input_unit_,
                          projection_unit_);
  CMemNetForwardPropagate(projection_unit_,
                          encoder_unit_);
  CMemNetForwardPropagate(projection_unit_,
                          encoder_unit_,
                          decoder_unit_);
  CMemNetForwardPropagate(decoder_unit_,
                          output_unit_);
}

void CMemNetDParser::BackwardPropagate() {
  CMemNetBackwardPropagate(decoder_unit_,
                           output_unit_);
  CMemNetBackwardPropagate(projection_unit_,
                           encoder_unit_,
                           decoder_unit_);
  CMemNetBackwardPropagate(projection_unit_,
                           encoder_unit_);
  CMemNetBackwardPropagate(input_unit_,
                           projection_unit_);
  CMemNetBackwardPropagate(input_unit_);
}

void CMemNetDParser::Predict(const DataPointBatch *ptr_databatch,
                             const vector<IndexType> &idx_datapoints,
                             vector<vector<DataPointToken>> &datapoints) {
  assert(decoder_unit_.size == ptr_databatch->sequence_length());
  assert(output_unit_.size == ptr_databatch->sequence_length() - 1);
  assert(!idx_datapoints.empty());

  IndexType last_num_datapoints = static_cast<IndexType>(datapoints.size());
  datapoints.resize(datapoints.size() + idx_datapoints.size());
  auto it_datapoints_begin = datapoints.begin() + last_num_datapoints;

  const IndexType bs = modelopts().batch_size;
  IndexType dependency_vocab_size = dependency_vocab().size();
  assert(output_unit_.output_layer.nneurons() == dependency_vocab_size);

  IndexType bout_offset = 0;
  auto it_l2r_decoderatom = decoder_unit_.l2r.decoderatoms.begin();
  auto it_r2l_decoderatom = decoder_unit_.r2l.decoderatoms.begin();
  ++it_l2r_decoderatom; //! \note: skip <root>.
  ++it_r2l_decoderatom; //! \note: skip <root>.
  for (IndexType t = 1; t < ptr_databatch->sequence_length(); t++) {
    auto it_output_activations = output_unit_.output_layer.activations().begin() + dependency_vocab_size * bout_offset;
    auto it_next_output_activations = it_output_activations;

    auto it_datapoint = it_datapoints_begin;

    IndexType k = 0; //<! Index in batch.
    for (IndexType idx : idx_datapoints) {
      memnet::DataPointToken tok = ptr_databatch->data(idx, t);
      //! \note Skips the blank tokens.
      if (tok.form != word_vocab().blank_idx()) {
        it_output_activations = it_next_output_activations;
        it_next_output_activations += output_unit_.output_layer.nneurons();

        tok.dependency = static_cast<IndexType>(distance(it_output_activations,
                                                         max_element(it_output_activations, it_next_output_activations)));

        tok.headword_position = 0;
        ActivationType max_log_ac = log(it_l2r_decoderatom->attention_unit.hidden1_layer.activations(k, 0));
        max_log_ac += log(it_r2l_decoderatom->attention_unit.hidden1_layer.activations(k, 0));
        for (IndexType tt = 1; tt < it_l2r_decoderatom->attention_unit.size; tt++) {
          //! \note Skips the word itself.
          if (tt == t) {
            continue;
          }
          //! \note Skips the <blank>.
          if (ptr_databatch->data(idx, tt).form == word_vocab().blank_idx()) {
            break;
          }

          ActivationType log_ac = log(it_l2r_decoderatom->attention_unit.hidden1_layer.activations(k, tt));
          log_ac += log(it_r2l_decoderatom->attention_unit.hidden1_layer.activations(k, tt));
          if (log_ac > max_log_ac) {
            max_log_ac = log_ac;
            tok.headword_position = tt;
          }
        }

        it_datapoint->push_back(tok);
        ++it_datapoint;
      }

      k++;
    }

    bout_offset += bs;
    ++it_l2r_decoderatom;
    ++it_r2l_decoderatom;
  }
}

void CMemNetDParser::MSTPredict(const DataPointBatch *ptr_databatch,
                                const vector<IndexType> &idx_datapoints,
                                vector<vector<DataPointToken>> &datapoints) {
  assert(decoder_unit_.size == ptr_databatch->sequence_length());
  assert(output_unit_.size == ptr_databatch->sequence_length() - 1);
  assert(!idx_datapoints.empty());

  IndexType last_num_datapoints = static_cast<IndexType>(datapoints.size());
  datapoints.resize(datapoints.size() + idx_datapoints.size());
  auto it_datapoints_begin = datapoints.begin() + last_num_datapoints;

  const IndexType bs = modelopts().batch_size;
  IndexType dependency_vocab_size = dependency_vocab().size();
  assert(output_unit_.output_layer.nneurons() == dependency_vocab_size);

  typedef boost::property<boost::edge_weight_t, ActivationType> EdgeProperty;
  typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, EdgeProperty> Graph;
  typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
  typedef boost::graph_traits<Graph>::edge_descriptor Edge;

  //! Predict arc labels and construct graphs for datapoints.
  vector<Graph> graphs;
  for (IndexType k = 0; k < bs; k++) {
    graphs.push_back(Graph(ptr_databatch->sequence_length()));
  }

  IndexType bout_offset = 0;
  auto it_l2r_decoderatom = decoder_unit_.l2r.decoderatoms.begin();
  auto it_r2l_decoderatom = decoder_unit_.r2l.decoderatoms.begin();
  ++it_l2r_decoderatom; //! \note: skip <root>.
  ++it_r2l_decoderatom; //! \note: skip <root>.
  for (IndexType t = 1; t < ptr_databatch->sequence_length(); t++) {
    auto it_output_activations = output_unit_.output_layer.activations().begin() + dependency_vocab_size * bout_offset;
    auto it_next_output_activations = it_output_activations;

    auto it_datapoint = it_datapoints_begin;

    IndexType k = 0; //<! Index in batch.
    for (IndexType idx : idx_datapoints) {
      memnet::DataPointToken tok = ptr_databatch->data(idx, t);
      auto &g = graphs[k];
      //! \note Skips the blank tokens.
      if (tok.form != word_vocab().blank_idx()) {
        it_output_activations = it_next_output_activations;
        it_next_output_activations += output_unit_.output_layer.nneurons();

        tok.dependency = static_cast<IndexType>(distance(it_output_activations,
                                                         max_element(it_output_activations, it_next_output_activations)));

        tok.headword_position = 0;
        for (IndexType tt = 0; tt < it_l2r_decoderatom->attention_unit.size; tt++) {
          //! \note Skips the word itself.
          if (tt == t) {
            continue;
          }

          //! \note Skips the <blank>.
          if (ptr_databatch->data(idx, tt).form == word_vocab().blank_idx()) {
            break;
          }

          ActivationType log_ac = log(it_l2r_decoderatom->attention_unit.hidden1_layer.activations(k, tt));
          log_ac += log(it_r2l_decoderatom->attention_unit.hidden1_layer.activations(k, tt));
          //! source: tt
          //! target: t
          boost::add_edge(tt, t, log_ac, g);
        }

        it_datapoint->push_back(tok);
        ++it_datapoint;
      }

      k++;
    }

    bout_offset += bs;
    ++it_l2r_decoderatom;
    ++it_r2l_decoderatom;
  }

  auto it_datapoint = it_datapoints_begin;
  for (auto &g : graphs) {
    //! This is how we can get a property map that gives the weights of
    //! the edges.
    auto weights = get(boost::edge_weight_t(), g);
    //! This is how we can get a property map mapping the vertices to
    //! integer indices.
    auto vertex_indices =  get(boost::vertex_index_t(), g);
    //! Find the maximum branching.
    vector<Edge> branching;
    edmonds_optimum_branching<true, true, true>(g,
                                                vertex_indices,
                                                weights,
                                                static_cast<Vertex *>(0),
                                                static_cast<Vertex *>(0),
                                                std::back_inserter(branching));
    //! Save the edges of the maximum branching
    BOOST_FOREACH(Edge e, branching) {
      memnet::DataPointToken &tok = (*it_datapoint)[boost::target(e, g) - 1];
      tok.headword_position = static_cast<IndexType>(boost::source(e, g));
    }
    ++it_datapoint;
  }
}

bool CMemNetDParser::CheckParamsImpl() {
  if (num_composition_neurons_ == 0) {
    cerr << "num_composition_neurons_ should be positive!" << endl;
    return false;
  }
  if (num_projection_neurons_ == 0) {
    cerr << "num_projection_neurons_ should be positive!" << endl;
    return false;
  }
  if (num_hidden_neurons_ == 0) {
    cerr << "num_hidden_neurons_ should be positive!" << endl;
    return false;
  }
  return true;
}

void CMemNetDParser::PrintParamsImpl() {
  cout << "num_composition_neurons_: " << num_composition_neurons_ << endl;
  cout << "num_projection_neurons_: " << num_projection_neurons_ << endl;
  cout << "num_hidden_neurons_: " << num_hidden_neurons_ << endl;
}

void CMemNetDParser::ReadMemNetImpl(ifstream &ifs) {
  tialnn::read_single(ifs, num_composition_neurons_);
  tialnn::read_single(ifs, num_projection_neurons_);
  tialnn::read_single(ifs, num_hidden_neurons_);

  AllocateModel();

  //! \note Registerd connections are loaded automatically.
  //! Un-registered connections need to be loaded here.
}

void CMemNetDParser::WriteMemNetImpl(ofstream &ofs) {
  tialnn::write_single(ofs, num_composition_neurons_);
  tialnn::write_single(ofs, num_projection_neurons_);
  tialnn::write_single(ofs, num_hidden_neurons_);

  //! \note Registerd connections are saved automatically.
  //! Un-registered connections need to be saved here.
}

} // namespace cmemnet
