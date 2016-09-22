/*!
 * \file cmemnet_utils.cpp
 *
 * 
 * \version 0.0.6
 */

#include "cmemnet_utils.h"

// tialnn::IndexType
// tialnn::ActivationType
// tialnn::ErrorType
#include <tialnn/base.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::IdentityNeurons
#include <tialnn/neuron/identity_neurons.h>
// tialnn::SoftmaxNeurons
#include <tialnn/neuron/softmax_neurons.h>
// cmemnet::CMemNetInputUnit
#include "cmemnet_input_unit.h"
// cmemenet::CMemNetProjectionUnit
#include "cmemnet_projection_unit.h"
// cmemnet::CMemNetEncoderUnit
#include "cmemnet_encoder_unit.h"
// cmemnet::CMemNetAttentionUnit
#include "cmemnet_attention_unit.h"
// cmemnet::CMemNetDecoderUnit
#include "cmemnet_decoder_unit.h"
// cmemnet::CMemNetOutputUnit
#include "cmemnet_output_unit.h"

using tialnn::IndexType;
using tialnn::ActivationType;
using tialnn::ErrorType;
using tialnn::XBatchLayer;
using tialnn::IdentityNeurons;
using tialnn::SoftmaxNeurons;

namespace cmemnet {

void CMemNetForwardPropagate(CMemNetInputUnit &input_unit) {
  assert(input_unit.size <= input_unit.capacity);

  IndexType t = 0;
  IndexType by_start = 0;
  for (auto &inputatom : input_unit.inputatoms) {
    if (t == input_unit.size) {
      break;
    }

    input_unit.composition_layer.YForwardPropagateR(inputatom.input_layer,
                                                    *input_unit.ptr_conn_input_composition,
                                                    by_start, input_unit.batchsize);

    t++;
    by_start += input_unit.batchsize;
  }
  assert(by_start == input_unit.batchsize * input_unit.size);

  input_unit.composition_layer.XForwardPropagateA(*input_unit.ptr_bias_layer, 0,
                                                  *input_unit.ptr_conn_bias_composition,
                                                  0, by_start);
  input_unit.composition_layer.XForwardPropagateA(*input_unit.ptr_bias_layer, 0,
                                                  *input_unit.ptr_conn_globalbias_composition,
                                                  0, by_start);
  input_unit.composition_layer.ComputeActivations(0, by_start);
}

void CMemNetForwardPropagate(const CMemNetInputUnit &input_unit,
                             CMemNetProjectionUnit  &projection_unit) {
  assert(input_unit.capacity == projection_unit.capacity);
  assert(input_unit.batchsize == projection_unit.batchsize);
  assert(input_unit.size <= projection_unit.capacity);
  assert(input_unit.size > 1);

  projection_unit.size = input_unit.size;
  const IndexType bs_x_sz = projection_unit.batchsize * projection_unit.size;

  projection_unit.projection_layer.XForwardPropagateR(input_unit.composition_layer, 0,
                                                      *projection_unit.ptr_conn_composition_projection,
                                                      0, bs_x_sz);
  projection_unit.projection_layer.XForwardPropagateA(*projection_unit.ptr_bias_layer, 0,
                                                      *projection_unit.ptr_conn_bias_projection,
                                                      0, bs_x_sz);
  projection_unit.projection_layer.XForwardPropagateA(*projection_unit.ptr_bias_layer, 0,
                                                      *projection_unit.ptr_conn_globalbias_projection,
                                                      0, bs_x_sz);
  projection_unit.projection_layer.ComputeActivations(0, bs_x_sz);
}

void CMemNetForwardPropagate(const CMemNetProjectionUnit &projection_unit,
                             CMemNetEncoderUnit &encoder_unit) {
  assert(projection_unit.capacity == encoder_unit.capacity);
  assert(projection_unit.batchsize == encoder_unit.batchsize);
  assert(projection_unit.size <= encoder_unit.capacity);
  assert(projection_unit.size > 1);

  encoder_unit.size = projection_unit.size;
  const IndexType bs_x_sz = encoder_unit.batchsize * encoder_unit.size;

  //!=====================
  //! projection -> memout
  //!======================
  assert(encoder_unit.capacity == encoder_unit.l2r.capacity);
  assert(encoder_unit.batchsize == encoder_unit.l2r.batchsize);
  encoder_unit.l2r.size = encoder_unit.size;
  encoder_unit.l2r.memout_layer.XForwardPropagateR(projection_unit.projection_layer, 0,
                                                   *encoder_unit.l2r.ptr_conn_projection_memoutreset,
                                                   *encoder_unit.l2r.ptr_conn_projection_memoutupdate,
                                                   *encoder_unit.l2r.ptr_conn_projection_memouthtilde,
                                                   0, bs_x_sz);


  assert(encoder_unit.capacity == encoder_unit.r2l.capacity);
  assert(encoder_unit.batchsize == encoder_unit.r2l.batchsize);
  encoder_unit.r2l.size = projection_unit.size;
  encoder_unit.r2l.memout_layer.XForwardPropagateR(projection_unit.projection_layer, 0,
                                                   *encoder_unit.r2l.ptr_conn_projection_memoutreset,
                                                   *encoder_unit.r2l.ptr_conn_projection_memoutupdate,
                                                   *encoder_unit.r2l.ptr_conn_projection_memouthtilde,
                                                   0, bs_x_sz);


  //!=====================
  //! prev_memout -> memout
  //!======================
  IndexType t = 0;
  encoder_unit.l2r.memout_layer.ComputeActivations(*encoder_unit.l2r.ptr_init_hidden_layer, t);
  while (t < encoder_unit.size - 1) {
    t++;
    encoder_unit.l2r.memout_layer.ComputeActivations(t - 1, t);
  }

  encoder_unit.r2l.memout_layer.ComputeActivations(*encoder_unit.r2l.ptr_init_hidden_layer, t);
  assert(t == encoder_unit.size - 1);
  while (t > 0) {
    t--;
    encoder_unit.r2l.memout_layer.ComputeActivations(t + 1, t);
  }

  //!======================
  //! memout     -> memin
  //!======================
  encoder_unit.l2r.memin_layer.XForwardPropagateR(encoder_unit.l2r.memout_layer, 0,
                                                  *encoder_unit.l2r.ptr_conn_memout_memin,
                                                  0, bs_x_sz);
  encoder_unit.l2r.memin_layer.ComputeActivations(0, bs_x_sz);


  encoder_unit.r2l.memin_layer.XForwardPropagateR(encoder_unit.r2l.memout_layer, 0,
                                                  *encoder_unit.r2l.ptr_conn_memout_memin,
                                                  0, bs_x_sz);
  encoder_unit.r2l.memin_layer.ComputeActivations(0, bs_x_sz);
}

void CMemNetForwardPropagate(const CMemNetEncoderUnit &encoder_unit,
                             const XBatchLayer<IdentityNeurons> &query_layer,
                             IndexType bq_start,
                             CMemNetAttentionUnit &attention_unit) {
  assert(encoder_unit.capacity == attention_unit.capacity);
  assert(encoder_unit.batchsize == attention_unit.batchsize);
  assert(encoder_unit.size <= attention_unit.capacity);
  assert(encoder_unit.size > 1);

  attention_unit.size = encoder_unit.size;
  const IndexType bs_x_sz = attention_unit.batchsize * attention_unit.size;

  //!=====================
  //! encoder.l2r.memin -> l2rhidden0
  //! encoder.r2l.memin -> r2lhidden0
  //!======================
  //! tanh(encoder_unit.memin_layer[:, :, te] + decoder_unit.memin_layer[:, :, td])
  //! -> attention_unit.hidden0_layer[:, :, te] 
  attention_unit.hidden0_layer.XForwardBPropagateR(encoder_unit.l2r.memin_layer, 0,
                                                   1.0f,
                                                   0, bs_x_sz);
  attention_unit.hidden0_layer.XForwardBPropagateA(encoder_unit.r2l.memin_layer, 0,
                                                   1.0f,
                                                   0, bs_x_sz);
  IndexType by_start = 0;
  for (IndexType t = 0; t < attention_unit.size; t++) {
    attention_unit.hidden0_layer.XForwardBPropagateA(query_layer, bq_start,
                                                     1.0f,
                                                     by_start, attention_unit.batchsize);
    by_start += attention_unit.batchsize;
  }
  assert(by_start == bs_x_sz);
  attention_unit.hidden0_layer.ComputeActivations(0, by_start);

  //!=====================
  //! l2rhidden0 -> hidden1
  //! r2lhidden0 -> hidden1
  //!======================
  attention_unit.hidden1_layer.XForwardPropagateR(attention_unit.hidden0_layer, 0,
                                                  *attention_unit.ptr_attention_conn_hidden0_hidden1,
                                                  0, bs_x_sz);
  attention_unit.hidden1_layer.XForwardPropagateA(*attention_unit.ptr_bias_layer, 0,
                                                  *attention_unit.ptr_attention_conn_bias_hidden1,
                                                  0, bs_x_sz);
  attention_unit.hidden1_layer.ComputeActivations(attention_unit.size);

  //!=====================
  //! hidden1_layer .* encoder.l2r.memout_layer -> l2rhidden_layer
  //! hidden1_layer .* encoder.r2l.memout_layer -> r2lhidden_layer
  //!=====================
  IndexType benc_start = 0;
  IndexType batt_start = 0;
  attention_unit.l2rhidden_layer.XForwardBPropagateR(encoder_unit.l2r.memout_layer, benc_start,
                                                     attention_unit.hidden1_layer, batt_start,
                                                     0, attention_unit.batchsize);
  attention_unit.r2lhidden_layer.XForwardBPropagateR(encoder_unit.r2l.memout_layer, benc_start,
                                                     attention_unit.hidden1_layer, batt_start,
                                                     0, attention_unit.batchsize);
  for (IndexType t = 1; t < attention_unit.size; t++) {
    benc_start += encoder_unit.batchsize;
    batt_start += attention_unit.batchsize;
    attention_unit.l2rhidden_layer.XForwardBPropagateA(encoder_unit.l2r.memout_layer, benc_start,
                                                       attention_unit.hidden1_layer, batt_start,
                                                       0, attention_unit.batchsize);
    attention_unit.r2lhidden_layer.XForwardBPropagateA(encoder_unit.r2l.memout_layer, benc_start,
                                                       attention_unit.hidden1_layer, batt_start,
                                                       0, attention_unit.batchsize);
  }
  assert(benc_start + encoder_unit.batchsize == bs_x_sz);
  assert(batt_start + attention_unit.batchsize == bs_x_sz);
  attention_unit.l2rhidden_layer.ComputeActivations(0, attention_unit.batchsize);
  attention_unit.r2lhidden_layer.ComputeActivations(0, attention_unit.batchsize);
}

void CMemNetForwardPropagate(const CMemNetProjectionUnit &projection_unit,
                             const CMemNetEncoderUnit &encoder_unit,
                             CMemNetDecoderUnit &decoder_unit) {
  assert(projection_unit.capacity == decoder_unit.capacity);
  assert(projection_unit.batchsize == decoder_unit.batchsize);
  assert(projection_unit.size <= decoder_unit.capacity);
  assert(projection_unit.size > 1);

  decoder_unit.size = projection_unit.size;
  const IndexType bs_x_sz = decoder_unit.batchsize * decoder_unit.size;

  //!=====================
  //! projection -> memout
  //! \note decoder_unit.{l2r,r2l} [:, :, 0] is for <ROOT> and not used.
  //!======================
  assert(decoder_unit.l2r.capacity == decoder_unit.capacity);
  assert(decoder_unit.l2r.batchsize == decoder_unit.batchsize);
  decoder_unit.l2r.size = decoder_unit.size;
  decoder_unit.l2r.memout_layer.XForwardPropagateR(projection_unit.projection_layer, decoder_unit.l2r.batchsize,
                                                   *decoder_unit.l2r.ptr_conn_projection_memoutreset,
                                                   *decoder_unit.l2r.ptr_conn_projection_memoutupdate,
                                                   *decoder_unit.l2r.ptr_conn_projection_memouthtilde,
                                                   decoder_unit.l2r.batchsize, bs_x_sz - decoder_unit.l2r.batchsize);

  assert(decoder_unit.r2l.capacity == decoder_unit.capacity);
  assert(decoder_unit.r2l.batchsize == decoder_unit.batchsize);
  decoder_unit.r2l.size = decoder_unit.size;
  decoder_unit.r2l.memout_layer.XForwardPropagateR(projection_unit.projection_layer, decoder_unit.r2l.batchsize,
                                                   *decoder_unit.r2l.ptr_conn_projection_memoutreset,
                                                   *decoder_unit.r2l.ptr_conn_projection_memoutupdate,
                                                   *decoder_unit.r2l.ptr_conn_projection_memouthtilde,
                                                   decoder_unit.r2l.batchsize, bs_x_sz - decoder_unit.r2l.batchsize);

  //! Left-to-right forward
  //!=====================
  //! prev_memout -> memout
  //!======================
  IndexType t = 1;
  IndexType bdec_start = decoder_unit.l2r.batchsize;
  decoder_unit.l2r.memout_layer.ComputeActivations(*decoder_unit.l2r.ptr_init_hidden_layer, t);
  //!======================
  //! memout     -> memin
  //!======================
  assert(decoder_unit.l2r.memout_layer.atom_batchsize() == decoder_unit.batchsize);
  decoder_unit.l2r.memin_layer.XForwardPropagateR(decoder_unit.l2r.memout_layer, bdec_start,
                                                  *decoder_unit.l2r.ptr_conn_memout_memin,
                                                  bdec_start, decoder_unit.l2r.batchsize);
  decoder_unit.l2r.memin_layer.ComputeActivations(bdec_start, decoder_unit.l2r.batchsize);

  //!=====================
  //! memin -> attention_unit 
  //!======================
  auto it_decoderatom = decoder_unit.l2r.decoderatoms.begin();
  ++it_decoderatom; //! \note decoderatoms[0] is for <ROOT> and not used.
  CMemNetForwardPropagate(encoder_unit,
                          decoder_unit.l2r.memin_layer,
                          bdec_start,
                          it_decoderatom->attention_unit);

  while (t < decoder_unit.size - 1) {
    t++;
    bdec_start += decoder_unit.batchsize;

    //!=====================
    //! prev attention_unit -> memout
    //!======================

    decoder_unit.l2r.memout_layer.XForwardPropagateA(it_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                     *decoder_unit.l2r.ptr_conn_attl2rhidden_memoutreset,
                                                     *decoder_unit.l2r.ptr_conn_attl2rhidden_memoutupdate,
                                                     *decoder_unit.l2r.ptr_conn_attl2rhidden_memouthtilde,
                                                     bdec_start, decoder_unit.batchsize);
    decoder_unit.l2r.memout_layer.XForwardPropagateA(it_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                     *decoder_unit.l2r.ptr_conn_attr2lhidden_memoutreset,
                                                     *decoder_unit.l2r.ptr_conn_attr2lhidden_memoutupdate,
                                                     *decoder_unit.l2r.ptr_conn_attr2lhidden_memouthtilde,
                                                     bdec_start, decoder_unit.batchsize);
    decoder_unit.l2r.memout_layer.ComputeActivations(t - 1, t);

    ++it_decoderatom;

    //!======================
    //! memout     -> memin
    //!======================

    decoder_unit.l2r.memin_layer.XForwardPropagateR(decoder_unit.l2r.memout_layer, bdec_start,
                                                    *decoder_unit.l2r.ptr_conn_memout_memin,
                                                    bdec_start, decoder_unit.l2r.batchsize);
    decoder_unit.l2r.memin_layer.ComputeActivations(bdec_start, decoder_unit.l2r.batchsize);
    //!=====================
    //! memout -> attention_unit 
    //!======================

    CMemNetForwardPropagate(encoder_unit,
                            decoder_unit.l2r.memin_layer,
                            bdec_start,
                            it_decoderatom->attention_unit);

  }
  assert(bdec_start == (bs_x_sz - decoder_unit.l2r.batchsize));
  assert(t == decoder_unit.size - 1);

  // Right-to-left forward
  //!=====================
  //! prev_memout -> memout
  //!======================
  decoder_unit.r2l.memout_layer.ComputeActivations(*decoder_unit.r2l.ptr_init_hidden_layer, t);
  //!======================
  //! memout     -> memin
  //!======================
  assert(decoder_unit.r2l.memout_layer.atom_batchsize() == decoder_unit.batchsize);
  decoder_unit.r2l.memin_layer.XForwardPropagateR(decoder_unit.r2l.memout_layer, bdec_start,
                                                  *decoder_unit.r2l.ptr_conn_memout_memin,
                                                  bdec_start, decoder_unit.r2l.batchsize);
  decoder_unit.r2l.memin_layer.ComputeActivations(bdec_start, decoder_unit.r2l.batchsize);

  //!=====================
  //! memout -> attention_unit 
  //!======================
  it_decoderatom = decoder_unit.r2l.decoderatoms.begin();
  it_decoderatom += (decoder_unit.r2l.size - 1);
  CMemNetForwardPropagate(encoder_unit,
                          decoder_unit.r2l.memin_layer,
                          bdec_start,
                          it_decoderatom->attention_unit);

  while (t > 1) {
    t--;
    bdec_start -= decoder_unit.batchsize;

    //!=====================
    //! prev attention_unit -> memout
    //!======================
    decoder_unit.r2l.memout_layer.XForwardPropagateA(it_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                     *decoder_unit.r2l.ptr_conn_attl2rhidden_memoutreset,
                                                     *decoder_unit.r2l.ptr_conn_attl2rhidden_memoutupdate,
                                                     *decoder_unit.r2l.ptr_conn_attl2rhidden_memouthtilde,
                                                     bdec_start, decoder_unit.batchsize);
    decoder_unit.r2l.memout_layer.XForwardPropagateA(it_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                     *decoder_unit.r2l.ptr_conn_attr2lhidden_memoutreset,
                                                     *decoder_unit.r2l.ptr_conn_attr2lhidden_memoutupdate,
                                                     *decoder_unit.r2l.ptr_conn_attr2lhidden_memouthtilde,
                                                     bdec_start, decoder_unit.batchsize);
    decoder_unit.r2l.memout_layer.ComputeActivations(t + 1, t);


    //!=====================
    //! memin -> memout 
    //!======================
    decoder_unit.r2l.memin_layer.XForwardPropagateR(decoder_unit.r2l.memout_layer, bdec_start,
                                                    *decoder_unit.r2l.ptr_conn_memout_memin,
                                                    bdec_start, decoder_unit.r2l.batchsize);
    decoder_unit.r2l.memin_layer.ComputeActivations(bdec_start, decoder_unit.r2l.batchsize);
    //!=====================
    //! memin -> attention_unit 
    //!======================
    --it_decoderatom;
    CMemNetForwardPropagate(encoder_unit,
                            decoder_unit.r2l.memin_layer,
                            bdec_start,
                            it_decoderatom->attention_unit);

  }
  assert(bdec_start == decoder_unit.r2l.batchsize);
  assert(t == 1);
  assert(--it_decoderatom == decoder_unit.r2l.decoderatoms.begin());

}

void CMemNetForwardPropagate(const CMemNetDecoderUnit &decoder_unit,
                             CMemNetOutputUnit &output_unit) {
  assert(decoder_unit.capacity == output_unit.capacity + 1);
  assert(decoder_unit.batchsize == output_unit.batchsize);
  assert(decoder_unit.size <= output_unit.capacity + 1);
  assert(decoder_unit.size > 1);

  output_unit.size = decoder_unit.size - 1;
  const IndexType bs_x_sz = output_unit.batchsize * output_unit.size;

  //! =====================================================
  //! decoder_unit.memout_layer -> output_unit.output_layer
  output_unit.output_layer.XForwardPropagateR(decoder_unit.l2r.memout_layer, 
                                              decoder_unit.l2r.batchsize, //!< \note decoder_unit[:, :, 0] is used for <ROOT>.
                                              *output_unit.ptr_conn_decoderl2rmemout_output,
                                              0, bs_x_sz);
  output_unit.output_layer.XForwardPropagateA(decoder_unit.r2l.memout_layer,
                                              decoder_unit.r2l.batchsize, //!< \note decoder_unit[:, :, 0] is used for <ROOT>.
                                              *output_unit.ptr_conn_decoderr2lmemout_output,
                                              0, bs_x_sz);
  //! =====================================================

  //! =====================================================
  //! attention_unit.hidden_layer -> output_unit.output_layer
  auto it_l2r_decoderatom = decoder_unit.l2r.decoderatoms.begin();
  auto it_r2l_decoderatom = decoder_unit.r2l.decoderatoms.begin();
  ++it_l2r_decoderatom; //!< \note decoder_unit[:, :, 0] is used for <ROOT>.
  ++it_r2l_decoderatom; //!< \note decoder_unit[:, :, 0] is used for <ROOT>.
  IndexType by_start = 0;
  for (IndexType t = 0; t < output_unit.size; t++) {
    output_unit.output_layer.XForwardPropagateA(it_l2r_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                *output_unit.ptr_conn_l2ratt_l2rhidden_output,
                                                by_start, output_unit.batchsize);
    output_unit.output_layer.XForwardPropagateA(it_l2r_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                *output_unit.ptr_conn_l2ratt_r2lhidden_output,
                                                by_start, output_unit.batchsize);
    output_unit.output_layer.XForwardPropagateA(it_r2l_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                *output_unit.ptr_conn_r2latt_l2rhidden_output,
                                                by_start, output_unit.batchsize);
    output_unit.output_layer.XForwardPropagateA(it_r2l_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                *output_unit.ptr_conn_r2latt_r2lhidden_output,
                                                by_start, output_unit.batchsize);
    by_start += output_unit.batchsize;
    ++it_l2r_decoderatom;
    ++it_r2l_decoderatom;
  }
  assert(by_start == bs_x_sz);
  //! =====================================================

  output_unit.output_layer.XForwardPropagateA(*output_unit.ptr_bias_layer, 0,
                                              *output_unit.ptr_conn_bias_output,
                                              0, bs_x_sz);
  output_unit.output_layer.XForwardPropagateA(*output_unit.ptr_bias_layer, 0,
                                              *output_unit.ptr_conn_globalbias_output,
                                              0, bs_x_sz);

  output_unit.output_layer.ComputeActivations(0, bs_x_sz);
}

void CMemNetBackwardPropagate(CMemNetDecoderUnit &decoder_unit,
                              CMemNetOutputUnit &output_unit) {
  assert(output_unit.size + 1 == decoder_unit.size);
  assert(output_unit.size > 0);

  const IndexType bs_x_sz = output_unit.batchsize * output_unit.size;
  output_unit.output_layer.set_errors(0, bs_x_sz, output_unit.output_errors);

  output_unit.output_layer.XBackwardPropagateNA(*output_unit.ptr_bias_layer, 0,
                                                *output_unit.ptr_conn_globalbias_output,
                                                0, bs_x_sz);
  output_unit.output_layer.XBackwardPropagateNA(*output_unit.ptr_bias_layer, 0,
                                                *output_unit.ptr_conn_bias_output,
                                                0, bs_x_sz);

  //! =====================================================
  //! attention_unit.hidden_layer <- output_unit.output_layer
  auto it_l2r_decoderatom = decoder_unit.l2r.decoderatoms.begin();
  auto it_r2l_decoderatom = decoder_unit.r2l.decoderatoms.begin();
  it_l2r_decoderatom += decoder_unit.size;
  it_r2l_decoderatom += decoder_unit.size;
  IndexType by_start = bs_x_sz;
  for (IndexType t = 0; t < output_unit.size; t++) {
    --it_l2r_decoderatom;
    --it_r2l_decoderatom;
    by_start -= output_unit.batchsize;
    output_unit.output_layer.XBackwardPropagateRA(it_r2l_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                  *output_unit.ptr_conn_r2latt_r2lhidden_output,
                                                  by_start, output_unit.batchsize);
    output_unit.output_layer.XBackwardPropagateRA(it_r2l_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                  *output_unit.ptr_conn_r2latt_l2rhidden_output,
                                                  by_start, output_unit.batchsize);
    output_unit.output_layer.XBackwardPropagateRA(it_l2r_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                  *output_unit.ptr_conn_l2ratt_r2lhidden_output,
                                                  by_start, output_unit.batchsize);
    output_unit.output_layer.XBackwardPropagateRA(it_l2r_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                  *output_unit.ptr_conn_l2ratt_l2rhidden_output,
                                                  by_start, output_unit.batchsize);
  }
  //!< \note decoder_unit[:, :, 0] is used for <ROOT>.
  assert(--it_l2r_decoderatom == decoder_unit.l2r.decoderatoms.begin());
  assert(--it_r2l_decoderatom == decoder_unit.r2l.decoderatoms.begin());
  assert(by_start == 0);
  //! =====================================================

  //! =====================================================
  //! decoder_unit.memout_layer <- output_unit.output_layer
  output_unit.output_layer.XBackwardPropagateRA(decoder_unit.r2l.memout_layer,
                                                decoder_unit.r2l.batchsize, //!< \note decoder_unit[:, :, 0] is used for <ROOT>.
                                                *output_unit.ptr_conn_decoderr2lmemout_output,
                                                0, bs_x_sz);
  output_unit.output_layer.XBackwardPropagateRA(decoder_unit.l2r.memout_layer,
                                                decoder_unit.l2r.batchsize, //!< \note decoder_unit[:, :, 0] is used for <ROOT>.
                                                *output_unit.ptr_conn_decoderl2rmemout_output,
                                                0, bs_x_sz);
  //! =====================================================
}

void CMemNetBackwardPropagate(CMemNetProjectionUnit &projection_unit,
                              CMemNetEncoderUnit &encoder_unit,
                              CMemNetDecoderUnit &decoder_unit) {
  assert(decoder_unit.size == projection_unit.size);
  assert(decoder_unit.size > 1);

  const IndexType bs_x_sz = decoder_unit.batchsize * decoder_unit.size;

  //!======================
  //! Right-to-left backpropagate
  //!======================
  IndexType bdec_start = decoder_unit.r2l.batchsize;
  IndexType t = 1;
  auto it_decoderatom = decoder_unit.r2l.decoderatoms.begin();
  ++it_decoderatom;
  bool reset = true;
  while (t < decoder_unit.size - 1) {
    //!=====================
    //! memin <- attention_unit 
    //!======================
    CMemNetBackwardPropagate(encoder_unit,
                             decoder_unit.r2l.memin_layer,
                             bdec_start,
                             it_decoderatom->attention_unit,
                             reset);
    reset = false;
    ++it_decoderatom;

    //!======================
    //! memout     <- memin
    //!======================
    decoder_unit.r2l.memin_layer.ComputeErrors(bdec_start, decoder_unit.r2l.batchsize);
    decoder_unit.r2l.memin_layer.XBackwardPropagateAA(decoder_unit.r2l.memout_layer, bdec_start,
                                                      *decoder_unit.r2l.ptr_conn_memout_memin,
                                                      bdec_start, decoder_unit.r2l.batchsize);

    //!=====================
    //! prev attention_unit <- memout
    //!======================
    decoder_unit.r2l.memout_layer.ComputeErrors(t + 1, t);
    decoder_unit.r2l.memout_layer.XBackwardPropagateAA(it_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                       *decoder_unit.r2l.ptr_conn_attr2lhidden_memoutreset,
                                                       *decoder_unit.r2l.ptr_conn_attr2lhidden_memoutupdate,
                                                       *decoder_unit.r2l.ptr_conn_attr2lhidden_memouthtilde,
                                                       bdec_start, decoder_unit.batchsize);
    decoder_unit.r2l.memout_layer.XBackwardPropagateAA(it_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                       *decoder_unit.r2l.ptr_conn_attl2rhidden_memoutreset,
                                                       *decoder_unit.r2l.ptr_conn_attl2rhidden_memoutupdate,
                                                       *decoder_unit.r2l.ptr_conn_attl2rhidden_memouthtilde,
                                                       bdec_start, decoder_unit.batchsize);


    bdec_start += decoder_unit.r2l.batchsize;
    t++;
  }

  CMemNetBackwardPropagate(encoder_unit,
                           decoder_unit.r2l.memin_layer,
                           bdec_start,
                           it_decoderatom->attention_unit,
                           false);
  //assert(it_decoderatom == decoder_unit.r2l.decoderatoms.begin() + (decoder_unit.r2l.size - 1));

  decoder_unit.r2l.memin_layer.ComputeErrors(bdec_start, decoder_unit.r2l.batchsize);
  decoder_unit.r2l.memin_layer.XBackwardPropagateAA(decoder_unit.r2l.memout_layer, bdec_start,
                                                    *decoder_unit.r2l.ptr_conn_memout_memin,
                                                    bdec_start, decoder_unit.r2l.batchsize);

  decoder_unit.r2l.memout_layer.ComputeErrors(*decoder_unit.r2l.ptr_init_hidden_layer, t);

  assert(t == decoder_unit.size - 1);
  assert(bdec_start == bs_x_sz - decoder_unit.r2l.batchsize);

  //!======================
  //! Left-to-right backpropagate
  //!======================
  it_decoderatom = decoder_unit.l2r.decoderatoms.begin();
  it_decoderatom += (decoder_unit.l2r.size - 1);

  while (t > 1) {
    //! \note encoder.{l2r,r2l} has been reseted.
    CMemNetBackwardPropagate(encoder_unit,
                             decoder_unit.l2r.memin_layer,
                             bdec_start,
                             it_decoderatom->attention_unit,
                             false);

    decoder_unit.l2r.memin_layer.ComputeErrors(bdec_start, decoder_unit.l2r.batchsize);
    decoder_unit.l2r.memin_layer.XBackwardPropagateAA(decoder_unit.l2r.memout_layer, bdec_start,
                                                      *decoder_unit.l2r.ptr_conn_memout_memin,
                                                      bdec_start, decoder_unit.l2r.batchsize);

    --it_decoderatom;

    //!=====================
    //! prev_memout <- memout
    //!======================
    decoder_unit.l2r.memout_layer.ComputeErrors(t - 1, t);
    decoder_unit.l2r.memout_layer.XBackwardPropagateAA(it_decoderatom->attention_unit.r2lhidden_layer, 0,
                                                       *decoder_unit.l2r.ptr_conn_attr2lhidden_memoutreset,
                                                       *decoder_unit.l2r.ptr_conn_attr2lhidden_memoutupdate,
                                                       *decoder_unit.l2r.ptr_conn_attr2lhidden_memouthtilde,
                                                       bdec_start, decoder_unit.batchsize);
    decoder_unit.l2r.memout_layer.XBackwardPropagateAA(it_decoderatom->attention_unit.l2rhidden_layer, 0,
                                                       *decoder_unit.l2r.ptr_conn_attl2rhidden_memoutreset,
                                                       *decoder_unit.l2r.ptr_conn_attl2rhidden_memoutupdate,
                                                       *decoder_unit.l2r.ptr_conn_attl2rhidden_memouthtilde,
                                                       bdec_start, decoder_unit.batchsize);

    bdec_start -= decoder_unit.l2r.batchsize;
    t--;
  }
  CMemNetBackwardPropagate(encoder_unit,
                           decoder_unit.l2r.memin_layer,
                           bdec_start,
                           it_decoderatom->attention_unit,
                           false);
  assert(--it_decoderatom == decoder_unit.l2r.decoderatoms.begin());

  decoder_unit.l2r.memin_layer.ComputeErrors(bdec_start, decoder_unit.l2r.batchsize);
  decoder_unit.l2r.memin_layer.XBackwardPropagateAA(decoder_unit.l2r.memout_layer, bdec_start,
                                                    *decoder_unit.l2r.ptr_conn_memout_memin,
                                                    bdec_start, decoder_unit.l2r.batchsize);

  decoder_unit.l2r.memout_layer.ComputeErrors(*decoder_unit.l2r.ptr_init_hidden_layer, t);
  assert(bdec_start == decoder_unit.l2r.batchsize);
  assert(t == 1);


  //!=====================
  //! projection <- memout
  //!======================
  decoder_unit.r2l.memout_layer.XBackwardPropagateRA(projection_unit.projection_layer, decoder_unit.r2l.batchsize,
                                                     *decoder_unit.r2l.ptr_conn_projection_memoutreset,
                                                     *decoder_unit.r2l.ptr_conn_projection_memoutupdate,
                                                     *decoder_unit.r2l.ptr_conn_projection_memouthtilde,
                                                     decoder_unit.r2l.batchsize, bs_x_sz - decoder_unit.r2l.batchsize);
  decoder_unit.l2r.memout_layer.XBackwardPropagateAA(projection_unit.projection_layer, decoder_unit.l2r.batchsize,
                                                     *decoder_unit.l2r.ptr_conn_projection_memoutreset,
                                                     *decoder_unit.l2r.ptr_conn_projection_memoutupdate,
                                                     *decoder_unit.l2r.ptr_conn_projection_memouthtilde,
                                                     decoder_unit.l2r.batchsize, bs_x_sz - decoder_unit.l2r.batchsize);
}

void CMemNetBackwardPropagate(CMemNetEncoderUnit &encoder_unit,
                              XBatchLayer<IdentityNeurons> &query_layer,
                              IndexType bq_start,
                              CMemNetAttentionUnit &attention_unit,
                              bool reset) {
  assert(attention_unit.size == encoder_unit.size);
  assert(attention_unit.size > 0);

  const IndexType bs_x_sz = attention_unit.batchsize * attention_unit.size;

  //!=====================
  //! hidden1_layer .* encoder.l2r.memout_layer <- l2rhidden_layer
  //! hidden1_layer .* encoder.r2l.memout_layer <- r2lhidden_layer
  //!=====================
  attention_unit.r2lhidden_layer.ComputeErrors(0, attention_unit.batchsize);
  attention_unit.l2rhidden_layer.ComputeErrors(0, attention_unit.batchsize);

  IndexType batt_start = bs_x_sz - attention_unit.batchsize;
  IndexType benc_start = bs_x_sz - encoder_unit.batchsize;
  for (IndexType t = 1; t < attention_unit.size; t++) {
    if (reset) {
      attention_unit.r2lhidden_layer.XBackwardBPropagateR(encoder_unit.r2l.memout_layer, benc_start,
                                                          attention_unit.hidden1_layer, batt_start,
                                                          0, attention_unit.batchsize);
      attention_unit.l2rhidden_layer.XBackwardBPropagateR(encoder_unit.l2r.memout_layer, benc_start,
                                                          attention_unit.hidden1_layer, batt_start,
                                                          0, attention_unit.batchsize);
    } else {
      attention_unit.r2lhidden_layer.XBackwardBPropagateA(encoder_unit.r2l.memout_layer, benc_start,
                                                          attention_unit.hidden1_layer, batt_start,
                                                          0, attention_unit.batchsize);
      attention_unit.l2rhidden_layer.XBackwardBPropagateA(encoder_unit.l2r.memout_layer, benc_start,
                                                          attention_unit.hidden1_layer, batt_start,
                                                          0, attention_unit.batchsize);
    }
    attention_unit.r2lhidden_layer.XBackwardPropagateR(attention_unit.hidden1_layer, batt_start,
                                                       encoder_unit.r2l.memout_layer, benc_start,
                                                       0, attention_unit.batchsize);
    attention_unit.l2rhidden_layer.XBackwardPropagateA(attention_unit.hidden1_layer, batt_start, 
                                                       encoder_unit.l2r.memout_layer, benc_start,
                                                       0, attention_unit.batchsize);
    batt_start -= attention_unit.batchsize;
    benc_start -= encoder_unit.batchsize;
  }
  if (reset) {
    attention_unit.r2lhidden_layer.XBackwardBPropagateR(encoder_unit.r2l.memout_layer, benc_start,
                                                        attention_unit.hidden1_layer, batt_start,
                                                        0, attention_unit.batchsize);
    attention_unit.l2rhidden_layer.XBackwardBPropagateR(encoder_unit.l2r.memout_layer, benc_start,
                                                        attention_unit.hidden1_layer, batt_start,
                                                        0, attention_unit.batchsize);
  } else {
    attention_unit.r2lhidden_layer.XBackwardBPropagateA(encoder_unit.r2l.memout_layer, benc_start,
                                                        attention_unit.hidden1_layer, batt_start,
                                                        0, attention_unit.batchsize);
    attention_unit.l2rhidden_layer.XBackwardBPropagateA(encoder_unit.l2r.memout_layer, benc_start,
                                                        attention_unit.hidden1_layer, batt_start,
                                                        0, attention_unit.batchsize);
  }
  attention_unit.r2lhidden_layer.XBackwardPropagateR(attention_unit.hidden1_layer, batt_start,
                                                     encoder_unit.r2l.memout_layer, benc_start,
                                                     0, attention_unit.batchsize);
  attention_unit.l2rhidden_layer.XBackwardPropagateA(attention_unit.hidden1_layer, batt_start,
                                                     encoder_unit.l2r.memout_layer, benc_start,
                                                     0, attention_unit.batchsize);
  assert(batt_start == 0);
  assert(benc_start == 0);

  //!=====================
  //! l2rhidden0 <- hidden1
  //!======================
  attention_unit.hidden1_layer.ComputeErrors(attention_unit.size);
  //! \note Adds the label errors.
  attention_unit.hidden1_layer.AccumulateErrors(attention_unit.hidden1_label_errors, 0, bs_x_sz);
  attention_unit.hidden1_layer.XBackwardPropagateNA(*attention_unit.ptr_bias_layer, 0,
                                                    *attention_unit.ptr_attention_conn_bias_hidden1,
                                                    0, bs_x_sz);
  attention_unit.hidden1_layer.XBackwardPropagateRA(attention_unit.hidden0_layer, 0,
                                                    *attention_unit.ptr_attention_conn_hidden0_hidden1,
                                                    0, bs_x_sz);

  //!=====================
  //! encoder.l2r.memin <- l2rhidden0
  //! encoder.r2l.memin <- r2lhidden0
  //!======================
  //! attention_unit.hidden0_layer[:, :, te] 
  //! <- tanh(encoder_unit.memin_layer[:, :, te] + decoder_unit.memin_layer[:, :, td])
  attention_unit.hidden0_layer.ComputeErrors(0, bs_x_sz);
  IndexType by_start = bs_x_sz - attention_unit.batchsize;
  attention_unit.hidden0_layer.XBackwardBPropagateR(query_layer, bq_start,
                                                    1.0f,
                                                    by_start, attention_unit.batchsize);
  for (IndexType t = 1; t < attention_unit.size; t++) {
    by_start -= attention_unit.batchsize;
    attention_unit.hidden0_layer.XBackwardBPropagateA(query_layer, bq_start,
                                                      1.0f,
                                                      by_start, attention_unit.batchsize);
  }
  assert(by_start == 0);
  if (reset) {
    attention_unit.hidden0_layer.XBackwardBPropagateR(encoder_unit.r2l.memin_layer, 0,
                                                      1.0f,
                                                      0, bs_x_sz);
    attention_unit.hidden0_layer.XBackwardBPropagateR(encoder_unit.l2r.memin_layer, 0,
                                                      1.0f,
                                                      0, bs_x_sz);
  } else {
    attention_unit.hidden0_layer.XBackwardBPropagateA(encoder_unit.r2l.memin_layer, 0,
                                                      1.0f,
                                                      0, bs_x_sz);
    attention_unit.hidden0_layer.XBackwardBPropagateA(encoder_unit.l2r.memin_layer, 0,
                                                      1.0f,
                                                      0, bs_x_sz);
  }
}

void CMemNetBackwardPropagate(CMemNetProjectionUnit &projection_unit,
                              CMemNetEncoderUnit &encoder_unit) {
  assert(encoder_unit.size == projection_unit.size);
  assert(encoder_unit.size > 1);

  const IndexType bs_x_sz = encoder_unit.batchsize * encoder_unit.size;

  //!=====================
  //! memout     <- memin
  //!======================
  encoder_unit.r2l.memin_layer.ComputeErrors(0, bs_x_sz);
  encoder_unit.r2l.memin_layer.XBackwardPropagateAA(encoder_unit.r2l.memout_layer, 0,
                                                    *encoder_unit.r2l.ptr_conn_memout_memin,
                                                    0, bs_x_sz);

  encoder_unit.l2r.memin_layer.ComputeErrors(0, bs_x_sz);
  encoder_unit.l2r.memin_layer.XBackwardPropagateAA(encoder_unit.l2r.memout_layer, 0,
                                                    *encoder_unit.l2r.ptr_conn_memout_memin,
                                                    0, bs_x_sz);

  //!=====================
  //! prev_memout <- memout
  //!======================
  IndexType t = 0;
  while (t < encoder_unit.size - 1) {
    encoder_unit.r2l.memout_layer.ComputeErrors(t + 1, t);
    t++;
  }
  assert(t == encoder_unit.size - 1);
  encoder_unit.r2l.memout_layer.ComputeErrors(*encoder_unit.r2l.ptr_init_hidden_layer, t);

  while (t > 0) {
    encoder_unit.l2r.memout_layer.ComputeErrors(t - 1, t);
    t--;
  }
  encoder_unit.l2r.memout_layer.ComputeErrors(*encoder_unit.l2r.ptr_init_hidden_layer, t);
  assert(t == 0);

  //!=====================
  //! projection <- memout
  //!======================
  //! \note projection_unit[:, :, 0] was not reset.
  projection_unit.projection_layer.SetInputOfErrorsToValue(0, projection_unit.batchsize, 0.0f);
  encoder_unit.r2l.memout_layer.XBackwardPropagateAA(projection_unit.projection_layer, 0,
                                                     *encoder_unit.r2l.ptr_conn_projection_memoutreset,
                                                     *encoder_unit.r2l.ptr_conn_projection_memoutupdate,
                                                     *encoder_unit.r2l.ptr_conn_projection_memouthtilde,
                                                     0, bs_x_sz);
  encoder_unit.l2r.memout_layer.XBackwardPropagateAA(projection_unit.projection_layer, 0,
                                                     *encoder_unit.l2r.ptr_conn_projection_memoutreset,
                                                     *encoder_unit.l2r.ptr_conn_projection_memoutupdate,
                                                     *encoder_unit.l2r.ptr_conn_projection_memouthtilde,
                                                     0, bs_x_sz);

}

void CMemNetBackwardPropagate(CMemNetInputUnit &input_unit,
                              CMemNetProjectionUnit &projection_unit) {
  assert(projection_unit.size > 1);

  const IndexType bs_x_sz = projection_unit.batchsize *  projection_unit.size;

  projection_unit.projection_layer.ComputeErrors(0, bs_x_sz);
  projection_unit.projection_layer.XBackwardPropagateNA(*projection_unit.ptr_bias_layer, 0,
                                                        *projection_unit.ptr_conn_bias_projection,
                                                        0, bs_x_sz);
  projection_unit.projection_layer.XBackwardPropagateNA(*projection_unit.ptr_bias_layer, 0,
                                                        *projection_unit.ptr_conn_globalbias_projection,
                                                        0, bs_x_sz);
  projection_unit.projection_layer.XBackwardPropagateRA(input_unit.composition_layer, 0,
                                                        *projection_unit.ptr_conn_composition_projection,
                                                        0, bs_x_sz);
}

void CMemNetBackwardPropagate(CMemNetInputUnit &input_unit) {
  assert(input_unit.size > 1);

  const IndexType bs_x_sz = input_unit.batchsize * input_unit.size;
  input_unit.composition_layer.ComputeErrors(0, bs_x_sz);
  input_unit.composition_layer.XBackwardPropagateNA(*input_unit.ptr_bias_layer, 0,
                                                    *input_unit.ptr_conn_globalbias_composition,
                                                    0, bs_x_sz);
  input_unit.composition_layer.XBackwardPropagateNA(*input_unit.ptr_bias_layer, 0,
                                                    *input_unit.ptr_conn_bias_composition,
                                                    0, bs_x_sz);

  IndexType t = 0;
  IndexType by_start = 0;
  for (auto &inputatom : input_unit.inputatoms) {
    if (t == input_unit.size) {
      break;
    }

    input_unit.composition_layer.YBackwardPropagateNA(inputatom.input_layer,
                                                      *input_unit.ptr_conn_input_composition,
                                                      by_start, input_unit.batchsize);

    t++;
    by_start += input_unit.batchsize;
  }
  assert(by_start == bs_x_sz);
}

} // namespace cmemnet
