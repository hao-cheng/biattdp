/*!
 * \file cmemnet_utils.h
 *
 * 
 * \version 0.0.6
 */

#ifndef CMEMNET_CMEMNET_UTILS_H_
#define CMEMNET_CMEMNET_UTILS_H_

// tialnn::IndexType
#include <tialnn/base.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::IdentityNeurons
#include <tialnn/neuron/identity_neurons.h>
// cmemnet::CMemNetInputUnit
#include "cmemnet_input_unit.h"
// amemnet::CMemNetProjectionUnit
#include "cmemnet_projection_unit.h"
// cmemnet::CMemNetEncoderUnit
#include "cmemnet_encoder_unit.h"
// cmemnet::CMemNetAttentionUnit
#include "cmemnet_attention_unit.h"
// cmemnet::CMemNetDecoderUnit
#include "cmemnet_decoder_unit.h"
// cmemnet::CMemNetOutputUnit
#include "cmemnet_output_unit.h"

// This header defines the utilities for unit at cmemnet level.
namespace cmemnet {

//! ==========================
//! Forward propagation.
//! ==========================

//! input -> input_unit
void CMemNetForwardPropagate(CMemNetInputUnit &input_unit);

//! input_unit -> projection_unit
void CMemNetForwardPropagate(const CMemNetInputUnit &input_unit,
                             CMemNetProjectionUnit &projection_unit);

//! projection_unit     -> encoder_unit
void CMemNetForwardPropagate(const CMemNetProjectionUnit &projection_unit,
                             CMemNetEncoderUnit &encoder_unit);

//! encoder_unit                -> attention_unit
//! decoderatom.memin_layer     -> attention_unit
//! \param query_layer      decoderatom.memin_layer
void CMemNetForwardPropagate(const CMemNetEncoderUnit &encoder_unit,
                             const tialnn::XBatchLayer<tialnn::IdentityNeurons> &query_layer,
                             tialnn::IndexType bq_start,
                             CMemNetAttentionUnit &attention_unit);

//! projection_unit           -> decoder_unit
//! encoder_unit              -> decoder_unit (through decoderatom.attention_unit)
void CMemNetForwardPropagate(const CMemNetProjectionUnit &projection_unit,
                             const CMemNetEncoderUnit &encoder_unit,
                             CMemNetDecoderUnit &decoder_unit);

//! decoder_unit   -> output_unit
void CMemNetForwardPropagate(const CMemNetDecoderUnit &decoder_unit,
                             CMemNetOutputUnit &output_unit);

//! ==========================
//! Backward propagation.
//! ==========================

//! attention_unit <- output_unit
//! decoder_unit   <- output_unit
void CMemNetBackwardPropagate(CMemNetDecoderUnit &decoder_unit,
                              CMemNetOutputUnit &output_unit);

//! projection_unit     <- decoder_unit
//! encoder_unit        <- decoder_unit (through decoderatom.attention_unit)
void CMemNetBackwardPropagate(CMemNetProjectionUnit &projection_unit,
                              CMemNetEncoderUnit &encoder_unit,
                              CMemNetDecoderUnit &decoder_unit);

//! encoder_unit                <- attention_unit
//! decoderatom.memin_layer     <- attention_unit
//! \param query_layer      decoderatom.memin_layer
void CMemNetBackwardPropagate(CMemNetEncoderUnit &encoder_unit,
                              tialnn::XBatchLayer<tialnn::IdentityNeurons> &query_layer,
                              tialnn::IndexType bq_start,
                              CMemNetAttentionUnit &attention_unit,
                              bool reset);

//! projection_unit     <- encoder_unit
void CMemNetBackwardPropagate(CMemNetProjectionUnit &projection_unit,
                              CMemNetEncoderUnit &encoder_unit);

//! input_unit <- projection_unit
void CMemNetBackwardPropagate(CMemNetInputUnit &input_unit,
                              CMemNetProjectionUnit &projection_unit);

//! input <- input_unit
void CMemNetBackwardPropagate(CMemNetInputUnit &input_unit);

} // namespace cmemnet

#endif
