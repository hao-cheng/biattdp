/*!
 * \file cmemnet_dparser.h
 *
 * 
 * \version 0.0.6
 */

#ifndef CMEMNET_CMEMNET_DPARSER_BASE_H_
#define CMEMNET_CMEMNET_DPARSER_BASE_H_

// std::ifstream
// std::ofstream
#include <fstream>
// std::vector
#include <vector>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::IndexType
#include <tialnn/base.h>
// tialnn::XBatchLayer
#include <tialnn/layer/xbatchlayer.h>
// tialnn::IdentityNeurons
#include <tialnn/neuron/identity_neurons.h>
// tialnn::GatedRecurrentXUnit
#include <tialnn/unit/gated_recurrent_xunit.h>
// memnet::MemNetDParserBase
#include "../memnet/memnet_dparser_base.h"
// memnet::DataPointBatch
#include "../memnet/datapoint_batch.h"
// cmemnet:CMemNetInputUnit
#include "cmemnet_input_unit.h"
// cmemnet:CMemNetProjectionUnit
#include "cmemnet_projection_unit.h"
// cmemnet:CMemNetEncoderUnit
#include "cmemnet_encoder_unit.h"
// cmemnet:CMemNetDecoderUnit
#include "cmemnet_decoder_unit.h"
// cmemnet:CMemNetOutputUnit
#include "cmemnet_output_unit.h"

namespace cmemnet {

//! Bi-directional encoder + bi-directional decoder
class CMemNetDParser : public memnet::MemNetDParserBase {
 public:
  //! Constructor.
  explicit CMemNetDParser() : num_composition_neurons_(0), num_projection_neurons_(0), num_hidden_neurons_(0) {}
  //! Destructor.
  ~CMemNetDParser() {}

  //! Sets the number of neurons for the composition layer.
  void set_num_composition_neurons(tialnn::IndexType n) { num_composition_neurons_ = n; }
  //! Sets the number of neurons for the projection layer.
  void set_num_projection_neurons(tialnn::IndexType n) { num_projection_neurons_ = n; }
  //! Sets the number of hidden neurons.
  void set_num_hidden_neurons(tialnn::IndexType n) { num_hidden_neurons_ = n; }
  
  //! Allocates the model.
  virtual void AllocateModel() override;
  //! Destroys the model.
  virtual void DestroyModel() override;
  //! Initialize the neural network.
  virtual void InitializeNeuralNet() override;
  //! Sets the word embeddings.
  virtual void SetWordEmbeddings(tialnn::IndexType widx, const std::vector<tialnn::WeightType> &embd) override;
  //! Resets the neural network activations.
  virtual void ResetActivations() override;
  //! Set activations for input layer.
  //! Only uses datapoints in ptr_databatch specified by idx_datapoints.
  virtual void SetActivationsForInputLayers(const memnet::DataPointBatch *ptr_databatch,
                                            const std::vector<tialnn::IndexType> &index_datapoints) override;
  //! Sets errors for the output layers.
  //! Since it is for sure to iterate over all output layers, it returns the log-probability (base e)
  //! of the datapoint batch with little cost.
  //! Only uses datapoints in ptr_databatch specified by idx_datapoints.
  virtual double SetErrorsForOutputLayers(const memnet::DataPointBatch *ptr_databatch,
                                          const std::vector<tialnn::IndexType> &index_datapoints,
                                          float multitask_weight_arclabel) override;
  //! Gets the log-probaility (base e) of the datapoint batch.
  //! Only uses datapoints in ptr_databatch specified by idx_atapoints.
  virtual double GetLogProb(const memnet::DataPointBatch *ptr_databatch,
                            const std::vector<tialnn::IndexType> &index_datapoints,
                            float multitask_weight_arclabel) override;
  //! Forward propagates to output layers.
  virtual void ForwardPropagate() override;
  //! Backward propagates from output layers.
  virtual void BackwardPropagate() override;

  //! Predicts the arcs and labels.
  virtual void Predict(const memnet::DataPointBatch *ptr_databatch,
                       const std::vector<tialnn::IndexType> &idx_datapoints,
                       std::vector<std::vector<memnet::DataPointToken>> &datapoints) override;
  //! Predicts the arcs and lables.
  //! Arcs are predicted by searching for the MST \cite{McDonald2005EMNLP},
  //! using the Trajan's efficient implemnetation of Chu-Liu-Edmonds algorithm.
  virtual void MSTPredict(const memnet::DataPointBatch *ptr_databatch,
                          const std::vector<tialnn::IndexType> &idx_datapoints,
                          std::vector<std::vector<memnet::DataPointToken>> &datapoints) override;

 private:
   //! Implementation of CheckParams.
   virtual bool CheckParamsImpl() override;
   //! Implementation of PrintParams.
   virtual void PrintParamsImpl() override;
   //! Implementation of ReadMemNet.
   virtual void ReadMemNetImpl(std::ifstream &ifs) override;
   //! Implemenation of WriteMemNet. 
   virtual void WriteMemNetImpl(std::ofstream &ofs) override;

  //! Number of hidden neurons in composition layer.
  tialnn::IndexType num_composition_neurons_;
  //! Number of hidden neurons in projection layer.
  tialnn::IndexType num_projection_neurons_;
  //! Number of hidden neurons in encoder and decoder.
  tialnn::IndexType num_hidden_neurons_;

  //! Input unit.
  CMemNetInputUnit input_unit_;
  //! Projection unit.
  CMemNetProjectionUnit projection_unit_;
  //! Encoder unit.
  CMemNetEncoderUnit encoder_unit_;
  //! Decoder unit.
  CMemNetDecoderUnit decoder_unit_;
  //! Output unit.
  CMemNetOutputUnit output_unit_;

  //! Bias layer.
  tialnn::XBatchLayer<tialnn::IdentityNeurons> bias_layer_;

  //! Auxilary layer of all -1s for encoder unit and decoder unit.
  tialnn::XBatchLayer<tialnn::IdentityNeurons> negones_layer_;

  //! Intial hidden/memout layer for encoder unit and decoder unit.
  tialnn::XBatchLayer<tialnn::IdentityNeurons> init_hidden_layer_;

  //! Random engine.
  boost::mt19937 rng_engine_;
};

} // namespace memnet

#endif
