/*!
 * \file memnet_dparser_base.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_MEMNET_DPARSER_BASE_H_
#define MEMNET_MEMNET_DPARSER_BASE_H_

// std::string
#include <string>
// std::vector
#include <vector>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// tialnn::IndexType
// tialnn::WeightType
#include <tialnn/base.h>
// tialnn::ConnectionBase
#include <tialnn/connection/connection_base.h>
// tialnn::MikolovInitializer
#include <tialnn/initializer/mikolov_initializer.h>
// tialnn::UniformInitializer
#include <tialnn/initializer/uniform_initializer.h>
// tialnn::GaussianInitializer
#include <tialnn/initializer/gaussian_initializer.h>
// TIALNN_ERR
#include <tialnn/util/logging.h>
// memnet::DataPointBatch
#include "datapoint_batch.h"
// memnet::WordVocab
#include "word_vocab.h"
// memnet::GenericVocab
#include "generic_vocab.h"
// memnet::ModelOptions
#include "model_options.h"

namespace memnet {

class MemNetDParserBase {
 public:
  //! Constructor.
  MemNetDParserBase() : debug_(0) {}
  //! Destructor.
  virtual ~MemNetDParserBase() {}

  //! Returns the debug level.
  int debug() const { return debug_; }
  //! Returns the model options.
  const ModelOptions& modelopts() const { return modelopts_; }
  //! Returns the word vocabulary.
  const WordVocab& word_vocab() const { return word_vocab_; }
  //! Returns the lemma vocabulary.
  const WordVocab& lemma_vocab() const { return lemma_vocab_; }
  //! Returns the cpos vocabulary.
  const WordVocab& cpos_vocab() const { return cpos_vocab_; }
  //! Returns the pos vocabulary.
  const WordVocab& pos_vocab() const { return pos_vocab_; }
  //! Returns the feature vocabulary.
  const GenericVocab& feature_vocab() const { return feature_vocab_; }
  //! Returns the depenency vocabulary.
  const GenericVocab& dependency_vocab() const { return dependency_vocab_; }

  //! Sets the debug level.
  void set_debug(int debug) { debug_ = debug; }
  //! Sets the general options related to the model setup.
  void set_modelopts(const ModelOptions &opts) { modelopts_ = opts; }
  
  //! Register connection in the update queue.
  void RegisterConnection(tialnn::ConnectionBase *ptr_conn) {
#ifdef _TIALNN_DEBUG
    for (auto &pc : conn_ptrs_) {
      if (ptr_conn == pc) {
        TIALNN_ERR("doubly registered connection!");
        std::exit(EXIT_FAILURE);
      }
    }
#endif
    conn_ptrs_.push_back(ptr_conn);
  }
  //! Initializes the connections using Mikolov initializer.
  void InitializeWeights(tialnn::MikolovInitializer &initializer, boost::mt19937 &rng_engine) {
    for (auto &ptr_conn : conn_ptrs_) {
      ptr_conn->RandomlyInitialize(initializer, rng_engine);
    }
  }
  //! Initializes the connections using Xarvier initializer.
  //! \note For sigmoid == true, needs to set manually.
  void InitializeWeights(tialnn::UniformInitializer &initializer, boost::mt19937 &rng_engine) {
    for (auto &ptr_conn : conn_ptrs_) {
      initializer.XavierSetScale(ptr_conn, false);
      ptr_conn->RandomlyInitialize(initializer, rng_engine);
    }
  }
  //! Initializes the connections using Gaussian initializer.
  void InitializeWeights(tialnn::GaussianInitializer &initializer, boost::mt19937 &rng_engine) {
    for (auto &ptr_conn : conn_ptrs_) {
      ptr_conn->RandomlyInitialize(initializer, rng_engine);
    }
  }
  //! Updates the connections. 
  void UpdateWeights(float learning_rate) {
    for (auto &ptr_conn : conn_ptrs_) {
      ptr_conn->UpdateWeights(learning_rate);
    }
  }
  //! Caches weightss in current iteration.
  void CacheCurrentWeights() {
    for (auto &ptr_conn : conn_ptrs_) {
      ptr_conn->CacheCurrentWeights();
    }
  }
  //! Restores weights in last iteration.
  void RestoreLastWeights() {
    for (auto &ptr_conn : conn_ptrs_) {
      ptr_conn->RestoreLastWeights();
    }
  }

  //! Reads the vocabularies from txt filenames.
  void ReadVocabsFromTxt(const std::string &word_vocab_filneame,
                         const std::string &lemma_vocab_filneame,
                         const std::string &cpos_vocab_filename,
                         const std::string &pos_vocab_filename,
                         const std::string &feature_vocab_filename,
                         const std::string &dependency_vocab_filename);
  
  //! Checks the parameters.
  bool CheckParams();
  //! Prints parameters.
  void PrintParams();
  //! Reads the model from the file.
  void ReadMemNet(std::ifstream &ifs);
  //! Writes the model to the file.
  void WriteMemNet(std::ofstream &ofs);

  //! Allocates the model.
  virtual void AllocateModel() = 0;
  //! Destroys the model.
  virtual void DestroyModel() = 0;
  //! Initialize the neural network.
  virtual void InitializeNeuralNet() = 0;
  //! Sets the word embeddings.
  virtual void SetWordEmbeddings(tialnn::IndexType widx, const std::vector<tialnn::WeightType> &embd) = 0;
  //! Resets the neural network activations.
  virtual void ResetActivations() = 0;
  //! Set activations for input layer.
  //! Only uses datapoints in ptr_databatch specified by idx_datapoints.
  virtual void SetActivationsForInputLayers(const DataPointBatch *ptr_databatch,
                                            const std::vector<tialnn::IndexType> &idx_datapoints) = 0;
  //! Sets errors for the output layers.
  //! Since it is for sure to iterate over all output layers, it returns the log-probability (base e)
  //! of the datapoint batch with little cost.
  //! Only uses datapoints in ptr_databatch specified by idx_datapoints.
  virtual double SetErrorsForOutputLayers(const DataPointBatch *ptr_databatch,
                                          const std::vector<tialnn::IndexType> &idx_datapoints,
                                          float multitask_weight_arclabel) = 0;
  //! Gets the log-probaility (base e) of the datapoint batch.
  //! Only uses datapoints in ptr_databatch specified by idx_datapoints.
  virtual double GetLogProb(const DataPointBatch *ptr_databatch,
                            const std::vector<tialnn::IndexType> &idx_datapoints,
                            float multitask_weight_arclabel) = 0;
  //! Forward propagates to output layers.
  virtual void ForwardPropagate() = 0;
  //! Backward propagates from output layers.
  virtual void BackwardPropagate() = 0;

  //! Predicts the arcs and labels.
  virtual void Predict(const DataPointBatch *ptr_databatch,
                       const std::vector<tialnn::IndexType> &idx_datapoints,
                       std::vector<std::vector<DataPointToken>> &datapoints) = 0;
  //! Predicts the arcs and lables.
  //! Arcs are predicted by searching for the MST \cite{McDonald2005EMNLP},
  //! using the Trajan's efficient implemnetation of Chu-Liu-Edmonds algorithm.
  virtual void MSTPredict(const DataPointBatch *ptr_databatch,
                          const std::vector<tialnn::IndexType> &idx_datapoints,
                          std::vector<std::vector<DataPointToken>> &datapoints) = 0;

private:

  //! Implementation of CheckParams.
  virtual bool CheckParamsImpl() = 0;
  //! Implementation of PrintParams.
  virtual void PrintParamsImpl() = 0;
  //! Implementation of ReadMemNet.
  virtual void ReadMemNetImpl(std::ifstream &ifs) = 0;
  //! Implemenation of WriteMemNet. 
  virtual void WriteMemNetImpl(std::ofstream &ofs) = 0;

  //! Debug level.
  int debug_;

  //! Options related to the model, which can be changed at test time.
  ModelOptions modelopts_;

  //! Word vocabulary.
  WordVocab word_vocab_;
  //! Lemma vocabulary.
  WordVocab lemma_vocab_;
  //! CPOS vocabulary.
  WordVocab cpos_vocab_;
  //! POS vocabulary.
  WordVocab pos_vocab_;
  //! Feature vocabulary.
  GenericVocab feature_vocab_;
  //! Dependency vocabulary.
  GenericVocab dependency_vocab_;

  //! All connections that need to be updated.
  std::vector<tialnn::ConnectionBase*> conn_ptrs_;
};

} // namespace memnet

#endif
