/*!
 * \file memnet_dparser_trainer.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_MEMNET_DPARSER_TRAINER_H_
#define MEMNET_MEMNET_DPARSER_TRAINER_H_

// std::string
#include <string>
// memnet::DataPointReader
#include "datapoint_reader.h"
// memnet::AlgorithmOptions
#include "algorithm_options.h"
// memnet::MemNetDParserBase
#include "memnet_dparser_base.h"

namespace memnet {

class MemNetDParserTrainer {
public:
  //! Constructor.
  MemNetDParserTrainer() : debug_(0), ptr_model_(nullptr) {}
  MemNetDParserTrainer(MemNetDParserBase *pm) : debug_(0), ptr_model_(pm) {}
  //! Destructor.
  virtual ~MemNetDParserTrainer() {}

  //! Returns the debug level.
  int debug() const { return debug_; }
  //! Returns the algorithm options.
  const AlgorithmOptions& algopts() { return algopts_; }

  //! Sets the debug level.
  void set_debug(int debug) { debug_ = debug; }
  //! Sets the general options related to the training algorithm.
  void set_algopts(const AlgorithmOptions &opts) { algopts_ = opts; }
  
  //! Checks the parameters.
  void CheckParams();
  //! Prints parameters.
  void PrintParams();

  //! Trains the model.
  void TrainMemNet(const std::string &outbase);
  //! Evaluates the model on the file.
  void EvalMemNet(const std::string &infile);

  //! Reads the model from the file.
  void ReadMemNet(const std::string &infile);
  //! Writes the model to the file.
  void WriteMemNet(const std::string &outbase);

  //! Reads initial word embeddings from the file.
  void ReadWordEmbeddingsFromTxt(const std::string &infile);

private:

  //! Train the model using stochastic gradient descent.
  void SGDTrain(DataPointReader &train_data, DataPointReader &validation_data, const std::string &outbase);
  //! Evaluates the model on the DataPointReader object.
  //! Returns the log-likelihood of the data.
  double EvalMemNet(DataPointReader &data);

  //! Debug level.
  int debug_;

  //! Options related to the training algorithm.
  AlgorithmOptions algopts_;

  //! Pointer to the model.
  MemNetDParserBase *const ptr_model_;

};

} // namespace memnet

#endif
