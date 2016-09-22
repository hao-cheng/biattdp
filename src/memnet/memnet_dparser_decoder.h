/*!
 * \file memnet_dparser_decoder.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_MEMNET_DPARSER_DECODER_H_
#define MEMNET_MEMNET_DPARSER_DECODER_H_

// std::ofstream
#include <fstream>
// std::string
#include <string>
// std::vector
#include <vector>
// tialnn::IndexType
#include <tialnn/base.h>
// memnet::AlgorithmOptions
#include "algorithm_options.h"
// memnet::DataPointBatch
#include "datapoint_batch.h"
// memnet::MemNetDParserBase
#include "memnet_dparser_base.h"

namespace memnet {

class MemNetDParserDecoder {
public:
  //! Constructor.
  MemNetDParserDecoder() : debug_(0), ptr_model_(nullptr) {}
  MemNetDParserDecoder(MemNetDParserBase *pm) : debug_(0), ptr_model_(pm) {}
  //! Destructor.
  virtual ~MemNetDParserDecoder() {}

  //! Returns the debug level.
  int debug() const { return debug_; }

  //! Sets the debug level.
  void set_debug(int debug) { debug_ = debug; }
  
  //! Checks the parameters.
  void CheckParams();
  //! Prints parameters.
  void PrintParams();

  //! Reads the model from the file.
  void ReadMemNet(const std::string &infile);

  //! Predicts labels using the model.
  void Predict(const std::string &infile, const std::string &outfile, bool mst);

private:
  //! Writes the data batch to ofstream in CoNLL-X format.
  void WriteDataPointBatchToCoNLLX(const std::vector<std::vector<DataPointToken>> &datapoints,
                                   std::ofstream &ofs);

  //! Debug level.
  int debug_;

  //! Options related to the training algorithm.
  AlgorithmOptions algopts_;

  //! Pointer to the model.
  MemNetDParserBase *const ptr_model_;
};

} // namespace memnet

#endif
