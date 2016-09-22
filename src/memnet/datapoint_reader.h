/*!
 * \file datapoint_reader.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_DATAPOINT_READER_H_
#define MEMNET_DATAPOINT_READER_H_

// std::string
#include <string>
// std::vector
#include <vector>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// boost::uniform_int
#include <boost/random/uniform_int.hpp>
// tialnn::IndexType
#include <tialnn/base.h>
// memnet::WordVocab
#include "word_vocab.h"
// memnet::GenericVocab
#include "generic_vocab.h"
// memnet:DataPointToken
#include "datapoint_token.h"
// memnet:DataPointBatch
#include "datapoint_batch.h"

namespace memnet {

//!  DataPoint reader for MemNet.
class DataPointReader {
 public:
  //! Constructors.
  DataPointReader() : ptr_word_vocab_(nullptr), ptr_lemma_vocab_(nullptr), 
    ptr_cpos_vocab_(nullptr), ptr_pos_vocab_(nullptr),
    ptr_feature_vocab_(nullptr), ptr_dependency_vocab_(nullptr),
    shuffle_datafiles_(false), shuffle_datapoints_(false), 
    max_sequence_length_(0), batch_size_(0),
    filenames_(), it_next_filename_(), databatches_(), idx_remaining_databatches_(),
    ptr_current_databatch_(nullptr), ptr_current_idx_remaining_datapoints_(nullptr),
    rng_engine_(), rng_uni_dist_() {}
  DataPointReader(const std::vector<std::string> &fns, 
                  const WordVocab *pwv, const WordVocab *plv,
                  const WordVocab *pcpv, const WordVocab *ppv,
                  const GenericVocab *pfeatv, const GenericVocab *pdv,
                  bool sdata, bool ssent,
                  tialnn::IndexType len, tialnn::IndexType bs) :
      ptr_word_vocab_(pwv), ptr_lemma_vocab_(plv),
      ptr_cpos_vocab_(pcpv), ptr_pos_vocab_(ppv),
      ptr_feature_vocab_(pfeatv), ptr_dependency_vocab_(pdv),
      shuffle_datafiles_(sdata), shuffle_datapoints_(ssent),
      max_sequence_length_(len), batch_size_(bs),
      filenames_(fns), it_next_filename_(), databatches_(), idx_remaining_databatches_(),
      ptr_current_databatch_(nullptr), ptr_current_idx_remaining_datapoints_(nullptr),
      rng_engine_(), rng_uni_dist_() {}
  //! Destructor.
  ~DataPointReader() {}

  //! Initializes data reader.
  void StartEpoch();
  //! Returns the pointers for current databatch,
  //! and sets the indices of at most batch_size remaining datapoints in current databatch.
  //! In this case, ensure the all datapoint have the same length.
  //! Returns to nullptr if reaches the end of the file list.
  const DataPointBatch* GetDataPointBatchPtr(std::vector<tialnn::IndexType> &idx_datapoints);

 private:
  //! Loads next data file, and assigns ptr_current_databatch_.
  //! If shuffle_datapoints_ is True, also shuffles idx_remaining_databatches_.
  //! If reaches end of the file list, return false.
  bool LoadNextDataFile();
  //! Updates idx_remaining_databaches_ and idx_remaining_datapoints_for_databatch_.
  void UpdateIdxRemaining();
  //! Get a DataPointToken from the line in CoNLL-X format.
  void GetDataPointTokenFromCoNLLXLine(const std::string &line, DataPointToken &token);

  //! Pointer to the word vocabulary.
  const WordVocab *ptr_word_vocab_;
  //! Pointer to the lemma vocabulary.
  const WordVocab *ptr_lemma_vocab_;
  //! Pointer to the CPOS vocabulary.
  const WordVocab *ptr_cpos_vocab_;
  //! Pointer to the POS vocabulary.
  const WordVocab *ptr_pos_vocab_;
  //! Pointer to the feature vocabulary.
  const GenericVocab *ptr_feature_vocab_;
  //! Pointer to the dependency vocabulary.
  const GenericVocab *ptr_dependency_vocab_;

  //! If true, data files will be shuffled.
  bool shuffle_datafiles_;
  //! If true, datapoints within the file will be shuffled.
  bool shuffle_datapoints_;
  //! Maximum allowable sequence length.
  //! \note It includes the first <root>.
  tialnn::IndexType max_sequence_length_;
  //! Target batch size.
  tialnn::IndexType batch_size_;

  //! Filenames of the data files.
  //! Each file is a batch.
  std::vector<std::string> filenames_;
  //! Iterator pointing to next data file.
  std::vector<std::string>::const_iterator it_next_filename_;

  //! All datapoints in the data files.
  std::vector<DataPointBatch> databatches_;
  //! Index of remaining datapoints in current data file.
  std::vector<tialnn::IndexType> idx_remaining_databatches_;
  //! Pointer to the current databatch.
  const DataPointBatch *ptr_current_databatch_;
  //! Index of remaining datapoints for each databatch in current data file.
  std::vector<std::vector<tialnn::IndexType>> idx_remaining_datapoints_for_databatch_;
  //! Pointer to the current idx_remaining_datapoints.
  std::vector<tialnn::IndexType> *ptr_current_idx_remaining_datapoints_;

  //! Random number generator related parameters.
  boost::mt19937 rng_engine_;
  boost::uniform_int<long long> rng_uni_dist_;
};

} // namespace memnet

#endif
