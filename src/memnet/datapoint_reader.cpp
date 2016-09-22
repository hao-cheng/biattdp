/*!
 * \file datapoint_reader.cpp
 *
 * 
 * \version 0.0.6
 */

#include "datapoint_reader.h"

// std::exit
#include <cstdlib>
// std::ceil
#include <cmath>
// std::cerr
// std::endl
#include <iostream>
// std::ifstream
#include <fstream>
// std::ios
#include <ios>
// std::string
// std::getline
// std::stoull
#include <string>
// std::vector
#include <vector>
// std::unordered_map
#include <unordered_map>
// std::random_shuffle
#include <algorithm>
// boost::variate_generator
#include <boost/random/variate_generator.hpp>
// boost::mt19937
#include <boost/random/mersenne_twister.hpp>
// boost::uniform_int
#include <boost/random/uniform_int.hpp>
// boost::tokenizer
// boost::char_separator
#include <boost/tokenizer.hpp>
// tialnn::IndexType
#include <tialnn/base.h>
// memnet::DataPointToken
#include "datapoint_token.h"

using std::exit;
using std::ceil;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ios;
using std::string;
using std::getline;
using std::stoull;
using std::vector;
using std::unordered_map;
using std::random_shuffle;
using boost::variate_generator;
using boost::mt19937;
using boost::uniform_int;
using boost::tokenizer;
using boost::char_separator;
using tialnn::IndexType;

namespace memnet {

void DataPointReader::StartEpoch() {
  assert(max_sequence_length_ > 0);
  assert(batch_size_ > 0);

  if (filenames_.empty()) {
    cerr << "Filenames_ is empty!" << endl;
    exit(EXIT_FAILURE);
  }

  if (filenames_.size() == 1 && !databatches_.empty()) {
    // Avoids unnecessary tokenization.
    assert(idx_remaining_databatches_.empty());
    UpdateIdxRemaining();

    assert(idx_remaining_databatches_.back() < static_cast<IndexType>(databatches_.size()));
    ptr_current_databatch_ = &databatches_[idx_remaining_databatches_.back()];
    assert(idx_remaining_databatches_.back() < static_cast<IndexType>(idx_remaining_datapoints_for_databatch_.size()));
    ptr_current_idx_remaining_datapoints_ = &idx_remaining_datapoints_for_databatch_[idx_remaining_databatches_.back()];
    assert(!ptr_current_idx_remaining_datapoints_->empty());
  } else {
    if (shuffle_datafiles_) {
      variate_generator<mt19937&, uniform_int<long long>> rng(rng_engine_, rng_uni_dist_);
      random_shuffle(filenames_.begin(), filenames_.end(), rng);
    }
    it_next_filename_ = filenames_.begin();
    if (!LoadNextDataFile()) {
      cerr << "ERROR: empty filenames_ ! (should not reach here)" << endl;
      exit(EXIT_FAILURE);
    }
  }
}

const DataPointBatch* DataPointReader::GetDataPointBatchPtr(vector<IndexType> &idx_datapoints) {
  idx_datapoints.clear();

  if (idx_remaining_databatches_.empty()) {
    if (!LoadNextDataFile()) {
      return nullptr;
    }
  } else {
    assert(idx_remaining_databatches_.back() < static_cast<IndexType>(databatches_.size()));
    ptr_current_databatch_ = &databatches_[idx_remaining_databatches_.back()];
    assert(idx_remaining_databatches_.back() < static_cast<IndexType>(idx_remaining_datapoints_for_databatch_.size()));
    ptr_current_idx_remaining_datapoints_ = &idx_remaining_datapoints_for_databatch_[idx_remaining_databatches_.back()];
  }
  idx_remaining_databatches_.pop_back();

  assert(!ptr_current_idx_remaining_datapoints_->empty());
  for (IndexType i = 0; i < batch_size_; i++) {
    idx_datapoints.push_back(ptr_current_idx_remaining_datapoints_->back());
    ptr_current_idx_remaining_datapoints_->pop_back();
    if (ptr_current_idx_remaining_datapoints_->empty()) {
      break;
    }
  }

  assert(!idx_datapoints.empty());
  assert(ptr_current_databatch_ != nullptr);

  return ptr_current_databatch_;
}

//! Parse the token from a line in CoNLL-X format.
void DataPointReader::GetDataPointTokenFromCoNLLXLine(const string &line, DataPointToken &token) {
  char_separator<char> tab_separator("\t");
  tokenizer<char_separator<char>> units(line, tab_separator);
  tokenizer<char_separator<char>>::const_iterator it = units.begin();

  // ID
  token.token_position = static_cast<IndexType>(stoull(*it));
  // FORM
  token.form = ptr_word_vocab_->idx4type(*(++it));
  // LEMMA
  ++it;
  if (ptr_lemma_vocab_) {
    token.lemma = ptr_lemma_vocab_->idx4type(*it);
  }
  // CPOSTAG
  ++it;
  if (ptr_cpos_vocab_) {
    token.cpos_tag = ptr_cpos_vocab_->idx4type(*it);
  }
  // POSTAG
  ++it;
  if (ptr_pos_vocab_) {
    token.pos_tag = ptr_pos_vocab_->idx4type(*it);
  }
  // FEATS
  ++it;
  if (ptr_feature_vocab_) {
    char_separator<char> bar_separator("|");
    tokenizer<char_separator<char>> feats(*it, bar_separator);
    token.feats.clear();
    for (auto fit = feats.begin(); fit != feats.end();  ++fit) {
      token.feats.push_back(ptr_feature_vocab_->idx4type(*fit));
    }
  }
  // HEAD
  token.headword_position = static_cast<IndexType>(stoull(*(++it)));
  // DEPREL
  token.dependency = static_cast<IndexType>(ptr_dependency_vocab_->idx4type(*(++it)));
  // PHEAD
  ++it;
  // PDEPREL
  ++it;
  assert(++it == units.end());
}

bool DataPointReader::LoadNextDataFile() {
  if (it_next_filename_ == filenames_.end()) {
    return false;
  }

  // Remember to close the stream.
  ifstream ifs(*it_next_filename_, ios::in);
  if (ifs.fail()) {
    cerr << "Unable to open " << *it_next_filename_ << endl;
    exit(EXIT_FAILURE);
  }

  ptr_current_databatch_ = nullptr;
  databatches_.clear();

  unordered_map<IndexType, IndexType> batchidx_for_length;
  char_separator<char> tab_separator("\t");
  DataPointToken token;
  string line;
  vector<DataPointToken> dp;
  //! \note insert root, the headword and relation are not used.
  dp.push_back(DataPointToken(0, ptr_word_vocab_->root_idx(), 0, 0));
  auto it_root = dp.begin();
  if (ptr_lemma_vocab_) {
    it_root->lemma = ptr_lemma_vocab_->root_idx();
  }
  if (ptr_cpos_vocab_) {
    it_root->cpos_tag = ptr_cpos_vocab_->root_idx();
  }
  if (ptr_pos_vocab_) {
    it_root->pos_tag = ptr_pos_vocab_->root_idx();
  }
  if (ptr_feature_vocab_) {
    it_root->feats.push_back(ptr_feature_vocab_->idx4type("_"));
  }
  while (getline(ifs, line)) {
    if (line.empty()) {
      if (static_cast<IndexType>(dp.size()) > max_sequence_length_) {
        //! \note max_sequence_length_ includes the first <root>.
        cerr << "Sequence length exceeds " << max_sequence_length_;
        cerr << " in " << *it_next_filename_ << endl;
        exit(EXIT_FAILURE);
      }

      IndexType len = static_cast<IndexType>(dp.size());
      if (len == 1) {
        cerr << "Empty sequence in " << *it_next_filename_ << endl;
        exit(EXIT_FAILURE);
      }

      auto it = batchidx_for_length.find(len);
      if (it != batchidx_for_length.end()) {
        databatches_[it->second].AddDataPoint(dp);
      } else {
        batchidx_for_length[len] = static_cast<IndexType>(databatches_.size());
        databatches_.push_back(DataPointBatch(len));
        auto rit = databatches_.rbegin();
        rit->AddDataPoint(dp);
      }

      dp.clear();
      //! \note insert root, the headword and relation are not used.
      dp.push_back(DataPointToken(0, ptr_word_vocab_->root_idx(), 0, 0));
      it_root = dp.begin();
      if (ptr_lemma_vocab_) {
        it_root->lemma = ptr_lemma_vocab_->root_idx();
      }
      if (ptr_cpos_vocab_) {
        it_root->cpos_tag = ptr_cpos_vocab_->root_idx();
      }
      if (ptr_pos_vocab_) {
        it_root->pos_tag = ptr_pos_vocab_->root_idx();
      }
      if (ptr_feature_vocab_) {
        it_root->feats.push_back(ptr_feature_vocab_->idx4type("_"));
      }
    } else {
      GetDataPointTokenFromCoNLLXLine(line, token);
      dp.push_back(token);
    }
  }
  if (dp.size() > 1) {
    cerr << "Missing last empty line in " << *it_next_filename_ << endl;
    exit(EXIT_FAILURE);
  }

  // Remember to close the stream.
  ifs.close();
  ++it_next_filename_;

  UpdateIdxRemaining();

  assert(idx_remaining_databatches_.back() < static_cast<IndexType>(databatches_.size()));
  ptr_current_databatch_ = &databatches_[idx_remaining_databatches_.back()];
  assert(idx_remaining_databatches_.back() < static_cast<IndexType>(idx_remaining_datapoints_for_databatch_.size()));
  ptr_current_idx_remaining_datapoints_ = &idx_remaining_datapoints_for_databatch_[idx_remaining_databatches_.back()];
  assert(!ptr_current_idx_remaining_datapoints_->empty());

  return true;
}

void DataPointReader::UpdateIdxRemaining() {
  vector<IndexType> idx_datapoints;
  //! Ensure begin() and back() are valid.
  //! Pre-compute for all databatches.
  for (IndexType t = 0; t < batch_size_; t++) {
    idx_datapoints.push_back(t);
  }

  idx_remaining_databatches_.clear();
  idx_remaining_datapoints_for_databatch_.clear();
  IndexType i = 0;
  for (auto &databatch : databatches_) {
    const IndexType num_dps = databatch.num_datapoints();
    IndexType num_occurrences = static_cast<IndexType>(ceil(static_cast<float>(num_dps) / batch_size_));
    //! \note Insert n occurrences of current data batch.
    for (IndexType t = 0; t < num_occurrences; t++) {
      idx_remaining_databatches_.push_back(i);
    }

    for (IndexType t = idx_datapoints.back() + 1; t < num_dps; t++) {
      idx_datapoints.push_back(t);
    }

    auto it_idx_datapoints = idx_datapoints.begin(); //! Since idx_datapoints has been changed, the iterator should be re-assigned.
    idx_remaining_datapoints_for_databatch_.push_back(vector<IndexType>(it_idx_datapoints, 
                                                                        it_idx_datapoints + num_dps));
    i++;
  }
  if (shuffle_datapoints_) {
    variate_generator<mt19937&, uniform_int<long long>> rng(rng_engine_, rng_uni_dist_);
    random_shuffle(idx_remaining_databatches_.begin(), idx_remaining_databatches_.end(), rng);
    for (auto &idx_remaining_datapoints : idx_remaining_datapoints_for_databatch_) {
      random_shuffle(idx_remaining_datapoints.begin(), idx_remaining_datapoints.end(), rng);
    }
  }
}

} // namespace memnet
