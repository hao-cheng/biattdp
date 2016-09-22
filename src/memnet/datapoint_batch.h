/*!
 * \file datapoint_batch.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_DATAPOINT_BATCH_H_
#define MEMNET_DATAPOINT_BATCH_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::cerr
// std::endl
#include <iostream>
// std::vector
#include <vector>
// std::fill
// std::copy
#include <algorithm>
// tialnn::IndexType
#include <tialnn/base.h>
// memnet::DataPointToken
#include "datapoint_token.h"

namespace memnet {

class DataPointBatch {
 public:
  //! Constructors.
  explicit DataPointBatch() : num_datapoints_(0), sequence_length_(0) {}
  explicit DataPointBatch(tialnn::IndexType len) : num_datapoints_(0), sequence_length_(len) {}
  explicit DataPointBatch(const DataPointBatch &dpb) : 
    num_datapoints_(dpb.num_datapoints()),
    sequence_length_(dpb.sequence_length()),
    data_(dpb.data()) {}
  //! Destructor.
  ~DataPointBatch() {}

  //! Returns the number of datapoints in the batch.
  tialnn::IndexType num_datapoints() const { return num_datapoints_; }
  //! Returns the sequence legnth of datapoints in the batch.
  tialnn::IndexType sequence_length() const { return sequence_length_; }
  //! Returns the data.
  const std::vector<DataPointToken>& data() const { return data_; }
  //! Returns the data token of the idx-th datapoint at timestep t.
  const DataPointToken& data(tialnn::IndexType idx,
                             tialnn::IndexType t) const {
    assert(idx < num_datapoints_);
    assert(t < sequence_length_);
    assert(idx * sequence_length_ + t < static_cast<tialnn::IndexType>(data_.size()));
    return data_[idx * sequence_length_ + t];
  }

  //! Sets the data of the idx-th datapoint at timestep t.
  void set_data(tialnn::IndexType idx, tialnn::IndexType t,
                const DataPointToken &tok) {
    assert(idx < num_datapoints_);
    assert(t < sequence_length_);
    assert(idx * sequence_length_ + t < static_cast<tialnn::IndexType>(data_.size()));
    data_[idx * sequence_length_ + t] = tok;
  }

  //! Clears the batch.
  void Clear() {
    num_datapoints_ = 0;
    data_.clear();
  }

  //! Adds a datapoint to the batch.
  void AddDataPoint(const std::vector<DataPointToken> &dp) {
    if (static_cast<tialnn::IndexType>(dp.size()) != sequence_length_) {
      std::cerr << "Sequence length mismatch!" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    for (const DataPointToken &tok : dp) {
      data_.push_back(tok);
    }
    num_datapoints_++;
  }

 private:
  //! Number of datapoints in the batch.
  tialnn::IndexType num_datapoints_;
  //! Sequence length of datapoints.
  //! It is un-modifiable.
  const tialnn::IndexType sequence_length_;

  //! All data points in the batch share the same sequence length.
  std::vector<DataPointToken> data_;
};

} // namespace memnet

#endif
