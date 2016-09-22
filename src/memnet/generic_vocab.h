/*!
 * \file generic_vocab.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_GENERIC_VOCAB_H_
#define MEMNET_GENERIC_VOCAB_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::cerr
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// std::string
#include <string>
// std::vector
#include <vector>
// std::unordered_map
#include <unordered_map>
// neuarlnet::IndexType
#include <tialnn/base.h>

namespace memnet {

// Generic vocabulary.
class GenericVocab {
 public:
  // Constructor.
  GenericVocab() : size_(0) {}
  // Destructor.
  ~GenericVocab() {}

  // Returns whether vocabulary is empty.
  bool empty() const { return size_ == 0; }
  // Returns number of types.
  tialnn::IndexType size() const { return size_; }
  // Returns the index for type s.
  tialnn::IndexType idx4type(const std::string &s) const {
    assert(size_ > 0);
    std::unordered_map<std::string, tialnn::IndexType>::const_iterator mi = idx4type_.find(s);
    if (mi == idx4type_.end()) {
      std::cerr << "Out-of-vocabulary type:" << s << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      return mi->second;
    }
  }
  // Returns the type for index i.
  const std::string& type4idx(const tialnn::IndexType i) const {
    assert(size_ > 0);
    assert(i < size_);
    assert(i < static_cast<tialnn::IndexType>(type4idx_.size()));
    return type4idx_[i];
  }

  // Reads the vocabulary from txt file.
  // Each line is a type.
  void ReadVocabFromTxt(const std::string &vocabtxt);

  // Reads the vocabulary from the binary ifstream.
  void ReadVocab(std::ifstream &ifs);
  // Writes the vocabulary into the binary ofstream.
  void WriteVocab(std::ofstream &ofs);

 private:
  // Number of types in the vocabulary.
  tialnn::IndexType size_;
  
  // Index for type.
  std::unordered_map<std::string, tialnn::IndexType> idx4type_;
  // Generic for index.
  std::vector<std::string> type4idx_;
};

} // namespace memnet

#endif
