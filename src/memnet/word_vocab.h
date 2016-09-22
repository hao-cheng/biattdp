/*!
 * \file word_vocab.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_WORD_VOCAB_H_
#define MEMNET_WORD_VOCAB_H_

// assert
#include <cassert>
// std::exit
#include <cstdlib>
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

//! Word vocabulary.
class WordVocab {
 public:
  //! Constructor.
  WordVocab() : size_(0), unk_idx_(0), root_idx_(0), blank_idx_(0) {}
  //! Destructor.
  ~WordVocab() {}

  //! Returns whether vocabulary is empty.
  bool empty() const { return size_ == 0; }
  //! Returns number of types.
  tialnn::IndexType size() const { return size_; }
  //! Returns the index for type s.
  //! If it is an OOV, returns unk_idx_.
  tialnn::IndexType idx4type(const std::string &s) const {
    assert(size_ > 0);
    std::unordered_map<std::string, tialnn::IndexType>::const_iterator mi = idx4type_.find(s);
    if (mi == idx4type_.end()) {
      return unk_idx_;
    } else {
      return mi->second;
    }
  }
  //! Returns the type for index i.
  const std::string& type4idx(const tialnn::IndexType i) const {
    assert(size_ > 0);
    assert(i < size_);
    assert(i < static_cast<tialnn::IndexType>(type4idx_.size()));
    return type4idx_[i];
  }
  //! Returns the unk_idx_.
  tialnn::IndexType unk_idx() const { return unk_idx_; }
  //! Returns the root_idx_.
  tialnn::IndexType root_idx() const { return root_idx_; }
  //! Returns the blank_idx_.
  tialnn::IndexType blank_idx() const { return blank_idx_; }

  //! Reads the vocabulary from txt file.
  //! Each line is a type.
  void ReadVocabFromTxt(const std::string &vocabtxt);

  //! Reads the vocabulary from the binary ifstream.
  void ReadVocab(std::ifstream &ifs);
  //! Writes the vocabulary into the binary ofstream.
  void WriteVocab(std::ofstream &ofs);

 private:
  //! Number of types in the vocabulary.
  //! Including <root> and <unk>.
  tialnn::IndexType size_;
  
  //! Index for type.
  std::unordered_map<std::string, tialnn::IndexType> idx4type_;
  //! Word for index.
  std::vector<std::string> type4idx_;
  //! Index of <unk>.
  tialnn::IndexType unk_idx_;
  //! Index of <root>. 
  tialnn::IndexType root_idx_;
  //! Index of <blank>.
  //! \note <blank> is used for filling a data batch.
  //!       The corresponding label for it is invalid.
  tialnn::IndexType blank_idx_;
};

} // namespace memnet

#endif
