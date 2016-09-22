/*!
 * \file word_vocab.cpp
 *
 * 
 * \version 0.0.6
 */

#include "word_vocab.h"

// std::exit
#include <cstdlib>
// std::cout
// std::cerr
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// std::ios
#include <ios>
// std::string
// std::getline
#include <string>
// std::unordered_map
#include <unordered_map>
// tialnn::write_xxx
// tialnn::read_xxx
#include <tialnn/util/futil.h>
// tialnn::IndexType
#include <tialnn/base.h>

using std::exit;
using std::cout;
using std::cerr;
using std::endl;
using std::ios;
using std::ifstream;
using std::ofstream;
using std::string;
using std::getline;
using std::unordered_map;

namespace memnet {

void WordVocab::ReadVocabFromTxt(const string &vocabtxt) {
  ifstream ifs;
  ifs.open(vocabtxt, ios::in);
  if (ifs.fail()) {
    cerr << "Unable to open " << vocabtxt << endl;
    exit(EXIT_FAILURE);
  }

  string line;
  tialnn::IndexType widx = 0;
  while (getline(ifs, line)) {
    if (idx4type_.find(line) != idx4type_.end()) {
      cerr << "double defined type " << line << " in the vocabulary!" << endl;
      exit(EXIT_FAILURE);
    }
    idx4type_[line] = widx;
    type4idx_.push_back(line);
    widx++;
  }
  ifs.close();

  if (idx4type_.find("<unk>") == idx4type_.end()) {
    cerr << "OOV <unk> is not defined in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  unk_idx_ = idx4type_["<unk>"];
  if (idx4type_.find("<root>") == idx4type_.end()) {
    cerr << "<root> is not defined in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  root_idx_ = idx4type_["<root>"];
  if (idx4type_.find("<blank>") == idx4type_.end()) {
    cerr << "<blank> is not defined in the vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  blank_idx_ = idx4type_["<blank>"];

  size_ = static_cast<tialnn::IndexType>(idx4type_.size());
}

void WordVocab::ReadVocab(ifstream &ifs) {
  cout << "***reading vocab***" << endl;
  tialnn::read_single(ifs, size_);
  cout << "size_: " << size_ << endl;
  tialnn::read_string2T_unordered_map(ifs, idx4type_);
  tialnn::read_1d_string(ifs, type4idx_);

  unordered_map<string, tialnn::IndexType>::iterator mi;

  if (size_ == 0) {
    //! \note unk_idx_, root_idx_, and blank_idx_ are invalid.
    cerr << "Warning: Empty vocab!" << endl;
  } else {
    mi = idx4type_.find("<unk>");
    if (mi == idx4type_.end()) {
      cerr << "Internal error: <unk> not in the vocabulary!" << endl;
      exit(EXIT_FAILURE);
    } else {
      unk_idx_ = mi->second;
    }
    cout << "unk_idx_: " << unk_idx_ << endl;

    mi = idx4type_.find("<root>");
    if (mi == idx4type_.end()) {
      cerr << "Internal error: <root> not in the vocabulary!" << endl;
      exit(EXIT_FAILURE);
    } else {
      root_idx_ = mi->second;
    }
    cout << "root_idx_: " << root_idx_ << endl;

    mi = idx4type_.find("<blank>");
    if (mi == idx4type_.end()) {
      cerr << "Internal error: <blank> not in the vocabulary!" << endl;
      exit(EXIT_FAILURE);
    } else {
      blank_idx_ = mi->second;
    }
    cout << "blank_idx_: " << blank_idx_ << endl;
  }
}

void WordVocab::WriteVocab(ofstream &ofs) {
  tialnn::write_single(ofs, size_);
  tialnn::write_string2T_unordered_map(ofs, idx4type_);
  tialnn::write_1d_string(ofs, type4idx_);
}

} // namespace memnet
