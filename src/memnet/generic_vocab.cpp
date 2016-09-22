/*!
 * \file generic_vocab.cpp
 *
 * 
 * \version 0.0.6
 */

#include "generic_vocab.h"

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
// tialnn::IndexType
#include <tialnn/base.h>
// tialnn::write_xxx
// tialnn::read_xxx
#include <tialnn/util/futil.h>

using std::exit;
using std::cout;
using std::cerr;
using std::endl;
using std::ios;
using std::ifstream;
using std::ofstream;
using std::string;
using std::getline;

namespace memnet {

void GenericVocab::ReadVocabFromTxt(const string &vocabtxt) {
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

  size_ = static_cast<tialnn::IndexType>(idx4type_.size());
}

void GenericVocab::ReadVocab(ifstream &ifs) {
  cout << "***reading vocab***" << endl;
  tialnn::read_single(ifs, size_);
  cout << "size_: " << size_ << endl;
  tialnn::read_string2T_unordered_map(ifs, idx4type_);
  tialnn::read_1d_string(ifs, type4idx_);
}

void GenericVocab::WriteVocab(ofstream &ofs) {
  tialnn::write_single(ofs, size_);
  tialnn::write_string2T_unordered_map(ofs, idx4type_);
  tialnn::write_1d_string(ofs, type4idx_);
}

} // namespace memnet
