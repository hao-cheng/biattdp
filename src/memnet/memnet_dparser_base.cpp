/*!
 * \file memnet_dparser_base.cpp
 *
 * 
 * \version 0.0.6
 */

#include "memnet_dparser_base.h"

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::cout
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
// tialnn::write_xxx
// tialnn::read_xxx
#include <tialnn/util/futil.h>

using std::exit;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

namespace memnet {

void MemNetDParserBase::ReadVocabsFromTxt(const string &word_vocab_filename,
                                          const string &lemma_vocab_filename,
                                          const string &cpos_vocab_filename,
                                          const string &pos_vocab_filename,
                                          const string &feature_vocab_filename,
                                          const string &dependency_vocab_filename) {
  word_vocab_.ReadVocabFromTxt(word_vocab_filename);
  if (word_vocab_.empty()) {
    cerr << "empty word vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }

  if (!lemma_vocab_filename.empty()) {
    lemma_vocab_.ReadVocabFromTxt(lemma_vocab_filename);
    if (lemma_vocab_.empty()) {
      cerr << "empty lemma vocabulary!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (!cpos_vocab_filename.empty()) {
    cpos_vocab_.ReadVocabFromTxt(cpos_vocab_filename);
    if (cpos_vocab_.empty()) {
      cerr << "empty cpos vocabulary!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (!pos_vocab_filename.empty()) {
    pos_vocab_.ReadVocabFromTxt(pos_vocab_filename);
    if (pos_vocab_.empty()) {
      cerr << "empty pos vocabulary!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  if (!feature_vocab_filename.empty()) {
    feature_vocab_.ReadVocabFromTxt(feature_vocab_filename);
    if (feature_vocab_.empty()) {
      cerr << "empty feature vocabulary!" << endl;
      exit(EXIT_FAILURE);
    }
  }

  dependency_vocab_.ReadVocabFromTxt(dependency_vocab_filename);
  if (dependency_vocab_.empty()) {
    cerr << "empty dependency vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
}

bool MemNetDParserBase::CheckParams() {
  modelopts_.CheckOpts();

  if (word_vocab_.empty()) {
    cerr << "word_vocabulary is empty!" << endl;
    return false;
  }
  if (dependency_vocab_.empty()) {
    cerr << "dependency_vocabulary is empty!" << endl;
    return false;
  }
  return CheckParamsImpl();
}

void MemNetDParserBase::PrintParams() {
  modelopts_.PrintOpts();

  cout << "word_vocab_.size_: " << word_vocab_.size() << endl;
  cout << "word_vocab_.root_idx_: " << word_vocab_.root_idx() << endl;
  cout << "word_vocab_.unk_idx_: " << word_vocab_.unk_idx() << endl;
  cout << "lemma_vocab_.size_: " << lemma_vocab_.size() << endl;
  cout << "cpos_vocab_.size_: " << cpos_vocab_.size() << endl;
  cout << "pos_vocab_.size_: " << pos_vocab_.size() << endl;
  cout << "feature_vocab_.size_: " << feature_vocab_.size() << endl;
  cout << "dependency_vocab_.size_: " << dependency_vocab_.size() << endl;

  PrintParamsImpl();
}

void MemNetDParserBase::ReadMemNet(ifstream &ifs) {
  //! \note modelopts are overwritten by commandline argument.
  ModelOptions dummy_memopts;
  dummy_memopts.ReadOpts(ifs);

  word_vocab_.ReadVocab(ifs);
  if (word_vocab_.empty()) {
    cerr << "empty word vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }
  lemma_vocab_.ReadVocab(ifs);
  cpos_vocab_.ReadVocab(ifs);
  pos_vocab_.ReadVocab(ifs);
  feature_vocab_.ReadVocab(ifs);
  dependency_vocab_.ReadVocab(ifs);
  if (dependency_vocab_.empty()) {
    cerr << "empty dependency vocabulary!" << endl;
    exit(EXIT_FAILURE);
  }

  ReadMemNetImpl(ifs);

  for (auto &ptr_conn : conn_ptrs_) {
    ptr_conn->ReadConnection(ifs);
  }
}

void MemNetDParserBase::WriteMemNet(ofstream &ofs) {
  modelopts_.WriteOpts(ofs);

  word_vocab_.WriteVocab(ofs);
  lemma_vocab_.WriteVocab(ofs);
  cpos_vocab_.WriteVocab(ofs);
  pos_vocab_.WriteVocab(ofs);
  feature_vocab_.WriteVocab(ofs);
  dependency_vocab_.WriteVocab(ofs);

  WriteMemNetImpl(ofs);

  for (auto &ptr_conn : conn_ptrs_) {
    ptr_conn->WriteConnection(ofs);
  }
}

} // namespace memnet
