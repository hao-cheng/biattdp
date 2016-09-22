/*!
 * \file memnet_dparser_decoder.cpp
 *
 * 
 * \version 0.0.6
 */

#include "memnet_dparser_decoder.h"

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// chrono::time_point
// chrono::system_clock
// chrono::duration_cast
// chrono::milliseconds
#include <chrono>
// std::cout
// std::cerr
// std::endl
// std::flush
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// std::ios
#include <ios>
// std::string
#include <string>
// std::vector
#include <vector>
// tialnn::IndexType
#include <tialnn/base.h>
// memnet::DataPointBatch
#include "datapoint_batch.h"
// memnet::DataPointReader
#include "datapoint_reader.h"

using std::exit;
using std::chrono::time_point;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::string;
using std::to_string;
using std::vector;
using tialnn::IndexType;

namespace memnet {

void MemNetDParserDecoder::CheckParams() {
  if (!algopts_.CheckOpts()) {
    cerr << "check algopts_ failed!" << endl;
    exit(EXIT_FAILURE);
  }
  if (!(ptr_model_->CheckParams())) {
    exit(EXIT_FAILURE);
  }
}

void MemNetDParserDecoder::PrintParams() {
  algopts_.PrintOpts();
  ptr_model_->PrintParams();
}

void MemNetDParserDecoder::ReadMemNet(const string &infile) {
  cout << "============================" << endl;
  cout << "reading " << infile << endl;
  ifstream ifs;
  ifs.open(infile, ios::binary | ios::in);
  if (ifs.fail()) {
    cout << "unable to open " << infile << endl;
    exit(EXIT_FAILURE);
  }

  algopts_.ReadOpts(ifs);

  ptr_model_->ReadMemNet(ifs);

  //! try one more read before check eof
  ifs.get(); 
  if (!ifs.eof()) {
    cout << "Error reading model: should be at the end of the file" << endl;
    exit(EXIT_FAILURE);
  }
  ifs.close();
  cout << "============================" << endl;
}

void MemNetDParserDecoder::Predict(const string &infile, const string &outfile, bool mst) {
  PrintParams();
  CheckParams();

  ofstream ofs;
  ofs.open(outfile, ios::out);
  if (ofs.fail()) {
    cout << "unable to open " << outfile << endl;
    exit(EXIT_FAILURE);
  }

  vector<string> filenames = { infile };
  const WordVocab *ptr_lemma_vocab = nullptr;
  const WordVocab *ptr_cpos_vocab = nullptr;
  const WordVocab *ptr_pos_vocab = nullptr;
  const GenericVocab *ptr_feature_vocab = nullptr;
  if (!algopts_.lemma_vocab_filename.empty()) {
    ptr_lemma_vocab = &(ptr_model_->lemma_vocab());
  }
  if (!algopts_.cpos_vocab_filename.empty()) {
    ptr_cpos_vocab = &(ptr_model_->cpos_vocab());
  }
  if (!algopts_.pos_vocab_filename.empty()) {
    ptr_pos_vocab = &(ptr_model_->pos_vocab());
  }
  if (!algopts_.feature_vocab_filename.empty()) {
    ptr_feature_vocab = &(ptr_model_->feature_vocab());
  }
  DataPointReader data(filenames, 
                       &(ptr_model_->word_vocab()),
                       ptr_lemma_vocab,
                       ptr_cpos_vocab,
                       ptr_pos_vocab,
                       ptr_feature_vocab,
                       &(ptr_model_->dependency_vocab()),
                       false, false,
                       ptr_model_->modelopts().max_sequence_length,
                       ptr_model_->modelopts().batch_size);

  const ModelOptions &modelopts = ptr_model_->modelopts();
  const DataPointBatch *ptr_databatch = nullptr;
  vector<IndexType> idx_datapoints;
  vector<vector<DataPointToken>> datapoints;

  auto start_time = system_clock::now();
  int num_processed = 0;

  cout << "**************************************************************" << endl;

  data.StartEpoch();
  while (true) {
    ptr_databatch = data.GetDataPointBatchPtr(idx_datapoints);
    if (!ptr_databatch) {
      break;
    }
    IndexType n = static_cast<IndexType>(modelopts.batch_size - idx_datapoints.size());
    if (n > 0) {
      cout << ":" << n << flush;
    }

    ptr_model_->ResetActivations();
    ptr_model_->SetActivationsForInputLayers(ptr_databatch, idx_datapoints);
    ptr_model_->ForwardPropagate();
    if (mst) {
      ptr_model_->MSTPredict(ptr_databatch, idx_datapoints, datapoints);
    } else {
      ptr_model_->Predict(ptr_databatch, idx_datapoints, datapoints);
    }
    
    num_processed += static_cast<int>(idx_datapoints.size());
  }
  cout << endl << "num of data samples: " << num_processed << endl;

  auto end_time = system_clock::now();
  cout << "time elapsed ";
  auto elapsed = duration_cast<milliseconds>(end_time - start_time).count();
  cout << elapsed / 1000.0 << " secs in total." << endl;

  cout << "Writing to " << outfile << endl;
  WriteDataPointBatchToCoNLLX(datapoints, ofs);

  ofs.close();
}

//! Write datapoints to ofs in CoNLL-X format.
void MemNetDParserDecoder::WriteDataPointBatchToCoNLLX(const vector<vector<DataPointToken>> &datapoints,
                                                       std::ofstream &ofs) {
  for (auto rit = datapoints.rbegin(); rit != datapoints.rend(); ++rit) {
    IndexType t = 1;
    for (auto &tok : *rit) {
      if (tok.form == ptr_model_->word_vocab().blank_idx()) {
        break;
      }

      // ID
      ofs << t++ << "\t";
      // FORM
      ofs << ptr_model_->word_vocab().type4idx(tok.form) << "\t";
      // LEMMA
      ofs << "_\t";
      // CPOSTAG
      ofs << "_\t";
      // POSTAG
      ofs << "_\t";
      // FEATS
      ofs << "_\t";
      // HEAD
      ofs << tok.headword_position << "\t";
      // DEPREL
      ofs << ptr_model_->dependency_vocab().type4idx(tok.dependency) << "\t";
      // PHEAD
      ofs << "_\t";
      // PDEPREL
      ofs << "_" << endl;
    }
    ofs << endl;
  }
}

} // namespace memnet
