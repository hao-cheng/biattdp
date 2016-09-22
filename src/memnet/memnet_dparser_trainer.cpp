/*!
 * \file memnet_dparser_trainer.cpp
 *
 * 
 * \version 0.0.6
 */

#include "memnet_dparser_trainer.h"

// assert
#include <cassert>
// std::exit
#include <cstdlib>
// std::log
// std::exp
// std::sqrt
// std::abs
#include <cmath>
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
// std::to_string
// std::stod
#include <string>
// std::vector
#include <vector>
// std::unordereds_set
#include <unordered_set>
// std::numeric_limits
#include <limits>
// boost::tokenizer
// boost::char_separator
#include <boost/tokenizer.hpp>
// tialnn::IndexType
// tialnn::WeightType
#include <tialnn/base.h>
// memnet::DataPointBatch
#include "datapoint_batch.h"
// memnet::DataPointReader
#include "datapoint_reader.h"

using std::exit;
using std::log;
using std::exp;
using std::sqrt;
using std::abs;
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
using std::stod;
using std::vector;
using std::unordered_set;
using std::numeric_limits;
using boost::tokenizer;
using boost::char_separator;
using tialnn::IndexType;
using tialnn::WeightType;

namespace memnet {

void MemNetDParserTrainer::CheckParams() {
  if (!algopts_.CheckOpts()) {
    cerr << "check algopts_ failed!" << endl;
    exit(EXIT_FAILURE);
  }
  if (!(ptr_model_->CheckParams())) {
    exit(EXIT_FAILURE);
  }
}

void MemNetDParserTrainer::PrintParams() {
  algopts_.PrintOpts();
  ptr_model_->PrintParams();
}

void MemNetDParserTrainer::TrainMemNet(const string &outbase) {
  if (algopts_.init_model_filename.empty()) {
    //! Prepare for the training.
    //! Equivalent to ReadMemNet.
    ptr_model_->ReadVocabsFromTxt(algopts_.word_vocab_filename,
                                  algopts_.lemma_vocab_filename,
                                  algopts_.cpos_vocab_filename,
                                  algopts_.pos_vocab_filename,
                                  algopts_.feature_vocab_filename,
                                  algopts_.dependency_vocab_filename);

    PrintParams();
    CheckParams();

    ptr_model_->AllocateModel();
    ptr_model_->InitializeNeuralNet();
  } else {
    ReadMemNet(algopts_.init_model_filename);
    PrintParams();
    CheckParams();
  }

  if (!algopts_.init_wordembd_filename.empty()) {
    ReadWordEmbeddingsFromTxt(algopts_.init_wordembd_filename);
  }

  //! Read the data
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
  DataPointReader train_data(algopts_.train_filenames, 
                             &(ptr_model_->word_vocab()),
                             ptr_lemma_vocab,
                             ptr_cpos_vocab,
                             ptr_pos_vocab,
                             ptr_feature_vocab,
                             &(ptr_model_->dependency_vocab()),
                             algopts_.shuffle_datafiles, algopts_.shuffle_datapoints,
                             ptr_model_->modelopts().max_sequence_length,
                             ptr_model_->modelopts().batch_size);
  DataPointReader validation_data(algopts_.validation_filenames,
                                  &(ptr_model_->word_vocab()),
                                  ptr_lemma_vocab,
                                  ptr_cpos_vocab,
                                  ptr_pos_vocab,
                                  ptr_feature_vocab,
                                  &(ptr_model_->dependency_vocab()),
                                  false, false, 
                                  ptr_model_->modelopts().max_sequence_length,
                                  ptr_model_->modelopts().batch_size);

  //! Train the model on the DataPointReader objects.
  SGDTrain(train_data, validation_data, outbase);

  cout << "================================================================================" << endl;
  cout << "log-likelihood (base e) of validation is: " \
      << EvalMemNet(validation_data) << endl;
}

void MemNetDParserTrainer::EvalMemNet(const string &infile) {
  CheckParams();
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
  cout << "log-likelihood (base e) of " << infile << " is: " << EvalMemNet(data) << endl;
}

void MemNetDParserTrainer::ReadMemNet(const string &infile) {
  cout << "============================" << endl;
  cout << "reading " << infile << endl;
  ifstream ifs;
  ifs.open(infile, ios::binary | ios::in);
  if (ifs.fail()) {
    cout << "unable to open " << infile << endl;
    exit(EXIT_FAILURE);
  }

  //! algopts is overwritten by commandline argument. 
  AlgorithmOptions dummy_algopts;
  dummy_algopts.ReadOpts(ifs);

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

void MemNetDParserTrainer::WriteMemNet(const string &outbase) {
  string modelname(outbase);
  modelname += ".model";
  cout << "writing " << modelname << endl;
  ofstream ofs;
  ofs.open(modelname, ios::binary | ios::out);
  if (ofs.fail()) {
    cout << "unable to open " << modelname << endl;
    exit(EXIT_FAILURE);
  }

  algopts_.WriteOpts(ofs);

  ptr_model_->WriteMemNet(ofs);

  ofs.close();
}

void MemNetDParserTrainer::ReadWordEmbeddingsFromTxt(const string &infile) {
  ifstream ifs;
  ifs.open(infile, ios::in);
  if (ifs.fail()) {
    cout << "unable to open " << infile << endl;
    exit(EXIT_FAILURE);
  }

  string line;
  char_separator<char> space_separator(" ");
  unordered_set<IndexType> touched;

  cout << "reading " << infile << endl;
  while (getline(ifs, line)) {
    if (ifs.eof()) {
      break;
    }

    tokenizer<char_separator<char>> tokens(line, space_separator);
    tokenizer<char_separator<char>>::const_iterator it = tokens.begin();
    IndexType widx = ptr_model_->word_vocab().idx4type(*it);
    if (widx == ptr_model_->word_vocab().unk_idx()) {
      continue;
    }
    if (touched.find(widx) != touched.end()) {
      cerr << "Doubly defined embeddings for word:" << *it << endl;
      exit(EXIT_FAILURE);
    }
    touched.insert(widx);

    vector<WeightType> embd;
    ++it;
    for (; it != tokens.end(); ++it) {
      embd.push_back(static_cast<WeightType>(stod(*it)));
    }
    ptr_model_->SetWordEmbeddings(widx, embd);
  }
  cout << "Loaded word embeddings for " << touched.size() << " words." << endl;
  cout << "=================================" << endl;
  // try one more read before check eof
  ifs.get();
  if (!ifs.eof()) {
    cout << "Error reading word embeddings: should be at the end of the file" << endl;
    exit(EXIT_FAILURE);
  }
  ifs.close();
}

double MemNetDParserTrainer::EvalMemNet(DataPointReader &data) {
  const ModelOptions &modelopts = ptr_model_->modelopts();
  const DataPointBatch *ptr_databatch = nullptr;
  vector<IndexType> idx_datapoints;

  double total_logp = 0.0;
  int last_num_processed = 0;
  int num_processed = 0;

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
    double logp = ptr_model_->GetLogProb(ptr_databatch, idx_datapoints, algopts_.multitask_weight_arclabel);
    total_logp += logp;
    if (debug_ > 1) {
      cerr << "log-likelihood (base e) of current batch is: " << logp;
      cerr << " (" << total_logp << ")" << endl;
    }

    last_num_processed = num_processed;
    num_processed += static_cast<int>(idx_datapoints.size());
    if ((last_num_processed / 500) < (num_processed / 500)) {
      cout << "." << flush;
    }
  }
  cout << endl << "num of data samples: " << num_processed << endl;

#ifdef _MSC_VER //!< seems like this memory leakage only occurrs under Windows.
#ifdef _TIALNN_USE_MKL
  //! \note Call free buffers every epoch to avoid memory leakage issue.
  //!       See https://software.intel.com/en-us/articles/memory-leak-when-using-intel-mkl.
  MKL_INT64 AllocatedBytes;
  int N_AllocatedBuffers;

  AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
  cout << (long)AllocatedBytes / 1024 << " KB in " << N_AllocatedBuffers << " buffers" << endl;

  //!< \note Only free mkl buffers for this thread.
  //!        Calling mkl_free_buffers() may free buffers allocated by other threads.
  mkl_thread_free_buffers();
#endif
#endif

  return total_logp;
}

void MemNetDParserTrainer::SGDTrain(DataPointReader &train_data,
                                    DataPointReader &validation_data,
                                    const string &outbase) {
  if (outbase.empty()) {
    cerr << "outbase is empty!" << endl;
    exit(EXIT_FAILURE);
  }
  //! Try to write the model in case the outbase is invalid.
  WriteMemNet(outbase + ".INIT");

  const ModelOptions &modelopts = ptr_model_->modelopts();
  const DataPointBatch *ptr_databatch = nullptr;
  vector<IndexType> idx_datapoints;

  cout << "Checking training data..." << endl;
  train_data.StartEpoch();
  while (true) {
    ptr_databatch = train_data.GetDataPointBatchPtr(idx_datapoints);
    if (!ptr_databatch) {
      break;
    }
  }
  cout << "Checking validation data..." << endl;
  validation_data.StartEpoch();
  while (true) {
    ptr_databatch = validation_data.GetDataPointBatchPtr(idx_datapoints);
    if (!ptr_databatch) {
      break;
    }
  }

  double last_logp = -numeric_limits<double>::max();
  bool halve_alpha = false;
  //! set the current learning rate.
  float curr_learning_rate = algopts_.init_learning_rate;
  int iteration = 0;
  int iteration_halve_alpha = 0;
  auto start_time = system_clock::now();
  auto end_time = start_time;
  auto elapsed = duration_cast<milliseconds>(end_time - start_time).count();

  while (true) {
    cout << "******************************* ITERATION " << iteration << " *******************************" << endl;
    cout << "learning_rate = " << curr_learning_rate << endl;

    //! total_logp records logp of train_data
    double total_logp = 0.0;
    int last_num_processed = 0;
    int num_processed = 0;

    train_data.StartEpoch();
    while (true) {
      ptr_databatch = train_data.GetDataPointBatchPtr(idx_datapoints);
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
      double logp = ptr_model_->SetErrorsForOutputLayers(ptr_databatch, idx_datapoints, algopts_.multitask_weight_arclabel);
      total_logp += logp;
      if (debug_ > 1) {
        cerr << "log-likelihood (base e) of current batch is: " << logp;
        cerr << " (" << total_logp << ")" << endl;
      }
      ptr_model_->BackwardPropagate();
      //! \note We may design clever learnig rate schedule according to sequence
      //!       length in future.
      ptr_model_->UpdateWeights(curr_learning_rate);

      last_num_processed = num_processed;
      num_processed += static_cast<int>(idx_datapoints.size());
      if ((last_num_processed / 500) < (num_processed / 500)) {
        cout << "." << flush;
      }
    }
    cout << endl << "epoch finished" << endl << flush;
    cout << "num of data samples: " << num_processed << endl;
    cout << "log-likelihood (base e) of training is: " << total_logp << endl;

#ifdef _MSC_VER //!< seems like this memory leakage only occurrs under Windows.
#ifdef _TIALNN_USE_MKL
    //! \note Call free buffers every epoch to avoid memory leakage issue.
    //!       See https://software.intel.com/en-us/articles/memory-leak-when-using-intel-mkl.
    MKL_INT64 AllocatedBytes;
    int N_AllocatedBuffers;

    AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);
    cout << (long)AllocatedBytes / 1024 << " KB in " << N_AllocatedBuffers << " buffers" << endl;

    //!< \note Only free mkl buffers for this thread.
    //!        Calling mkl_free_buffers() may free buffers allocated by other threads.
    mkl_thread_free_buffers();
#endif
#endif

    if (!outbase.empty()) {
      if (algopts_.save_all_models) {
        WriteMemNet(outbase + ".ITER_" + to_string(iteration));
      }
    }

    auto last_end_time = end_time;
    end_time = system_clock::now();
    cout << "time elapsed ";
    elapsed = duration_cast<milliseconds>(end_time - last_end_time).count();
    cout << elapsed / 1000.0 << " secs for this iteration out of ";
    elapsed = duration_cast<milliseconds>(end_time - start_time).count();
    cout << elapsed / 1000.0 << " secs in total." << endl;

    //! Stop criterion: max iterations
    iteration++;
    if (algopts_.max_iterations > 0) {
      if (iteration >= algopts_.max_iterations) {
        if (!outbase.empty()) {
          WriteMemNet(outbase);
        }
        break;
      }
    } else if (algopts_.max_iterations < 0) {
      assert(algopts_.num_noneval_iterations < 0);
      if (iteration_halve_alpha > 0 && iteration - iteration_halve_alpha >= -algopts_.max_iterations) {
        if (!outbase.empty()) {
          WriteMemNet(outbase);
        }
        break;
      }
    }

    //! Skip validation
    if (iteration <= abs(algopts_.num_noneval_iterations)) {
      continue;
    }

    if (halve_alpha) {
      curr_learning_rate /= 2;
      if (algopts_.num_noneval_iterations < 0) {
        continue;
      }
    }

    cout << "----------VALIDATION----------" << endl;
    double curr_logp = EvalMemNet(validation_data);
    cout << "log-likelihood (base e) of validation: " << curr_logp << endl;
    last_end_time = end_time;
    end_time = system_clock::now();
    cout << "time elapsed ";
    elapsed = duration_cast<milliseconds>(end_time - last_end_time).count();
    cout << elapsed / 1000.0 << " secs for this iteration out of ";
    elapsed = duration_cast<milliseconds>(end_time - start_time).count();
    cout << elapsed / 1000.0 << " secs in total." << endl;

    if (curr_logp < last_logp) {
      cout << "validation log-likelihood decrease; resetting parameters" << endl;
      ptr_model_->RestoreLastWeights();
    } else {
      ptr_model_->CacheCurrentWeights();
    }

    //! Stop criterion: validation
    if (!halve_alpha) {
      if (curr_logp <= last_logp) {
        halve_alpha = true;
        curr_learning_rate /= 2;
        iteration_halve_alpha = iteration;
      }
    } else {
      if (curr_logp * algopts_.min_improvement <= last_logp) {
        if (!outbase.empty()) {
          WriteMemNet(outbase);
        }
        break;
      }
    }

    if (last_logp < curr_logp) {
      last_logp = curr_logp;
    }
  }
}

} // namespace memnet
