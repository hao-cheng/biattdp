/*!
 * \file algorithm_options.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_ALGORITHM_OPTIONS_H_
#define MEMNET_ALGORITHM_OPTIONS_H_

// std::size_t
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

namespace memnet {

//! Options related to the training algorithm.
//! (Not include options for gradient calculation)
struct AlgorithmOptions {
  AlgorithmOptions() :shuffle_datafiles(false), shuffle_datapoints(false),
    init_learning_rate(1.0f), min_improvement(1.0f), 
    num_noneval_iterations(0), max_iterations(0),
    save_all_models(false), multitask_weight_arclabel(0.0f) {}
  ~AlgorithmOptions() {}

  //! Prints the options.
  void PrintOpts() {
    std::cout << "init_model_filename: " << init_model_filename << std::endl;
    std::cout << "word_vocab_filename: " << word_vocab_filename << std::endl;
    std::cout << "lemma_vocab_filename: " << lemma_vocab_filename << std::endl;
    std::cout << "cpos_vocab_filename: " << cpos_vocab_filename << std::endl;
    std::cout << "pos_vocab_filename: " << pos_vocab_filename << std::endl;
    std::cout << "feature_vocab_filename: " << feature_vocab_filename << std::endl;
    std::cout << "dependency_vocab_filename: " << dependency_vocab_filename << std::endl;
    std::cout << "init_wordembd_filename: " << init_wordembd_filename << std::endl;
    std::cout << "train_filenames:";
    for (auto &fn : train_filenames) {
      std::cout << " " << fn;
    }
    std::cout << std::endl;
    std::cout << "validation_filenames:";
    for (auto &fn : validation_filenames) {
      std::cout << " " << fn;
    }
    std::cout << std::endl;
    std::cout << "shuffle_datafiles: " << shuffle_datafiles << std::endl;
    std::cout << "shuffle_datapoints: " << shuffle_datapoints << std::endl;
    std::cout << "init_learning_rate: " << init_learning_rate << std::endl;
    std::cout << "min_improvement: " << min_improvement << std::endl;
    std::cout << "num_noneval_iterations: " << num_noneval_iterations << std::endl;
    std::cout << "max_iterations: " << max_iterations << std::endl;
    std::cout << "save_all_models: " << save_all_models << std::endl;
    std::cout << "multitask_weight_arclabel: " << multitask_weight_arclabel << std::endl;
  }

  //! Checks the options.
  bool CheckOpts() {
    if (init_learning_rate <= 0) {
      std::cerr << "initial learning rate should be greater than 0!" << std::endl;
      return false;
    }
    if (min_improvement < 1) {
      std::cerr << "min-improvement should be no less than 1!" << std::endl;
      return false;
    }
    if (num_noneval_iterations < 0 && max_iterations == 0) {
      std::cerr << "number of non-evaluated itereators is negative but max_iterations_ is 0!" << std::endl;
      return false;
    }
    if (num_noneval_iterations > 0 && max_iterations < 0) { 
      std::cerr << "number of non-evaluated itereators is positive but max_iterations_ is negative!" << std::endl;
      return false;
    }
    if (multitask_weight_arclabel < 0) {
      std::cerr << "multitask weight on arc label prediction is negative!" << std::endl;
      return false;
    }

    return true;
  }

  //! Reads the options from the binary ifstream.
  void ReadOpts(std::ifstream &ifs) {
    std::cout << "***reading algorithm options***" << std::endl;

    tialnn::read_string(ifs, init_model_filename);
    tialnn::read_string(ifs, word_vocab_filename);
    tialnn::read_string(ifs, lemma_vocab_filename);
    tialnn::read_string(ifs, cpos_vocab_filename);
    tialnn::read_string(ifs, pos_vocab_filename);
    tialnn::read_string(ifs, feature_vocab_filename);
    tialnn::read_string(ifs, dependency_vocab_filename);
    tialnn::read_string(ifs, init_wordembd_filename);
    std::size_t n = 0;
    tialnn::read_single(ifs, n);
    train_filenames.resize(n);
    for (auto &fn : train_filenames) {
      tialnn::read_string(ifs, fn);
    }
    tialnn::read_single(ifs, n);
    validation_filenames.resize(n);
    for (auto &fn : validation_filenames) {
      tialnn::read_string(ifs, fn);
    }
    tialnn::read_single(ifs, shuffle_datafiles);
    tialnn::read_single(ifs, shuffle_datapoints);
    tialnn::read_single(ifs, init_learning_rate);
    tialnn::read_single(ifs, min_improvement);
    tialnn::read_single(ifs, num_noneval_iterations);
    tialnn::read_single(ifs, max_iterations);
    tialnn::read_single(ifs, save_all_models);
    tialnn::read_single(ifs, multitask_weight_arclabel);

    PrintOpts();
  }

  //! Writes the options into the binary ofstream.
  void WriteOpts(std::ofstream &ofs) {
    tialnn::write_string(ofs, init_model_filename);
    tialnn::write_string(ofs, word_vocab_filename);
    tialnn::write_string(ofs, lemma_vocab_filename);
    tialnn::write_string(ofs, cpos_vocab_filename);
    tialnn::write_string(ofs, pos_vocab_filename);
    tialnn::write_string(ofs, feature_vocab_filename);
    tialnn::write_string(ofs, dependency_vocab_filename);
    tialnn::write_string(ofs, init_wordembd_filename);
    tialnn::write_single(ofs, train_filenames.size());
    for (auto &fn : train_filenames) {
      tialnn::write_string(ofs, fn);
    }
    tialnn::write_single(ofs, validation_filenames.size());
    for (auto &fn : validation_filenames) {
      tialnn::write_string(ofs, fn);
    }
    tialnn::write_single(ofs, shuffle_datafiles);
    tialnn::write_single(ofs, shuffle_datapoints);
    tialnn::write_single(ofs, init_learning_rate);
    tialnn::write_single(ofs, min_improvement);
    tialnn::write_single(ofs, num_noneval_iterations);
    tialnn::write_single(ofs, max_iterations);
    tialnn::write_single(ofs, save_all_models);
    tialnn::write_single(ofs, multitask_weight_arclabel);
  }

  //! Filename of the initial model.
  std::string init_model_filename;
  //! Filename of the word vocabulary in txt format.
  std::string word_vocab_filename;
  //! Filename of the lemma vocabulary in txt format.
  std::string lemma_vocab_filename;
  //! Filename of the CPOS tag vocabulary in txt format.
  std::string cpos_vocab_filename;
  //! Filename of the POS tag vocabulary in txt format.
  std::string pos_vocab_filename;
  //! Filename of the feature vocabulary in txt format.
  std::string feature_vocab_filename;
  //! Filename of the dependency vocabulary in txt format.
  std::string dependency_vocab_filename;
  //! Filename of the initial word embedding.
  std::string init_wordembd_filename;
  //! Filename(s) of the training data.
  std::vector<std::string> train_filenames;
  //! Filename(s) of the validation data.
  std::vector<std::string> validation_filenames;
  //! If true, data files will be shuffled.
  bool shuffle_datafiles;
  //! If true, data points within the file will be shuffled.
  bool shuffle_datapoints;
  //! Initial learning rate.
  float init_learning_rate;
  //! Minimum improvement for each iteration. (more details)
  double min_improvement;
  //! Number of leading iterations without evaluation on validation data.
  //! +x: skip evaluation of first x iterations
  //! -x: skip evaluation of both first x iterations and iterations after
  //! learning rate halves; in this case, the training terminates based on
  //! max_iterations_.
  int num_noneval_iterations;
  //! Maximum number of iterations.
  //! 0: do not check max_iterations. 
  //! +x: run at most x iterations.
  //! -x: only valid when num_noneval_iterations < 0; run x iterations after
  //! learning rate halves.
  int max_iterations;
  //! True if save models at every iteration.
  bool save_all_models;
  //! Multitask weight on the arc label prediction.
  float multitask_weight_arclabel;
  
};

} // namespace memnet
#endif
