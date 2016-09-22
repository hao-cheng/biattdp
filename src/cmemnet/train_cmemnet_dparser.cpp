/*!
 * \file train_cmemnet_dparser.cpp
 *
 * 
 * \version 0.0.6
 */

#ifdef __GNUC__
//! Disable this warning for BOOST.
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

// EXIT_SUCCESS
// EXIT_FAILURE
#include <cstdlib>
// std::cout
// std::cerr
// std::endl
#include <iostream>
// std::ifstream
#include <fstream>
// std::string
#include <string>
// std::vector
#include <vector>
// boost::program_options
#include <boost/program_options.hpp>
// tialnn::IndexType
#include <tialnn/base.h>
// memnet::AlgorithmOptions
#include "../memnet/algorithm_options.h"
// memnet::ModelOptions
#include "../memnet/model_options.h"
// memnet::MemNetDParserTrainer
#include "../memnet/memnet_dparser_trainer.h"
// cmemnet::CMemNetDParser
#include "cmemnet_dparser.h"

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::string;
using std::vector;
namespace po = boost::program_options;
using tialnn::IndexType;

#define PROGRAM_NAME "C-Memory Network Dependency Parser Trainer"
#define VERSION "0.0.6"

int main(int argc, char **argv) {

  string config_file;
  int num_threads;
  int debug;

  bool restart;
  string testfile;
  string outbase;

  memnet::AlgorithmOptions algopts;
  memnet::ModelOptions modelopts;
  int solver;
  int initializer;

  int num_composition_neurons;
  int num_projection_neurons;
  int num_hidden_neurons;

  // Declare a group of options that will be 
  // allowed only on command line
  po::options_description generic("Generic options");
  generic.add_options()
      ("version,v", "print version information")
      ("help,h", "print help message")
      ("config,c", po::value<string>(&config_file)->default_value(""),
       "name of a file of a configuration (can be overwritten by command line options)")
      ;

  // Declare a group of options that will be 
  // allowed both on command line and in
  // config file
  po::options_description config("Configuration");
  config.add_options()
      ("nthreads", po::value<int>(&num_threads)->default_value(0),
       "number of threads to use; 0 means using the default number of threads for OpenMP run-time library")
      ("debug", po::value<int>(&debug)->default_value(0),
       "debug level")
      ("inmodel", po::value<string>(&algopts.init_model_filename)->default_value(""),
       "name of the model for testing mode / training restart mode")
      ("restart", po::value<bool>(&restart)->default_value(false)->implicit_value(true),
       "initialize the model with --inmod")
      ("testfile", po::value<string>(&testfile)->default_value(""),
       "test data")
      ("outbase", po::value<string>(&outbase)->default_value(""),
       "basename for outbase.model")
      ("batch", po::value<IndexType>(&(modelopts.batch_size))->default_value(1),
       "batch size")
      ("max-len", po::value<IndexType>(&(modelopts.max_sequence_length))->default_value(0),
       "maximum acceptable sequence length in the data (including <root>), it is used to allocate enough memory for the model")
      ("trainfiles", po::value<vector<string>>(&(algopts.train_filenames))->multitoken(),
       "training data file(s)")
      ("validationfiles", po::value<vector<string>>(&(algopts.validation_filenames))->multitoken(),
       "validation data file(s)")
      ("word-vocfile", po::value<string>(&(algopts.word_vocab_filename))->default_value(""),
       "word vocabulary file (including <root> and <unk>)")
      ("lemma-vocfile", po::value<string>(&(algopts.lemma_vocab_filename))->default_value(""),
       "lemma vocabulary file (including <root> and <unk>)")
      ("cpos-vocfile", po::value<string>(&(algopts.cpos_vocab_filename))->default_value(""),
       "cpos vocabulary file)")
      ("pos-vocfile", po::value<string>(&(algopts.pos_vocab_filename))->default_value(""),
       "pos vocabulary file)")
      ("feature-vocfile", po::value<string>(&(algopts.feature_vocab_filename))->default_value(""),
       "feature vocabulary file)")
      ("dependency-vocfile", po::value<string>(&(algopts.dependency_vocab_filename))->default_value(""),
       "dependency vocabulary file")
      ("init-wordembd", po::value<string>(&algopts.init_wordembd_filename)->default_value(""),
       "name of the initial word embedding txt file")
      ("save-all-models", po::value<bool>(&(algopts.save_all_models))->default_value(false)->implicit_value(true),
       "save all models in each epoch")
      ("shuffle-datafiles", po::value<bool>(&(algopts.shuffle_datafiles))->default_value(false)->implicit_value(true),
       "shuffle training data files at the beginning of each epoch")
      ("shuffle-datapoints", po::value<bool>(&(algopts.shuffle_datapoints))->default_value(false)->implicit_value(true),
       "shuffle datapoints within each training data file")
      ("init-alpha", po::value<float>(&(algopts.init_learning_rate))->default_value(0.1f),
       "initial learning rate")
      ("min-improvement", po::value<double>(&(algopts.min_improvement))->default_value(1.0),
       "minimum improvement scale for the log-likelihood")
      ("skip-iters", po::value<int>(&(algopts.num_noneval_iterations))->default_value(0),
       "number of leading iterations without evaluation on validation data\n+x: skip evaluation of first x iterations\n-x: skip evaluation of both first x iterations and iterations after learning rate halves; in this case, the training terminates based on max_iterations")
      ("max-iters", po::value<int>(&(algopts.max_iterations))->default_value(0),
       "maximum number of iterations\n0: do not check max_iterations\n+x: run at most x iterations\n-x: only valid when num_noneval_iterations < 0; run x iterations after learning rate halves\n")
      ("gamma", po::value<float>(&(algopts.multitask_weight_arclabel))->default_value(0.0f),
       "multitask weight on arc label prediction\n")
      ("solver", po::value<int>(&solver)->default_value(0),
       "0-vanilla grad; 1-adagrad; 2-adadelta")
      ("initializer", po::value<int>(&(initializer))->default_value(0),
       "0-Mikolov; 1-Xarvier; 2-Gaussin")
      ("initializer-std", po::value<float>(&(modelopts.initializer_std))->default_value(0),
       "Standard deviation of the initializer (valid only for Gaussian initializer)")
      ("nhidden", po::value<int>(&num_hidden_neurons)->default_value(0),
       "number of hidden neurons")
      ("ncomposition", po::value<int>(&num_composition_neurons)->default_value(0),
       "number of neurons for composition layer")
      ("nprojection", po::value<int>(&num_projection_neurons)->default_value(0),
       "number of neurons for compositional projection layer")
      ;

  po::options_description cmdline_options;
  cmdline_options.add(generic).add(config);

  po::options_description config_file_options;
  config_file_options.add(config);

  po::variables_map vm;
  try {
    store(parse_command_line(argc, argv, cmdline_options), vm);
  } catch (po::error& e) {
    cerr << "ERROR: " << e.what() << endl;
    cerr << cmdline_options << endl;
    return EXIT_FAILURE;
  }
  notify(vm);

  if (vm.count("help")) {
    cout << PROGRAM_NAME << endl;
    cout << cmdline_options << endl;
    return EXIT_SUCCESS;
  }
  if (vm.count("version")) {
    cout << PROGRAM_NAME << endl;
    cout << "version: " << VERSION << endl;
    return EXIT_SUCCESS;
  }

  if (!config_file.empty()) {
    ifstream ifs(config_file.c_str());
    if (!ifs) {
      cerr << "can not open config file: " << config_file << endl;
      return 0;
    } else {
      try {
        store(parse_config_file(ifs, config_file_options), vm);
      } catch (po::error& e) {
        cerr << "ERROR: " << e.what() << endl;
        cerr << config_file_options << endl;
        return EXIT_FAILURE;
      }
    }

    // command-line options overwrite config file options
    store(parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);
  }

#ifdef _TIALNN_USE_MKL
  mkl_set_num_threads(num_threads);
#endif
#ifdef _TIALNN_USE_OPENBLAS
  openblas_set_num_threads(num_threads);
#endif
  cout << "num_threads: " << num_threads << endl;

  cmemnet::CMemNetDParser cmemnet_dparser;
  cmemnet_dparser.set_debug(debug);
  //! (Over)-write the modelopts.
  if (solver == 0) {
    modelopts.solver = modelopts.vanilla;
  } else if (solver == 1) {
    modelopts.solver = modelopts.adagrad;
  } else if (solver == 2) {
    modelopts.solver = modelopts.adadelta;
  } else if (solver == 3) {
    modelopts.solver = modelopts.adam;
  } else {
    cerr << "Invalid model options" << endl;
    return EXIT_FAILURE;
  }
  if (initializer == 0) {
    modelopts.initializer = modelopts.mikolov;
  } else if (initializer == 1) {
    modelopts.initializer = modelopts.xarvier;
  } else if (initializer == 2) {
    modelopts.initializer = modelopts.gaussian;
  } else {
    cerr << "Invalid initializer options" << endl;
    return EXIT_FAILURE;
  }
  cmemnet_dparser.set_modelopts(modelopts);

  memnet::MemNetDParserTrainer trainer(&cmemnet_dparser);
  trainer.set_debug(debug);

  if (!algopts.init_model_filename.empty() && !restart) {
    if (testfile.empty()) {
      cerr << "Confused: reading a model, but no test data specified" << endl;
      return EXIT_FAILURE;
    }
    trainer.ReadMemNet(algopts.init_model_filename);
    trainer.EvalMemNet(testfile);
    return EXIT_SUCCESS;
  }

  if (restart) {
    if (algopts.init_model_filename.empty()) {
      cerr << "Confused: reset is set but no --inmod specified" << endl;
      return EXIT_FAILURE;
    }
    if (!algopts.word_vocab_filename.empty()) {
      cerr << "should not specify --word-vocfile!" << endl;
      return EXIT_FAILURE;
    }
    if (!algopts.lemma_vocab_filename.empty()) {
      cerr << "should not specify --lemma-vocfile!" << endl;
      return EXIT_FAILURE;
    }
    if (!algopts.cpos_vocab_filename.empty()) {
      cerr << "should not specify --cpos-vocfile!" << endl;
      return EXIT_FAILURE;
    }
    if (!algopts.pos_vocab_filename.empty()) {
      cerr << "should not specify --pos-vocfile!" << endl;
      return EXIT_FAILURE;
    }
    if (!algopts.feature_vocab_filename.empty()) {
      cerr << "should not specify --feature-vocfile!" << endl;
      return EXIT_FAILURE;
    }
    if (!algopts.dependency_vocab_filename.empty()) {
      cerr << "should not specify --dependency-vocfile!" << endl;
      return EXIT_FAILURE;
    }
    if (!algopts.init_wordembd_filename.empty()) {
      cerr << "should not specify --init-wordembd" << endl;
      return EXIT_FAILURE;
    }

    if (num_composition_neurons != 0) {
      cerr << "should not specify --ncomposition !" << endl;
      return EXIT_FAILURE;
    }
    if (num_hidden_neurons != 0) {
      cerr << "should not specify --nhidden !" << endl;
      return EXIT_FAILURE;
    }
  } else {
    if (algopts.word_vocab_filename.empty()) {
      cerr << "--word-vocfile not specified!" << endl;
      return EXIT_FAILURE;
    }
    if (algopts.dependency_vocab_filename.empty()) {
      cerr << "--dependency-vocfile not specified!" << endl;
      return EXIT_FAILURE;
    }

    cmemnet_dparser.set_num_composition_neurons(num_composition_neurons);
    cmemnet_dparser.set_num_projection_neurons(num_projection_neurons);
    cmemnet_dparser.set_num_hidden_neurons(num_hidden_neurons);
  }

  if (outbase.empty()) {
    cerr << "--outbase not specified!" << endl;
    return EXIT_FAILURE;
  }
  if (!testfile.empty()) {
    cerr << "--testfile specified in train mode!" << endl;
    return EXIT_FAILURE;
  }

  trainer.set_algopts(algopts);
  trainer.TrainMemNet(outbase);

  cmemnet_dparser.DestroyModel();

  return EXIT_SUCCESS;
}
