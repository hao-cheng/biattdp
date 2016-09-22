/*!
 * \file run_cmemnet_dparser.cpp
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
// memnet::MemNetDParserDecoder
#include "../memnet/memnet_dparser_decoder.h"
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

#define PROGRAM_NAME "C-Memory Network Dependency Parser"
#define VERSION "0.0.6"

int main(int argc, char **argv) {

  string config_file;
  int num_threads;
  int debug;

  string inmod;
  string infile;
  string outfile;

  bool mst;

  memnet::ModelOptions modelopts;

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
      ("inmodel", po::value<string>(&inmod)->default_value(""),
       "name of the model to use in testing mode")
      ("infile", po::value<string>(&infile)->default_value(""),
       "test data")
      ("outfile", po::value<string>(&outfile)->default_value(""),
       "output prediction file")
      ("batch", po::value<IndexType>(&(modelopts.batch_size))->default_value(1),
       "batch size")
      ("max-len", po::value<IndexType>(&(modelopts.max_sequence_length))->default_value(0),
       "maximum acceptable sequence length in the data (including <root>), it is used to allocate enough memory for the model")
      ("mst", po::value<bool>(&mst)->default_value(false)->implicit_value(true),
       "use MST parse?")
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

  memnet::MemNetDParserDecoder decoder(&cmemnet_dparser);
  decoder.set_debug(debug);

  if (inmod.empty()) {
    cerr << "--inmod not specified" << endl;
    return EXIT_FAILURE;
  }
  if (outfile.empty()) {
    cerr << "--outfile not specified" << endl;
    return EXIT_FAILURE;
  }

  // Re-write the modelopts.
  cmemnet_dparser.set_modelopts(modelopts);
  decoder.ReadMemNet(inmod);

  decoder.Predict(infile, outfile, mst);

  cmemnet_dparser.DestroyModel();

  return EXIT_SUCCESS;
}
