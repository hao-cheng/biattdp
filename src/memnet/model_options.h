/*!
 * \file model_options.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_MODEL_OPTIONS_H_
#define MEMNET_MODEL_OPTIONS_H_

// std::cout
// std::cerr
// std::endl
#include <iostream>
// std::ifstream
// std::ofstream
#include <fstream>
// tialnn::ErrorType
#include <tialnn/base.h>
// tialnn::write_xxx
// tialnn::read_xxx
#include <tialnn/util/futil.h>

namespace memnet {

//! Model options that are un-used / overwritten during test time.
struct ModelOptions {
  ModelOptions() : batch_size(0), max_sequence_length(0), solver(vanilla), initializer(mikolov), initializer_std(0.0f) {}
  ~ModelOptions() {}

  //! Prints the options.
  void PrintOpts() {
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "max_sequence_length: " << max_sequence_length << std::endl;
    std::cout << "solver: " << solver << std::endl;
    std::cout << "initializer: " << initializer << std::endl;
    std::cout << "initializer_std: " << initializer_std << std::endl;
  }

  //! Checks the options.
  bool CheckOpts() {
    if (batch_size <= 0) {
      std::cerr << "batch size should be greater than 0!" << std::endl;
      return false;
    }
    if (max_sequence_length < 1) {
      std::cerr << "maximum allowable sequence length should greater than 1!" << std::endl;
      return false;
    }
    if (initializer == mikolov || initializer == xarvier) {
      if (initializer_std != 0) {
        std::cerr << "initializer_std is invalid for initializer == mikolov or xarvier!" << std::endl;
        return false;
      }
    }
    if (initializer == gaussian) {
      if (initializer_std <= 0) {
        std::cerr << "initializer_std should be positive for initializer == gaussian!" << std::endl;
        return false;
      }
    }

    return true;
  }

  //! Reads the options from the binary ifstream.
  void ReadOpts(std::ifstream &ifs) {
    std::cout << "***reading model options***" << std::endl;

    tialnn::read_single(ifs, batch_size);
    tialnn::read_single(ifs, max_sequence_length);
    tialnn::read_single(ifs, solver);
    tialnn::read_single(ifs, initializer);
    tialnn::read_single(ifs, initializer_std);

    PrintOpts();
  }

  //! Writes the options into the binary ofstream.
  void WriteOpts(std::ofstream &ofs) {
    tialnn::write_single(ofs, batch_size);
    tialnn::write_single(ofs, max_sequence_length);
    tialnn::write_single(ofs, solver);
    tialnn::write_single(ofs, initializer);
    tialnn::write_single(ofs, initializer_std);
  }

  //! Batch size.
  tialnn::IndexType batch_size;
  //! Maximum allowable sequence length.
  tialnn::IndexType max_sequence_length;

  //! True if using AdaGrad update.
  enum GradientSolver { vanilla = 0, adagrad = 1, adadelta = 2, adam = 3 };
  GradientSolver solver;

  //! Weight initializer.
  enum WeightInitializer { mikolov = 0, xarvier = 1, gaussian = 2 };
  WeightInitializer initializer;
  //! Standard deviation of the weight initializer.
  //! It is valid for for initilizer == gaussian.
  float initializer_std;
};

} // namespace memnet
#endif
