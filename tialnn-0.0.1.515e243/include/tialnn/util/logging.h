/*!
 * \file logging.h
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_UTIL_LOGGING_H_
#define TIALNN_UTIL_LOGGING_H_

// std::cerr
// std::endl
#include <iostream>

namespace tialnn {

#define TIALNN_ERR(msg) do { \
  tialnn::tialnn_err(__FILE__, \
                     __LINE__, \
                     __FUNCTION__, \
                     msg); \
} while(0)

inline void tialnn_err(const char *file, 
                       const int line, 
                       const char *func, 
                       const char *msg) {
  std::cerr << *file << "(" << line << ") ";
  std::cerr << *func << " : ";
  std::cerr << *msg << std::endl;
}

} // namespace tialnn

#endif
