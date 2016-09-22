/*!
 * \file base.h
 * \brief definitions of base types and macros.
 *
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_BASE_H_
#define TIALNN_BASE_H_

#ifndef NDEBUG
//! Debug mode.
#define _TIALNN_DEBUG
#endif

#ifdef _TIALNN_USE_MKL     //<! use MKL
#include <mkl.h>

namespace tialnn {

typedef MKL_INT IndexType;

} // namespace tialnn
#endif // _TIALNN_USE_MKL

#ifdef _TIALNN_USE_OPENBLAS     //<! use OpenBLAS
#include <cblas.h>

namespace tialnn {

typedef CBLAS_INDEX IndexType;

} // namespace tialnn
#endif // _TIALNN_USE_OPENBLAS

#ifdef _TIALNN_SINGLE_PRECISION
namespace tialnn {

typedef float ActivationType;
typedef float ErrorType;
typedef float WeightType;
typedef float GradientType;

} // namespace tialnn
#endif // _TIALNN_SINGLE_PRECISION

#ifdef _TIALNN_DOUBLE_PRECISION
namespace tialnn {

typedef double ActivationType;
typedef double ErrorType;
typedef double WeightType;
typedef double GradientType;

} // namespace tialnn
#endif // _TIALNN_DOUBLE_PRECISION

#endif
