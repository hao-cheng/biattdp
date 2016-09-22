/*!
 * \file numeric.h
 *
 * 
 * 
 * \version 0.0.1
 */

#ifndef TIALNN_UTIL_NUMERIC_H_
#define TIALNN_UTIL_NUMERIC_H_

// assert
#include <cassert>
#include "../base.h"

namespace tialnn {

//! =================================
//! \note Defines the BLAS interface based on single/double precision.
//! =================================
#if defined(_TIALNN_USE_MKL) || defined(_TIALNN_USE_OPENBLAS) //<! use MKL or OpenBLAS
#ifdef _TIALNN_SINGLE_PRECISION //<! use single precision
#define TIALNN_CBLAS_ASUM cblas_sasum
#define TIALNN_CBLAS_AXPY cblas_saxpy
#define TIALNN_CBLAS_DOT cblas_sdot
#define TIALNN_CBLAS_SCAL cblas_sscal

#define TIALNN_CBLAS_GEMV cblas_sgemv
#define TIALNN_CBLAS_GER cblas_sger
#define TIALNN_CBLAS_SBMV cblas_ssbmv

#define TIALNN_CBLAS_GEMM cblas_sgemm

#define TIALNN_CBLAS_AXPBY cblas_saxpby
#endif // _TIALNN_SINGLE_PRECISION

#ifdef _TIALNN_DOUBLE_PRECISION //<! use double precision
#define TIALNN_CBLAS_ASUM cblas_dasum
#define TIALNN_CBLAS_AXPY cblas_daxpy
#define TIALNN_CBLAS_DOT cblas_ddot
#define TIALNN_CBLAS_SCAL cblas_dscal

#define TIALNN_CBLAS_GEMV cblas_dgemv
#define TIALNN_CBLAS_GER cblas_dger
#define TIALNN_CBLAS_SBMV cblas_dsbmv

#define TIALNN_CBLAS_GEMM cblas_dgemm

#define TIALNN_CBLAS_AXPBY cblas_daxpby
#endif // _TIALNN_DOUBLE_PRECISION

#endif // _TIALNN_USE_MKL || _TIALNN_USE_OPENBLAS
//! =================================


//! =================================
//! Mikolov's fastexp implementation
//! =================================
static union {
  double d;
  struct {
    int j, i;
  } n;
} d2i;
#define EXP_A (1048576 / 0.69314718055994530942)
#define EXP_C 60801
#define MIKOLOV_FAST_EXP(y) (d2i.n.i = static_cast<int>(EXP_A * (y) + (1072693248 - EXP_C)), d2i.d)
//! =================================

//! =================================
//! Cutoff for numerical stability.
//! \note May need to tune these constants.
//! =================================
//! Cutoff of error inputs.
static const float kActivationCutoff = 50.0f;
//! Cutoff of activation inputs.
static const float kErrorCutoff = 15.0f;
//! =================================

//! Computes sigmoid activation function.
struct sigmoid_f {
  ActivationType operator() (ActivationType ac) {
    //! for numeric stability.
    if (ac > kActivationCutoff) ac = kActivationCutoff;
    if (ac < -kActivationCutoff) ac = -kActivationCutoff;

    return static_cast<ActivationType>(1.0f / (1.0f + MIKOLOV_FAST_EXP(-ac)));
  }
};

//! Computes sigmoid error function.
struct sigmoid_g {
  ErrorType operator() (ErrorType er, ActivationType ac) {
    if (er > kErrorCutoff) er = kErrorCutoff;
    if (er < -kErrorCutoff) er = -kErrorCutoff;

    return static_cast<ErrorType>(er * ac * (1.0f - ac));
  }
};

//! Computes tanh activation function.
struct tanh_f {
  ActivationType operator() (ActivationType ac) {
    //! for numeric stability.
    if (ac > kActivationCutoff) ac = kActivationCutoff;
    if (ac < -kActivationCutoff) ac = -kActivationCutoff;

    const double x = MIKOLOV_FAST_EXP(ac);
    const double y = MIKOLOV_FAST_EXP(-ac);
    return static_cast<ActivationType>((x - y) / (x + y));
  }
};

//! Computes tanh error function.
struct tanh_g {
  ErrorType operator() (ErrorType er, ActivationType ac) {
    //! for numeric stability.
    if (er > kErrorCutoff) er = kErrorCutoff;
    if (er < -kErrorCutoff) er = -kErrorCutoff;

    return static_cast<ErrorType>(er * (1.0f - ac) * (1.0f + ac));
  }
};

//! Computes exponent activation function.
struct exp_f {
  const ActivationType offset;
  exp_f(ActivationType _offset) : offset(_offset) {}

  ActivationType operator() (ActivationType ac) {
    ac -= offset;
    //! for numeric stability.
    assert(ac <= kActivationCutoff);
    if (ac < -kActivationCutoff) ac = -kActivationCutoff;

    return static_cast<ActivationType>(MIKOLOV_FAST_EXP(ac));
  }
};

//! Computes exponent error function.
struct exp_g {
  ErrorType operator() (ErrorType er, ActivationType ac) {
    //! for numeric stability.
    if (er > kErrorCutoff) er = kErrorCutoff;
    if (er < -kErrorCutoff) er = -kErrorCutoff;

    return static_cast<ErrorType>(er * ac);
  }
};

//! Computes leaky recifier activation function.
struct leaky_relu_f {
  ActivationType operator() (ActivationType ac) {
    //! for numeric stability.
    if (ac > kActivationCutoff) ac = kActivationCutoff;
    if (ac < -kActivationCutoff) ac = -kActivationCutoff;

    if (ac > -1e-6f) return ac;
    return 0.1f * ac;
  }
};

//! Computes leaky rectifier error function.
struct leaky_relu_g {
  ErrorType operator() (ErrorType er, ActivationType ac) {
    //! for numeric stability.
    if (er > kErrorCutoff) er = kErrorCutoff;
    if (er < -kErrorCutoff) er = -kErrorCutoff;

    if (ac > -1e-6f) return er;
    return 0.1f * er;
  }
};

//! Computes leaky tanh ativation function.
struct leaky_tanh_f {
  ActivationType operator() (ActivationType ac) {
    //! for numeric stability.
    if (ac > kActivationCutoff) ac = kActivationCutoff;
    if (ac < -kActivationCutoff) ac = -kActivationCutoff;
    const double x = MIKOLOV_FAST_EXP(ac);
    const double y = MIKOLOV_FAST_EXP(-ac);
    if (ac < -1e-6f) {
      return static_cast<ActivationType>(0.1f * (x - y) / (x + y));
    }

    return static_cast<ActivationType>((x - y) / (x + y));
  }
};

//! Computes leaky tanh error function.
struct leaky_tanh_g {
  ErrorType operator() (ErrorType er, ActivationType ac) {
    //! for numeric stability.
    if (er > kErrorCutoff) er = kErrorCutoff;
    if (er < -kErrorCutoff) er = -kErrorCutoff;

    return static_cast<ErrorType>(er * (1.0 - ac) * (1.0 + ac));
  }
};

} // namespace tialnn

#endif
