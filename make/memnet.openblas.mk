OPENBLAS_0_2_18_DIR = /g/tial/sw/pkgs/OpenBLAS-0.2.18

# NOTE: REPO_DIR is relative from make/?memnet/Makefile
REPO_DIR = ../..

TIALNN_0_0_1_DIR = $(REPO_DIR)/tialnn-0.0.1.515e243
BOOST_1_55_0_DIR = $(REPO_DIR)/boost_1_55_0
BOOST_PROGRAM_OPTIONS_LIBDIR = $(REPO_DIR)/libs/x64

BIN_DIR = $(REPO_DIR)/bin
BIN_SUFFIX = openblas

CXX = g++
CXXFLAGS = -m64 -O3 -funroll-loops -std=c++11 
LDFLAGS =

# show all warning
CXXFLAGS += -Wall

# include MST algorithm
CXXFLAGS += -I${REPO_DIR}/edmonds-alg.1.1.2

#====================
# for OpenBLAS
#====================
CXXFLAGS += -ffast-math
CXXFLAGS += -I$(OPENBLAS_0_2_18_DIR)/include
LDFLAGS += -L$(OPENBLAS_0_2_18_DIR)/lib -lopenblas

#====================
# for boost
#====================
CXXFLAGS += -I$(BOOST_1_55_0_DIR)
LDFLAGS += -L$(BOOST_PROGRAM_OPTIONS_LIBDIR) -lboost_program_options

#====================
# for TIALNN
#====================
CXXFLAGS += -I$(TIALNN_0_0_1_DIR)/include
# Disable assert
CXXFLAGS += -DNDEBUG
# Use OpenBLAS (see tiann/include/base.h)
CXXFLAGS += -D_TIALNN_USE_OPENBLAS
# Use single precision
CXXFLAGS += -D_TIALNN_SINGLE_PRECISION