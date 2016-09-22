# BOOST_1_55_0_DIR = /g/tial/sw/pkgs/boost_1_55_0
# BOOST_PROGRAM_OPTIONS_LIBDIR = $(BOOST_1_55_0_DIR)/stage/lib
# MKLROOT should be defined using compilervars.sh
# MKLROOT = /g/tial/sw/pkgs/intel_parallel_studio_xe_2016/mkl
include ../memnet.mkl.mk

MEMNET_DIR = $(REPO_DIR)/src/memnet
CMEMNET_DIR = $(REPO_DIR)/src/cmemnet

all: train_cmemnet_dparser run_cmemnet_dparser

generic_vocab.o: $(MEMNET_DIR)/generic_vocab.h $(MEMNET_DIR)/generic_vocab.cpp
	$(CXX) $(CXXFLAGS) -c $(MEMNET_DIR)/generic_vocab.cpp

word_vocab.o: $(MEMNET_DIR)/word_vocab.h $(MEMNET_DIR)/word_vocab.cpp
	$(CXX) $(CXXFLAGS) -c $(MEMNET_DIR)/word_vocab.cpp

datapoint_reader.o: $(MEMNET_DIR)/datapoint_reader.h $(MEMNET_DIR)/datapoint_reader.cpp
	$(CXX) $(CXXFLAGS) -c $(MEMNET_DIR)/datapoint_reader.cpp

memnet_dparser_base.o: $(MEMNET_DIR)/memnet_dparser_base.h $(MEMNET_DIR)/memnet_dparser_base.cpp
	$(CXX) $(CXXFLAGS) -c $(MEMNET_DIR)/memnet_dparser_base.cpp

memnet_dparser_trainer.o: $(MEMNET_DIR)/memnet_dparser_trainer.h $(MEMNET_DIR)/memnet_dparser_trainer.cpp
	$(CXX) $(CXXFLAGS) -c $(MEMNET_DIR)/memnet_dparser_trainer.cpp

memnet_dparser_decoder.o: $(MEMNET_DIR)/memnet_dparser_decoder.h $(MEMNET_DIR)/memnet_dparser_decoder.cpp
	$(CXX) $(CXXFLAGS) -c $(MEMNET_DIR)/memnet_dparser_decoder.cpp

cmemnet_utils.o: $(CMEMNET_DIR)/cmemnet_utils.h $(CMEMNET_DIR)/cmemnet_utils.cpp
	$(CXX) $(CXXFLAGS) -c $(CMEMNET_DIR)/cmemnet_utils.cpp

cmemnet_dparser.o: $(CMEMNET_DIR)/cmemnet_dparser.h $(CMEMNET_DIR)/cmemnet_dparser.cpp
	$(CXX) $(CXXFLAGS) -c $(CMEMNET_DIR)/cmemnet_dparser.cpp

train_cmemnet_dparser: $(CMEMNET_DIR)/train_cmemnet_dparser.cpp \
	generic_vocab.o word_vocab.o datapoint_reader.o \
	memnet_dparser_base.o memnet_dparser_trainer.o \
	cmemnet_utils.o cmemnet_dparser.o
	$(CXX) $(CXXFLAGS) \
		$(CMEMNET_DIR)/train_cmemnet_dparser.cpp \
		generic_vocab.o word_vocab.o datapoint_reader.o \
		memnet_dparser_base.o memnet_dparser_trainer.o \
		cmemnet_utils.o cmemnet_dparser.o \
		$(LDFLAGS) \
		-o $(BIN_DIR)/train_cmemnet_dparser.$(BIN_SUFFIX)

run_cmemnet_dparser: $(CMEMNET_DIR)/run_cmemnet_dparser.cpp \
	generic_vocab.o word_vocab.o datapoint_reader.o \
	memnet_dparser_base.o memnet_dparser_decoder.o \
	cmemnet_utils.o cmemnet_dparser.o
	$(CXX) $(CXXFLAGS) \
		$(CMEMNET_DIR)/run_cmemnet_dparser.cpp \
		generic_vocab.o word_vocab.o datapoint_reader.o \
		memnet_dparser_base.o memnet_dparser_decoder.o \
		cmemnet_utils.o cmemnet_dparser.o \
		$(LDFLAGS) \
		-o $(BIN_DIR)/run_cmemnet_dparser.$(BIN_SUFFIX)

clean:
	rm -f *.o 
	rm -f $(BIN_DIR)/train_cmemnet_dparser.$(BIN_SUFFIX) 
	rm -f $(BIN_DIR)/run_cmemnet_dparser.$(BIN_SUFFIX)
