/*!
 * \file datapoint_token.h
 *
 * 
 * \version 0.0.6
 */

#ifndef MEMNET_DATAPOINT_TOKEN_H_
#define MEMNET_DATAPOINT_TOKEN_H_

// tialnn::IndexType
#include <tialnn/base.h>

namespace memnet {

struct DataPointToken {
  //! Constructors.
  DataPointToken() : token_position(0), form(0), lemma(0),
    cpos_tag(0), pos_tag(0), headword_position(0), dependency(0) {}
  DataPointToken(tialnn::IndexType tp,
                 tialnn::IndexType fm,
                 tialnn::IndexType hp,
                 tialnn::IndexType dp) : token_position(tp), form(fm), lemma(0),
    cpos_tag(0), pos_tag(0), headword_position(hp), dependency(dp) {}
  //! Destructor.
  ~DataPointToken() {}

  //! Position of current token.
  tialnn::IndexType token_position;
  //! Form.
  tialnn::IndexType form;
  //! Lemma.
  tialnn::IndexType lemma;
  //! Coarse-grained POS tag.
  tialnn::IndexType cpos_tag;
  //! Fined-grained POS tag.
  tialnn::IndexType pos_tag;
  //! Extra features.
  std::vector<tialnn::IndexType> feats;
  //! Position of current word's head word.
  tialnn::IndexType headword_position;
  //! Dependency of word->head_word
  tialnn::IndexType dependency;
};

} // namespace memnet

#endif
