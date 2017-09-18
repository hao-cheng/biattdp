#!/usr/bin/env python

import os
import sys
import argparse
from collections import defaultdict

pa = argparse.ArgumentParser(
    description='sort CONLL-X data by length'
)
pa.add_argument('orig_conllx', help='original data in CoNLL-X format')
pa.add_argument('out_conllx', help='output data in CoNLL-X format')
args = pa.parse_args()

dp_for_len = defaultdict(list)
list_nwords = []

out_lines = []
num_samples = 0
for line in open(args.orig_conllx):
    units = line.rstrip().split('\t')
    if len(units) == 1:
        num_samples += 1
        dp_for_len[len(out_lines)].append(out_lines)
        list_nwords.append(len(out_lines))
        out_lines = []
    else:
        assert len(units) == 10
        out_lines.append(line.rstrip())
assert len(out_lines) == 0
print >> sys.stderr, \
        'In total {0} samples'.format(num_samples)

list_nwords = list(set(list_nwords))
list_nwords = sorted(list_nwords, reverse=False)
print >> sys.stderr, \
        'Max nwords:', list_nwords[-1]

fp = open(args.out_conllx, 'w')
for nwords in list_nwords:
    for dp in dp_for_len[nwords]:
        for line in dp:
            fp.write(line)
            fp.write('\n')
        fp.write('\n')
fp.close()
