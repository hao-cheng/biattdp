#!/usr/bin/env python

"""
Format the decode output from run_cmemnet_dparser.
The raw output does not contain the original word and postag.
"""

import os
import sys
import argparse
from itertools import izip

def main():
    cmdline_parser = argparse.ArgumentParser(
        description=__doc__
    )
    cmdline_parser.add_argument(
        'golden_conllx',
        help='golden data in conll-x format'
    )
    cmdline_parser.add_argument(
        'decode_conllx',
        help='raw decoded data in conll-x format'
    )
    cmdline_parser.add_argument(
        'out_conllx',
        help='formatted decoded data in conll-x format'
    )
    args = cmdline_parser.parse_args()

    fp = open(args.out_conllx, 'w')
    for line_g, line_d in izip(open(args.golden_conllx), open(args.decode_conllx)):
        units_g = line_g.rstrip().split('\t')
        units_d = line_d.rstrip().split('\t')

        if len(units_g) == 1:
            fp.write('\n')
        else:
            assert len(units_g) == 10
            assert len(units_d) == 10

            units_g[6] = units_d[6]
            units_g[7] = units_d[7]
            fp.write('\t'.join(units_g))
            fp.write('\n')
    fp.close()

if __name__ == '__main__':
    main()
