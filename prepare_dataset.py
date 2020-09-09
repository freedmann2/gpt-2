#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npz

import argparse
import numpy as np
import sys

from ftfy import fix_text

import tflex_utils

parser = argparse.ArgumentParser(
    description='Use FTFY to prepare a dataset for training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('infile', metavar='PATH', type=str, help='Input file, directory, or glob pattern (utf-8 text).')
parser.add_argument('--outfile', default="-", type=str, help='Output file path, or - for stdout')

def main():
    args = parser.parse_args()
    out = sys.stdout if args.outfile == '-' else open(args.outfile, "w")
    i = 0
    with open(args.infile) as f:
      for i, line in tflex_utils.for_each_line(f):
        fixed = fix_text(line)
        # replace unicode … with ... which ftfy doesn't do by default
        # NOTE: this departs from openai's convention of calling
        # ftfy.fix_text() with default arguments. In particular,
        # OpenAI's GPT-2 models do generate unicode ellipses.
        # Nonetheless, we replace unicdoe ellipses with ... to
        # increase the chances of semantic understanding.
        fixed = fixed.replace(' …', '...') # first pass: convert "foo  …" to "foo..."
        #fixed = fixed.replace(' …', '...') # second pass: convert "foo …" to "foo..."
        fixed = fixed.replace('…', '...') # final pass: convert "foo…" to "foo..."
        out.write(fixed)
        i += 1
        if i % 100 == 0:
          out.flush()

if __name__ == '__main__':
    main()
