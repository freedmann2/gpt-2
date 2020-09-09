#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npz

import argparse
import numpy as np
import sys

from ftfy import fix_text

import tflex_utils
import tqdm

from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(
    description='Use FTFY to prepare a dataset for training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('infile', metavar='PATH', type=str, help='Input file, directory, or glob pattern (utf-8 text).')
parser.add_argument('--outfile', default="-", type=str, help='Output file path, or - for stdout')
parser.add_argument('--nproc', default=0, type=int, help='Number of processes to use')

def fix(line):
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
  return fixed

def each_lines(f):
  for i, line in tflex_utils.for_each_line(f):
    yield line

def main():
    args = parser.parse_args()
    out = sys.stdout if args.outfile == '-' else open(args.outfile, "w")
    i = 0
    nproc = args.nproc or cpu_count()
    pool = Pool(processes=nproc) if nproc > 1 else None
    mapper = pool.imap if nproc > 1 else map
    lines = []
    with open(args.infile, encoding='utf-8') as f:
      for i, line in tflex_utils.for_each_line(f):
        lines.append(line)
        if len(lines) > 1000:
          for line in tqdm.tqdm(mapper(fix, lines), total=len(lines)):
            out.write(line)
          lines = []
        i += 1
        if i % 100 == 0:
          out.flush()
      for line in tqdm.tqdm(mapper(fix, lines), total=len(lines)):
        out.write(line)

if __name__ == '__main__':
    main()
