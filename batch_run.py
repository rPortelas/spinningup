import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('scripts', nargs='+', help='list of scripts to launch')
parser.add_argument('repeats', type=int, default=None)

args = parser.parse_args()

for script in args.scripts:
    for r in range(args.repeats):
        os.system("./{} {}".format(script,r))