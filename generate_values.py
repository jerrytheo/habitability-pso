#!/usr/bin/python

from functools import partial
from sys import argv

from source import exoplanets
from source import evaluate_cdhs_values
from source import evaluate_ceesa_values


# Parameters for the swarm.
pso_params = {
    'npart': 25,                        # Number of particles.
    'friction': .4,                     # Friction coefficient.
    'learnrate1': .06,                  # c1 learning rate.
    'learnrate2': .14,                  # c2 learning rate.
    'max_velocity': .3,                 # Max. velocity.
}


# Help text for the script.
help_text = """
USAGE: ./generate_values.py [-h] [--help] [-v] [--verbose] [--debug]
                            [--score <scorename>]
                            [--multiple <param> <start> <stop> [<step>]]
Generate the CDHS and CEESA score for exoplanets from the PHL-EC dataset.

OPTIONAL ARGUMENTS:
    --help
        Display this help message.
    --score <scorename>
        Generate score only for <scorename>. Can be either "cdhs" or "ceesa".
    --multiple <param> <start> <stop> [<step>]
        Generate the scores over multiple iterations by varying the parameter
        specified by <param> from <start> to <stop> by <step>. <step> is 1 by
        default. <param> may be "npart", "friction", "learnrate1", "learnrate2"
        or "max_velocity".
    --debug
        Run everything on only 10 random exoplanets. (Could include nan that
        will be removed.)
"""

evaluate = {
        'cdhs': evaluate_cdhs_values,
        'ceesa': evaluate_ceesa_values,
        }


# Handle arguments.
args = argv[1:]
single = True
verbose = False
invalid = 'Invalid usage.\n' + help_text
fname_debug = ''

try:
    while args:
        argname = args.pop(0)

        # --score <scorename>
        if argname == '--score':
            score = args.pop(0)
            if score == 'ceesa':
                del evaluate['cdhs']
            elif score == 'cdhs':
                del evaluate['ceesa']
            else:
                print(invalid)

        # --multiple <param> <start> <stop> [<step>]
        elif argname == '--multiple':
            single = False
            param = args.pop(0)
            start = int(args.pop(0))
            stop = int(args.pop(0))
            if args and not args[0].startswith('-'):
                step = int(args.pop(0))
            else:
                step = 1

        # --help or -h
        elif argname in ['--help', '-h']:
            print(help_text)

        # --verbose or -v
        elif argname in ['--verbose', '-v']:
            verbose = True

        # --debug
        elif argname == '--debug':
            exoplanets = exoplanets.sample(30)
            debug = '_sample'
        else:
            print(invalid)

except (IndexError, ValueError):
    print(invalid)


try:
    for score, fn in evaluate.items():
        fn = partial(fn, exoplanets, verbose=verbose)
        if single:                                              # Aww...
            fname = '{score}_{{0}}{debug}.csv'.format(sc=score, db=debug)
            fn(fname=fname, **pso_params)
        else:
            for pso_params[param] in range(start, stop+step, step):
                fname = '{sc}_{{0}}_{pm}_{vl}{db}.csv'.format(
                        sc=score, pm=param, vl=pso_params[param], db=debug)
                fn(fname=fname, **pso_params)
except KeyboardInterrupt:
    print('\nGood bye!')
