#!/usr/bin/python

from functools import partial
from sys import argv

from source import exoplanets
from source import evaluate_cdhs_values
from source import evaluate_ceesa_values


# Parameters for the swarm.
pso_args = {
    'npart': 100,                   # Number of particles.
    'ndim': 2,                      # Dimensions of input.
    'min_': 0,                      # Min. value for initial pos.
    'max_': 1,                      # Max. value for initial pos.
    'friction': .4,                 # Friction coefficient.
    'learn_rate1': 1e-1,            # c1 learning rate.
    'learn_rate2': 1e-1,            # c2 learning rate.
    'max_velocity': .1,             # Max. velocity.
}


# Help text for the script.
help_text = '''
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
        default. <param> may be "npart", "min_", "max_", "friction",
        "learn_rate1", "learn_rate2" or "max_velocity".
    --debug
        Run everything on only 10 random exoplanets. (Could include nan that
        will be removed.)
'''

evaluate = {
        'cdhs': evaluate_cdhs_values,
        'ceesa': evaluate_ceesa_values,
        }


# Handle arguments.
args = argv[1:]
single = True
verbose = False
invalid = 'Invalid usage.\n' + help_text

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
            exoplanets = exoplanets.sample(10)
        else:
            print(invalid)

except (IndexError, ValueError):
    print(invalid)


try:
    for k, fn in evaluate.items():
        fn = partial(fn, exoplanets, verbose=verbose)
        if single:                                          # Aww...
            fn(k + '_{0}.csv', pso_args)
        else:
            for pso_args[param] in range(start, stop+step, step):
                fname = k + '_{0}_' + param + str(pso_args[param]) + '.csv'
                fn(fname, pso_args)
except KeyboardInterrupt:
    print('\n\nGood bye!')
