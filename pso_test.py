#!/usr/bin/python

import numpy as np
from pso import Swarm

swarm = Swarm(10, ndim=1, fn=lambda x: -(x**2), friction=0.7,
              learn_rate1=.5, learn_rate2=.5, max_velocity=4.0)
swarm.converge()
print(np.round(swarm.best_particle.position, 4))
