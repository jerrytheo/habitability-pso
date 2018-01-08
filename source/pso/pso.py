import numpy as np
from numpy.random import uniform

from ..utils import _round


# Exception raised when Swarm does not converge.
class SwarmConvergeError(Exception):
    """Error raised when the swarm does not converge."""
    pass


class Particle:

    """A single particle of the swarm."""

    def __init__(self, ndim, fn, min_=0, max_=1, friction=1.0, learn_rate1=2.0,
                 learn_rate2=2.0, max_velocity=None):
        self.c1 = learn_rate1
        self.c2 = learn_rate2
        self.fc = friction

        self.fitness = lambda k=1: fn(self.position, k)
        self.position = _round(uniform(min_, max_, ndim))
        self.velocity = uniform(-np.abs(max_-min_), np.abs(max_-min_), ndim)

        self.best = np.array(self.position)
        self.best_fitness = self.fitness()
        self.velocity_cap = max_velocity

    def update(self, global_best, iteration=1):
        """Calculate velocity and update current position."""
        dv_l = _round(self.c1 * uniform(0, 1))
        dv_l = _round(dv_l * (self.best - self.position))

        dv_g = _round(self.c2 * uniform(0, 1))
        dv_g = _round(dv_g * (global_best - self.position))

        self.velocity = _round(self.fc * self.velocity) + dv_l + dv_g
        check = np.abs(self.velocity) >= self.velocity_cap
        self.velocity[check] = np.sign(self.velocity[check]) * \
            self.velocity_cap

        self.position = _round(self.position + self.velocity)
        curr_fitness = self.fitness(iteration)
        if curr_fitness > self.best_fitness:
            self.best = np.array(self.position)
            self.best_fitness = curr_fitness


class Swarm:

    """The swarm for PSO."""

    def __init__(self, npart, *args, **kwargs):
        self.particles = [Particle(*args, **kwargs) for it in range(npart)]
        self.best_particle = max(self.particles, key=lambda p: p.best_fitness)
        self.global_best = self.best_particle.best_fitness

    def update(self, iteration):
        """Update all particles of the swarm."""
        for particle in self.particles:
            particle.update(self.global_best, iteration)
            if particle.best_fitness > self.global_best:
                self.best_particle = particle
                self.global_best = particle.best_fitness

    def converge(self, max_stable=50, max_iter=10000, threshold=0.0001,
                 verbose=True, proglen=20):
        """Find the maxima for the fitness function specified."""
        stable_count = 0
        for it in range(max_iter):
            old_best = self.global_best
            self.update(it)
            if np.abs(old_best - self.global_best) < threshold:
                stable_count += 1
                if stable_count == max_stable:
                    return it
            else:
                stable_count = 0
        raise SwarmConvergeError('swarm did not converge. stable_count='
                                 + str(stable_count))


# Function for convergence.
def converge(restarts=3, **kwargs):
    """Wait for convergence by PSO."""
    for _ in range(restarts):
        try:
            swarm = Swarm(**kwargs)
            converged = swarm.converge(verbose=False)
            return swarm, converged
        except SwarmConvergeError:
            pass
    raise SwarmConvergeError('no convergence after '
                             + restarts + ' restarts.')
