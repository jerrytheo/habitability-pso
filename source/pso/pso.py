import numpy as np
import sys

from ..utils import _round, _uniform


# Exception raised when Swarm does not converge.
class SwarmConvergeError(Exception):
    """Error raised when the swarm does not converge."""
    pass


class Particle:

    """A single particle of the swarm."""

    def __init__(self, pos, fn, friction=1.0, lr1=2.0, lr2=2.0,
                 max_velocity=1.):
        self.c1 = lr1
        self.c2 = lr2
        self.fc = friction
        self.velocity_cap = max_velocity

        self.best = self.position = _round(pos)
        self.velocity = _uniform(-max_velocity, max_velocity,
                                 self.position.size)

        self.fitness = lambda k=1: fn(self.position, k)
        self.best_fitness = self.fitness()

    def update(self, global_best, iteration=1):
        """Calculate velocity and update current position."""
        co_l = _round(self.c1 * _uniform(0, 1))
        dv_l = _round(co_l * (self.best - self.position))

        co_g = _round(self.c2 * _uniform(0, 1))
        dv_g = _round(co_g * (global_best - self.position))

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

    def __init__(self, positions, *args, **kwargs):
        self.particles = [Particle(pos, *args, **kwargs) for pos in positions]
        self.best_particle = max(self.particles, key=lambda p: p.best_fitness)
        self.global_best = self.best_particle.best_fitness

    def update(self, iteration):
        """Update all particles of the swarm."""
        for particle in self.particles:
            particle.update(self.best_particle.best, iteration)
            if particle.best_fitness > self.global_best:
                self.best_particle = particle
                self.global_best = particle.best_fitness
        print(self.particles[0].position - self.best_particle.best, end='\t\t')
        print(self.particles[0].best - self.best_particle.best, end='\t\t')
        print(self.particles[0].position)

    def converge(self, max_stable=100, max_iter=1000, threshold=1e-5):
        """Find the maxima for the fitness function specified."""
        stable_count = 0
        for it in range(max_iter):
            old_best = self.global_best
            self.update(it)
            if np.abs(old_best - self.global_best) < threshold:
                stable_count += 1
                if stable_count == max_stable:
                    sys.exit()
                    return it
            else:
                stable_count = 0
        raise SwarmConvergeError('swarm did not converge. stable_count='
                                 + str(stable_count))


# Function for convergence.
def converge(pts, fn, pso_params, restarts=3):
    """Wait for convergence by PSO."""
    for _ in range(restarts):
        try:
            swarm = Swarm(pts, **pso_params)
            converged = swarm.converge()
            return swarm, converged
        except SwarmConvergeError:
            pass
    raise SwarmConvergeError('no convergence after '
                             + restarts + ' restarts.')
