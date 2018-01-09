import numpy as np
import sys

from ..utils import _round, _uniform


# Exception raised when Swarm does not converge.
class SwarmConvergeError(Exception):
    """Error raised when the swarm does not converge."""
    pass


class Swarm2:
    def __init__(self, start_points, fitness_fn, friction=1.0, learnrate1=0.04,
                 learnrate2=0.16, max_velocity=.1):
        self.c1 = learnrate1
        self.c2 = learnrate2
        self.fc = friction
        self.fn = lambda x: _round(fitness_fn(x))
        self.cap = max_velocity

        self.P = _round(start_points)
        self.L = self.P.copy()
        self.V = _uniform(-self.cap, self.cap, self.P.shape)
        self.C = self.eval_constraints(self.P)

        self.fit_P = self.eval_fitness(self.P)
        self.fit_L = self.fit_P.copy()

        self.fit_g = self.fit_L[self.fit_L == np.max(self.fit_L)]
        self.g = self.L[self.fit_L == self.fit_g]
    
    def eval_constraints(self, positions):
        pass

    def eval_fitness(self, positions):
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
        self.local_bests = np.array([p.best for p in self.particles])
        self.best_particle = max(self.particles, key=lambda p: p.best_fitness)
        self.global_best = self.best_particle.best_fitness

    def update(self, iteration):
        """Update all particles of the swarm."""
        for _, particle in enumerate(self.particles):
            particle.update(self.best_particle.best, iteration)
            if particle.best_fitness > self.global_best:
                self.best_particle = particle
                self.global_best = particle.best_fitness
        #  print(self.particles[0].position - self.best_particle.best, end='\t\t')
        #  print(self.particles[0].best - self.best_particle.best, end='\t\t')
        #  print(self.particles[0].position)

    def converge(self, max_stable=100, max_iter=1000, threshold=1e-5):
        """Find the maxima for the fitness function specified."""
        stable_count = 0
        for it in range(max_iter):
            old_best = self.global_best
            self.update(it)
            if np.abs(old_best - self.global_best) < threshold:
                stable_count += 1
                if stable_count == max_stable:
                    #  sys.exit()
                    return it
            else:
                stable_count = 0
        raise SwarmConvergeError('swarm did not converge. stable_count='
                                 + str(stable_count))


# Function for convergence.
def conmax_by_pso(function, start_points, penalty, friction=.8, learnrate1=.1,
                  learnrate2=.1, max_velocity=1.):
    """Perform constrained maximization of the given function using particle
    swarm optimization.

    Arguments:
        function: function(positions) -> fitness_values
            The function to maximize. The function should take as an argument a
            ndarray of shape (N, D) and return an 1d array of size N.
        start_points: ndarray of shape (N, D)
            An array of points to begin PSO from, where N is the number of
            points, and D the dimensionality of each point.
        penalty: function(position) -> penalty_value
            The penalty function corresponding to the constraints. The function
            should take as an argument a ndarray of shape (N, D) and return an
            1d array of size N.
        friction: float, default 0.8
            The velocity is scaled by friction before updating.
                    velocity = friction*velocity + dv_g + dv_l
        learnrate1: float, default 0.1
            The global learning rate.
                    dv_g = learnrate1 * random(0, 1) * (gbest - current)
        learnrate2: float, default 0.1
            The local learning rate.
                    dv_l = learnrate2 * random(0, 1) * (lbest - current)
        max_velocity: float, default 1.0
            The threshold for velocity.
    Returns:
        a 2-tuple (swarm, it), where swarm is the converged particle swarm and
        it are the number of iterations taken to converge.
    """
    pass
