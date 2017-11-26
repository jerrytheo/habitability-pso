import numpy as np
from numpy.random import uniform


class Particle:

    """A single particle of the swarm."""

    def __init__(self, ndim, fn, min_=0, max_=1, friction=1.0, learn_rate1=2.0,
                 learn_rate2=2.0, max_velocity=None):
        self.c1 = learn_rate1
        self.c2 = learn_rate2
        self.fc = friction

        self.fitness = lambda: fn(self.position)
        self.position = np.round(uniform(min_, max_, ndim), 1)
        self.velocity = uniform(-np.abs(max_-min_), np.abs(max_-min_), ndim)

        self.best = np.array(self.position)
        self.best_fitness = self.fitness()
        self.velocity_cap = max_velocity

    def update(self, global_best):
        """Calculate velocity and update current position."""
        dv_l = self.c1 * uniform(0, 1) * (self.best - self.position)
        dv_g = self.c2 * uniform(0, 1) * (global_best - self.position)

        self.velocity = self.fc * self.velocity + (dv_l + dv_g)
        check = np.abs(self.velocity) >= self.velocity_cap
        self.velocity[check] = np.sign(self.velocity[check]) * \
            self.velocity_cap

        self.position = np.round(self.position + self.velocity, 1)
        curr_fitness = self.fitness()
        if curr_fitness > self.best_fitness:
            self.best = np.array(self.position)
            self.best_fitness = curr_fitness
        # print(self.best, '\t\t', self.best_fitness)


class Swarm:

    """The swarm for PSO."""

    def __init__(self, npart, *args, **kwargs):
        self.particles = [Particle(*args, **kwargs) for it in range(npart)]
        self.best_particle = max(self, key=lambda p: p.best_fitness)
        self.global_best = self.best_particle.best_fitness

    def __iter__(self):
        """Iterate over the particles."""
        for particle in self.particles:
            yield particle

    def __len__(self):
        """Return number of particles."""
        return len(self.particles)

    def update(self):
        """Update all particles of the swarm."""
        for particle in self:
            particle.update(self.global_best)
            if particle.best_fitness > self.global_best:
                self.best_particle = particle
                self.global_best = particle.best_fitness

    def converge(self, max_stable=50, max_iter=10000, threshold=0.0001,
                 verbose=True, proglen=20):
        """Find the maxima for the fitness function specified."""
        stable_count = 0
        for it in range(max_iter):
            percent = round(it / max_iter * 100)
            progress = (it*proglen)//max_iter + 1
            old_best = self.global_best
            self.update()
            if np.abs(old_best - self.global_best) < threshold:
                stable_count += 1
                if stable_count == max_stable:
                    return True
            else:
                stable_count = 0
            if verbose:
                print('[', '=' * (progress), ' ' * (proglen-progress), ']',
                      '  ({:3}%)'.format(percent), end='\r', sep='')
        return False
