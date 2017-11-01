from numpy import abs as abs_
from numpy.random import uniform


class Particle:

    """A single particle of the swarm."""

    def __init__(self, ndim, fitness_function, min_=0, max_=1,
                 learn_rate1=2.0, learn_rate2=2.0, max_velocity=None):

        self.fitness = lambda: fitness_function(self.position)
        self.position = uniform(min_, max_, ndim)
        self.velocity = uniform(-abs_(max_-min_), abs_(max_-min_), ndim)

        self.best = self.position
        self.best_fitness = self.fitness()
        self.velocity_cap = max_velocity

        self.c1 = learn_rate1
        self.c2 = learn_rate2

    def update(self, global_best):
        """Calculate velocity and update current position."""
        dv_l = self.c1 * uniform(0, 1) * (self.best - self.position)
        dv_g = self.c2 * uniform(0, 1) * (global_best - self.position)

        self.velocity += dv_l + dv_g
        check = self.velocity >= self.velocity_cap
        self.velocity[check] = self.velocity_cap

        self.position += self.velocity
        curr_fitness = self.fitness()
        if curr_fitness > self.best_fitness:
            self.best = self.position
            self.best_fitness = curr_fitness


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

    def converge(self, max_iter=10000, threshold=0.00001, verbose=True):
        """Find the maxima for the fitness function specified."""
        for it in range(max_iter):
            progress = round((it/max_iter) * 20)
            old_best = self.global_best
            self.update()
            if abs_(old_best - self.global_best) < threshold:
                break
            print(it)
            if verbose:
                print('Iterations: [',
                      '=' * (progress-1),
                      '>',
                      ' ' * (20-progress),
                      ']',
                      end='\r', sep='')
        if verbose:
            print('Iterations: [',
                  '=' * (progress-1),
                  '#',
                  ' ' * (20-progress),
                  ']',
                  end='\n', sep='')
