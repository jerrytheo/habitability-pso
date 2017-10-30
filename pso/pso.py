import numpy as np


class Particle:
    """A single particle of the swarm."""

    def __init__(self, ndim, fitness_function, learn_rate1=2.0,
                 learn_rate2=2.0, max_velocity=None):
        self.curr_fitness = 0.0
        self.best_fitness = 0.0

        self.fitness = lambda: fitness_function(self.position)
        self.position = np.random.rand(ndim)

        self.velocity = 0
        self.velocity_cap = max_velocity

        self.c1 = learn_rate1
        self.c2 = learn_rate2

    def update(self):
        """Calculate velocity and update current position."""

        dv_1 = self.c1 * np.random.rand() * (self.curr_fitness - self.position)
        dv_2 = self.c2 * np.random.rand() * (self.best_fitness - self.position)
        self.velocity += (dv_1 + dv_2)
        if self.velocity >= self.velocity_cap:
            self.velocity = self.velocity_cap
        self.position += self.velocity

        self.curr_fitness = self.fitness()
        if self.curr_fitness > self.best_fitness:
            self.best_fitness = self.curr_fitness


class Swarm:
    """The swarm for PSO."""

    def __init__(self, npart, ndim, fitness_function, learn_rate1=2.0,
                 learn_rate2=2.0, max_velocity=None):
        self.particles = [
            Particle(ndim, fitness_function, learn_rate1, learn_rate2,
                     max_velocity) for it in range(npart)
        ]

    def __iter__(self):
        """Iterate over the particles."""
        for p in self.particles:
            yield p

    def __len__(self):
        """Return number of particles."""
        return len(self.particles)

    @property
    def global_best(self):
        """Returns the globally best fitness value."""
        max(self, key=lambda p: p.best_fitness)

    @property
    def best_particle(self):
        """Returns the first particle found with global_best fitness."""
        gbest = self.global_best
        for p in self:
            if p.best_fitness == gbest:
                return p

    def update(self):
        """Update all particles of the swarm."""
        for p in self:
            p.update()
