import numpy as np
import scipy.stats as sps
from tqdm.notebook import tqdm
rng = np.random.default_rng()


class RandomGraph:
    def __init__(self, N=1000, distr=sps.poisson(5)):
        self.N = N
        self.distr = distr
        degrees = distr.rvs(size=N)
        if np.sum(degrees) % 2 == 1:
            degrees = np.insert(degrees, degrees.size, 1)
            self.N += 1
        half_edges = []
        for num, deg in enumerate(degrees):
            half_edges += [num] * deg
        self.edges = len(half_edges) // 2
        self.state = -np.ones(self.N)

        first_indices = rng.choice(self.edges * 2, self.edges)
        second_indices = []
        current_pos = 0
        for i in range(2 * self.edges):
            if first_indices[current_pos] == i:
                current_pos += 1
            else:
                second_indices.append(i)
        permutation = rng.permutation(self.edges)
        self.connections = [[] for _ in range(self.N)]
        for i, index in enumerate(first_indices):
            v_first, v_second = half_edges[index], half_edges[second_indices[permutation[i]]]
            self.connections[v_first].append(v_second)
            self.connections[v_second].append(v_first)
            
    def print_connections(self):
        for i in range(min(self.N, 10)):
            print(i, ": ", sep='', end='')
            for neighbour in self.connections[i]:
                print(neighbour, end=' ')
            print()
            
    def set_state(self, state):
        self.state = state

    def sample_fields(self, distribution=sps.norm(loc=0, scale=1), multivariate=False):
        size = 1 if multivariate else self.N
        self.fields = distribution.rvs(size=size)

    def __get_noise(self, noise_distr):
        self.noise = noise_distr.rvs(size=(2, self.N))

    def get_next_state(self, H, J, noise_distr=sps.bernoulli(p=0)):
        self.__get_noise(noise_distr)
        utility = H + self.fields
        for agent in range(self.N):
            utility[agent] += J * np.sum(self.state[self.connections[agent]])
        temp = utility * np.array([[-1], [1]]) + self.noise
        self.state = 2 * np.argmax(utility * np.array([[-1], [1]]) + self.noise, axis=0) - 1
        return self.state
    
    def get_stable_state(self, H, J):
        current_state = self.state
        next_state = self.get_next_state(H, J)
        while not np.allclose(current_state, next_state):
            current_state = next_state
            next_state = self.get_next_state(H, J)
        return self.state
    
    def __get_states(self, H, J):
        self.set_state(-np.ones(self.N))
        state_first = self.get_stable_state(H, J)
        self.set_state(np.ones(self.N))
        state_second = self.get_stable_state(H, J)
        if np.allclose(state_first, state_second):
            return (state_first, )
        return (state_first, state_second)
    
    def get_equilibria(self, H, J):
        states = self.__get_states(H, J)
        if len(states) == 1:
            if np.mean(states[0]) == -1:
                return 1
            if np.mean(states[0]) == 1:
                return 2
            return 3
        first_part, second_part = np.mean(states[0]), np.mean(states[1])
        if first_part == -1 and second_part == 1:
            return -1
        if first_part == -1:
            return -2
        if second_part == 1:
            return -3
        return -4
    
    
def get_trajectories(G, J, H_grid, parameter, distr_name: str):
    multivariate = False
    if distr_name == 'norm': 
        distribution = sps.norm(loc=0, scale=parameter)
    elif distr_name == 'student':
        distribution = sps.t(df=parameter)
    elif distr_name == 'multivariate_norm':
        distribution = sps.multivariate_normal(np.zeros(G.N), parameter)
        multivariate = True
    else:
        raise NotImplementedError('Unknown distribution')
    fractions_low_to_high, fractions_high_to_low = [], []
    G.sample_fields(distribution, multivariate)
    G.set_state(np.zeros(G.N))
    for H in H_grid:
        new_state = G.get_stable_state(H, J)
        fractions_low_to_high.append(np.mean(new_state + 1) / 2)
    for H in np.flip(H_grid):
        new_state = G.get_stable_state(H, J)
        fractions_high_to_low.append(np.mean(new_state + 1) / 2)
    return fractions_low_to_high, fractions_high_to_low