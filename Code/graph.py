import numpy as np
import scipy.stats as sps
from tqdm.notebook import tqdm
from collections import deque
rng = np.random.default_rng()


def get_close_vector(v, alpha):
    '''
    Parameters:
    v: numpy vector
    alpha: float, -1 <= alpha <= 1
    Generates a random unit vector u,
    such that <u, v> = alpha. 
    '''
    unit = sps.norm.rvs(size=v.size)
    unit /= np.linalg.norm(unit)
    orth = unit - np.dot(unit, v) * v
    orth /= np.linalg.norm(orth)
    return alpha * v + (1 - alpha) * orth



class RandomGraph:
    def __init__(self, N=1000, distr=sps.poisson(4)):
        self.N = N
        self.distr = distr
        self.noise = np.zeros(N)
        degrees = distr.rvs(size=N)
        if np.sum(degrees) % 2 == 1:
            degrees = np.insert(degrees, degrees.size, 1)
            self.N += 1
        half_edges = []
        for num, deg in enumerate(degrees):
            half_edges += [num] * deg
        self.edges = len(half_edges) // 2
        self.state = -np.ones(self.N)

        first_indices = np.sort(rng.choice(self.edges * 2, self.edges, replace=False))
        second_indices = []
        current_pos = 0
        for i in range(2 * self.edges):
            if current_pos < self.edges and first_indices[current_pos] == i:
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
       
    def set_fields(self, fields):
        self.fields = fields

    def sample_fields(self, distribution=sps.norm(loc=0, scale=1), multivariate=False):
        size = 1 if multivariate else self.N
        self.fields = distribution.rvs(size=size)
        
    def __bfs(self, root, alpha):
        q = deque()
        self.__visited[root] = True
        q.append((float('nan'), root))
        while q:
            parent, node = q.popleft()
            if np.isnan(parent):
                self.__vectors[node] = sps.norm.rvs(size=self.N)
                self.__vectors[node] /= np.linalg.norm(self.__vectors[node])
            else:
                self.__vectors[node] = get_close_vector(self.__vectors[parent], alpha)
            for neighbour in self.connections[node]:
                if not self.__visited[neighbour]:
                    self.__visited[neighbour] = True
                    q.append((node, neighbour))
    
    def __get_correlation_matrix(self, alpha):
        self.__visited = [False] * self.N
        self.__vectors = np.zeros((self.N, self.N))
        for i in range(self.N):
            if not self.__visited[i]:
                self.__bfs(i, alpha)
        return self.__vectors @ self.__vectors.T + 0.01 * np.eye(self.N)

    def __get_noise(self, noise_distr):
        self.noise = noise_distr.rvs(size=(2, self.N))

    def get_next_state(self, H, J, noise_distr=sps.bernoulli(p=0)):
        # self.__get_noise(noise_distr) Позже понадобится, но пока лишь замедляет вычисления
        utility = H + self.fields
        for agent in range(self.N):
            utility[agent] += J * np.sum(self.state[self.connections[agent]])
        temp = utility * np.array([[-1], [1]])
        self.state = 2 * np.argmax(utility * np.array([[-1], [1]]), axis=0) - 1
        return self.state
    
    def get_stable_state(self, H, J, log_hubs=False, quantile=float('nan')):
        initial_state = current_state = self.state
        next_state = self.get_next_state(H, J)
        if np.isnan(quantile):
            threshold = -1
        else:
            threshold = self.distr.ppf(quantile)
        while not np.allclose(current_state, next_state):
            current_state = next_state
            next_state = self.get_next_state(H, J)
        if log_hubs:
            changed_hubs = np.sum(np.abs(initial_state - self.state)) / 2
#             for i in range(self.N):
#                 if initial_state[i] != self.state[i] and len(self.connections[i]) > threshold:
#                     changed_hubs += 1
            return self.state, changed_hubs
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
    
    def get_trajectories(self, J, H_grid, parameter, distr_name: str, log_hubs=False, quantile=float('nan')):
        multivariate = False
        if distr_name == 'no_distr':
            distribution = sps.norm(loc=0, scale=0)
        elif distr_name == 'norm': 
            distribution = sps.norm(loc=0, scale=parameter)
        elif distr_name == 'student':
            distribution = sps.t(df=parameter)
        elif distr_name == 'multivariate_norm':
            multivariate = True
            distribution = sps.multivariate_normal(cov=self.__get_correlation_matrix(parameter))
        elif distr_name == 'multivariate_student':
            multivariate = True
            distribution = sps.multivariate_t(df=3.5, shape=3/7 * self.__get_correlation_matrix(parameter))
        else:
            raise NotImplementedError('Unknown distribution')
        fractions_low_to_high, fractions_high_to_low = [], []
        self.sample_fields(distribution, multivariate)
        self.set_state(-np.ones(self.N))
        if not log_hubs:
            for H in H_grid:
                new_state = self.get_stable_state(H, J)
                fractions_low_to_high.append(np.mean(new_state + 1) / 2)
            for H in np.flip(H_grid):
                new_state = self.get_stable_state(H, J)
                fractions_high_to_low.append(np.mean(new_state + 1) / 2)
            return fractions_low_to_high, fractions_high_to_low
        hubs_low_to_high, hubs_high_to_low = [], []
        for H in H_grid:
            new_state, changed_hubs = self.get_stable_state(H, J, log_hubs, quantile)
            fractions_low_to_high.append(np.mean(new_state + 1) / 2)
            hubs_low_to_high.append(changed_hubs)
        for H in np.flip(H_grid):
            new_state, changed_hubs = self.get_stable_state(H, J, log_hubs, quantile)
            fractions_high_to_low.append(np.mean(new_state + 1) / 2)
            hubs_high_to_low.append(changed_hubs)
        return fractions_low_to_high, fractions_high_to_low, hubs_low_to_high, hubs_high_to_low