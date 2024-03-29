import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from collections import deque
rng = np.random.default_rng()

import graph_models as gm


def plot_trajectories(index, max_index, beta, H_grid, low_to_high, high_to_low, hubs_low=[], hubs_high=[], name=r'\beta'):
    hubs_low = np.array(hubs_low)
    high_to_low, hubs_high = np.flip(high_to_low), np.flip(hubs_high)
    left, right = np.argmax(high_to_low >= 0.5), np.argmax(np.array(low_to_high) >= 0.5)
    diff = (left + right) // 2 - len(H_grid) // 2
    step = H_grid[1] - H_grid[0]
    if diff > 0:
        H_grid = np.hstack((H_grid[diff:], np.linspace(H_grid[-1] + step, H_grid[-1] + step * diff, diff)))
        low_to_high = np.hstack((low_to_high[diff:], [1] * diff))
        high_to_low = np.hstack((high_to_low[diff:], [1] * diff))
        hubs_low = np.hstack((hubs_low[diff:], [0] * diff))
        hubs_high = np.hstack((hubs_high[diff:], [0] * diff))
    elif diff < 0:
        diff *= -1
        H_grid = np.hstack((np.linspace(H_grid[0] - step * diff, H_grid[0] - step, diff), H_grid[:-diff]))
        low_to_high = np.hstack(([0] * diff, low_to_high[:-diff]))
        high_to_low = np.hstack(([0] * diff, high_to_low[:-diff]))
        hubs_low = np.hstack(([0] * diff, hubs_low[:-diff]))
        hubs_high = np.hstack(([0] * diff, hubs_high[:-diff]))
        
    left_hubs_line, right_hubs_line = np.argmax(hubs_low > 0), len(H_grid) - np.argmax(np.flip(hubs_low) > 0)
    
    plt.subplot(max_index, 2, 2 * index + 1)
    plt.plot(H_grid, low_to_high, label='increasing H')
    plt.plot(H_grid, high_to_low, label='decreasing H')
    plt.grid(ls=':')
    plt.xlabel('H', fontsize='large')
    plt.title(r'Percentage of vertices with $\sigma_i = 1$, ${1} = {0}$'.format(round(beta, 3), name))
    plt.legend()
    
    plt.subplot(max_index, 2, 2 * index + 2)
    plt.bar(np.arange(H_grid.size), hubs_low, width=5, label='increasing H')
    plt.plot([left_hubs_line - 3] * 2, [0, hubs_low.max()], linestyle='dashed', color='red')
    plt.plot([right_hubs_line + 3] * 2, [0, hubs_low.max()], linestyle='dashed', color='green')
    plt.xticks(np.linspace(0, H_grid.size, 6), np.linspace(H_grid.min(), H_grid.max(), 6).round(2))
    plt.grid(ls=':')
    plt.xlabel('H', fontsize='large')
    plt.title(r'The number of switched vertices, ${1} = {0}$'.format(round(beta, 3), name))
    

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
    return alpha * v + np.sqrt(1 - alpha ** 2) * orth



class RandomGraph:
    def __init__(self, N=1000, distr=sps.poisson(4), topology='random'):
        self.N = N
        self.distr = distr
        self.noise = np.zeros(N)
        self.state = -np.ones(self.N)
        self.topology = topology
        if topology == 'complete':
            self.connections = gm.complete(self.N)
            return
        elif topology == 'star':
            self.connections = gm.star(self.N)
            return
        elif topology == 'circle':
            self.connections = gm.circle(self.N)
            return
        elif topology == 'cayley_tree':
            self.connections = gm.cayley_tree(self.N)
            return
        elif topology == 'regular':
            # регулярный граф степени 3
            self.distr = sps.randint(3, 4)
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
        if self.topology == 'complete':
            utility += J * (self.state.sum() - self.state)
        else:
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
            # changed_hubs = np.sum(np.abs(initial_state - self.state)) / 2
            changed_hubs = 0
            for i in range(self.N):
                if initial_state[i] != self.state[i] and len(self.connections[i]) > threshold:
                    changed_hubs += 1
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
    
    def get_trajectories(self, J, H_grid, parameter, distr_name: str, log_hubs=False, quantile=float('nan'), matrix='hard'):
        multivariate = False
        if distr_name == 'no_distr':
            distribution = sps.norm(loc=0, scale=0)
        elif distr_name == 'norm': 
            distribution = sps.norm(loc=0, scale=parameter)
        elif distr_name == 'student':
            distribution = sps.t(df=parameter)
        elif distr_name == 'multivariate_norm':
            multivariate = True
            if matrix == 'simple':
                cov_matrix = (1 - parameter) * np.eye(self.N) + parameter * np.ones((self.N, self.N))
                distribution = sps.multivariate_normal(cov=cov_matrix)
            else:
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