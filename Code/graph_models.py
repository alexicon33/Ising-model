def complete(N):
    return [[j for j in range(N) if j != i] for i in range(N)]
    

def star(N):
    res = [[] for _ in range(N)]
    res[0] = list(range(1, N))
    for i in range(1, N):
        res[i].append(0)
    return res
    

def circle(N):
    res = [[] for _ in range(N)]
    for i in range(N - 1):
        res[i].append(i + 1)
        res[i + 1].append(i)
    res[N - 1].append(0)
    return res
    
    
def cayley_tree(N):
    res = [[] for _ in range(N)]
    res[0] += [1, 2, 3]
    res[1].append(0)
    res[2].append(0)
    res[3].append(0)
    for i in range(1, N):
        if 2 * i + 2 < N:
            res[i].append(2 * i + 2)
            res[2 * i + 2].append(i)
        if 2 * i + 3 < N:
            res[i].append(2 * i + 3)
            res[2 * i + 3].append(i)
            
    return res
    