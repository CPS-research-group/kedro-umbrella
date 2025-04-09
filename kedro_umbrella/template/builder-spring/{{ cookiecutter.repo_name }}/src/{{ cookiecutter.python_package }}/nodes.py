import numpy as np

def physics_model(k, F):
  # Define the matrix A as above

  ## first row (special as fixed to the wall)
  A = np.hstack(([1], [0] * len(k)))

  ## middle rows: 0 to k-1. Last in special as we apply the forces to it
  for i in range(len(k) - 1):
    head = np.array([0] * i)
    mid = np.array([ -k[i], k[i] + k[i + 1], -k[i + 1] ])
    trail = np.array([0] * (len(k) + 1 - len(mid) - i))
    mid_row = np.hstack((head, mid, trail))
    A = np.vstack((A, mid_row))

  ## last row
  last = np.hstack(([0] * (len(k) - 1), [ -k[len(k) - 1], k[len(k) - 1] ]))
  A = np.vstack((A, last))

  # Solve it
  b = np.hstack(([0] * len(k), F))
  u = np.linalg.solve(A, b)
  # force u[0] to be zero, it can be perturbed by FP errors
  u[0] = 0
  return u


def get_gaussian_data(N, T, F, avg, std):
# baseline system
    Kb = [1] * N # k=1
    Ub = physics_model(Kb, 1)

      # noisy system
    rng = np.random.default_rng(0)
    dK = np.zeros((T, N))
    Kn = np.zeros((T, N))
    Un = np.zeros((T, N + 1))  # force + noise
    for t in range(T): # for-each trial
        dK[t] = rng.normal(loc = avg, scale = std, size = N)
        Kn[t] = 1 + dK[t]
          #Kn[t] = 1 + dK[t] #k=1
        Un[t] = physics_model(Kn[t], F)
    return {'Kb': Kb, 'Ub': Ub, 'Kn': Kn, 'Un': Un}



def get_fourier_data(N, T, F, p, gaussian = ()):
    # baseline system
    Kb = [1] * N
    Ub = physics_model(Kb, 1)

    # noisy system
    X = np.linspace(0, 1, N + 1)
    Xbar = (X[0:-1] + X[1:]) / 2

    rng = np.random.default_rng(0)
    noise = lambda : rng.uniform(0.01, 0.1)
    dK = np.zeros((T, N))
    Kn = np.zeros((T, N))
    Un = np.zeros((T, N + 1))  # force + noise
    for t in range(T): # for-each trial
        for n in range(1, p): # for-each noise dim
            dK[t] += noise() * np.sin(n * Xbar * np.pi) + \
                     noise() * np.cos(n * Xbar * np.pi)
        Kn[t] = 1 + dK[t]
        if gaussian:
           Kn[t] += rng.normal(loc = gaussian[0], scale = gaussian[1], size = N)
        Un[t] = physics_model(Kn[t], F)
    return {'Kb': Kb, 'Ub': Ub, 'Kn': Kn, 'Un': Un}

def get_data(parameters):
    '''
    Return the X => Y pair to be used for training and testing
    Args:
      noise: noise type ('fourier', 'noisy_fourier', 'gaussian')
      N: number of dimensions
      T: number of trials
      data_xform: data transformation ('U_dK' or 'dU_dK' or 'dEps_dK')
    '''
    noise = parameters['noise']
    N = parameters['N']
    T = parameters['T']
    F = parameters['F']
    data_xform = parameters['data_xform']
    if noise == 'fourier':
        data = get_fourier_data(N, T, 10)
    elif noise == 'noisy_fourier':
        data = get_fourier_data(N, T, F, 10, (0, 0.03))
    elif noise == 'gaussian':
        data = get_gaussian_data(N, T, F, 0.06, 0.20)
    else:
        assert 0

    dK = data['Kn'] - data['Kb']

    if data_xform == 'U_dK': # U => dK
        return data['Un'], dK
    elif data_xform == 'dU_dK': # dU => dK
        dU = data['Un'] - data['Ub']
        return dU, dK
    elif data_xform == 'dEps_dK':
        dU = data['Un'] - data['Ub']
        eps_b = np.diff(data['Ub'])
        eps_n = np.diff(data['Un'])
        dEps = eps_n - eps_b
        return dEps, dK
    elif data_xform == 'log(dEps)_log(dK)':
        dU = data['Un'] - data['Ub']
        eps_b = np.diff(data['Ub'])
        eps_n = np.diff(data['Un'])
        dEps = eps_n - eps_b
        return np.log(dEps + 1), np.log(dK + 1)
    else:
        assert 0
