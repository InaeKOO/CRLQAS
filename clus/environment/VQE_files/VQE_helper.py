import jax.numpy as jnp
import jax
from jax import jit
import torch
import numpy as np
from scipy.optimize import OptimizeResult
import pickle
from typing import List, Callable, Optional
from functools import partial

def putangles(angles,circ_instrs, rpos):
    circ_instrs[rpos,1] = angles
    return circ_instrs, jnp.array(circ_instrs)

def rotate_4q_actnum(rot_qub, rot_dir):
    return 3*rot_qub + rot_dir
def cx_4q_actnum(ctrl,targ):
    actnum = 12
    if ctrl ==0:
        if targ == 1:
            actnum += 0
        elif targ == 2:
            actnum += 1
        elif targ == 3:
            actnum += 2
    elif ctrl == 1:
        actnum += 3
        if targ == 0:
            actnum += 0
        elif targ == 2:
            actnum += 1
        elif targ == 3:
            actnum += 2
    elif ctrl == 2:
        actnum += 6
        if targ == 0:
            actnum += 0
        elif targ == 1:
            actnum += 1
        elif targ == 3:
            actnum += 2
    elif ctrl == 3:
        actnum += 9
        if targ == 0:
            actnum += 0
        elif targ == 1:
            actnum += 1
        elif targ == 2:
            actnum += 2

    return actnum

@jit
def get_instr( two_array ):
    return two_array[0].astype(int), two_array[1]

def shot_noise_np(weights, sigma):
    return np.real(weights.T @ np.random.normal(0, sigma, len(weights)))

def grad_shot_noise_np(weights, sigma, maxiter):
    fev_pl_shot_noise = np.real(weights.T @ np.random.normal(0, sigma, (len(weights),maxiter)))
    fev_min_shot_noise = np.real(weights.T @ np.random.normal(0, sigma, (len(weights),maxiter)))
    return jnp.array( fev_pl_shot_noise - fev_min_shot_noise )

def load_ham_dict(name):
    with open(name, 'rb') as handle:
        dictVar = pickle.load(handle)

    return dictVar


@jax.jit
def final_energy(st, Hamil, energy_shift):
    return opexpect(st, Hamil) + energy_shift

@jax.jit
def final_energy_ptm(rho_ptm, Hamil, energy_shift):
    return opexpect_ptm(rho_ptm, Hamil) + energy_shift


@jit
def apply_unitary(uni, dm):
    return uni @ dm @ uni.conj().Tdef
    apply


@jit
def opexpect(state, Ham):
    return jnp.real(state.conj().T @ Ham @ state)[0][0]

@jit
def opexpect_ptm(rho_ptm, Ham):
    return jnp.real(Ham @ rho_ptm)



@jit
def virtual_distill(rho, Ham):
    return opexpect_dm(rho @ rho, Ham) / purity(rho)


@jit
def opexpect_dm(op1, op2):
    return jnp.real(jnp.trace(op1 @ op2))


@jit
def purity(op):
    return opexpect_dm(op, op)


def min_zero_layer(A):
    l_min = A.size(0)  

    matrix_sum = torch.sum(torch.sum(A, dim=1), dim=1)

    zero_idx = torch.nonzero(matrix_sum == 0, as_tuple=True)[0]

    if zero_idx.numel() > 0:
        l_min = zero_idx[0]  

    return int(l_min)


def get_num_actionable_gates(n):
    tmp = 3 * n
    tmp += n * (n - 1)
    return tmp


def tuple2len(tpl):
    ln = 0
    for ary in tpl:
        ln += len(ary)

    return ln


def tuples2jnp(tpl):
    lst = []
    for tp in tpl:
        lst.append(jnp.array(tp))

    return tuple(lst)


def find_maxnum_rots_cnots(self, state):
    mnr = 0
    mnc = 0
    for l, local_state in enumerate(state):
        curr_nr = len((local_state[self.num_qubits: self.num_qubits + 3] == 1).nonzero(as_tuple=True)[0])
        if curr_nr > mnr:
            mnr = curr_nr


def num_rots_he(depth, n_qubits):
    return (2 * n_qubits * (depth + 1))


def num_tqg_he(depth, n_qubits):
    """ Returns number of two qubit gates in Hardware-Efficient Ansatz
    """
    ntqg = 0
    for d in range(depth):
        for i in range(n_qubits // 2):
            ntqg += 1

        for i in range(n_qubits // 2 - 1):
            ntqg += 1

    return ntqg

@jax.jit
def depolarizingchannel_jax_rv(xop, yop, zop, iop, rv):
    return rv[0] * xop  + rv[1] * yop + rv[2] * zop + rv[3] * iop


def seeds2rv(status, p):
    r = (jnp.sign(status - p) + jnp.sign(status - 2 * p) + jnp.sign(status - 3 * p))
    r = (r / 2 + 1.5).astype(int)
    rv = jax.nn.one_hot(r, 4, dtype=int)
    return rv


@jax.jit
def spsa_lr_dec(epoch, a, A, alpha):
    ak = a / (epoch + 1.0 + A) ** alpha
    return ak


@jax.jit
def spsa_lr_dec_new(epoch, a0, alpha=0.602):
    ak = a0 / (epoch + 1.0) ** alpha
    return ak


@jax.jit
def spsa_grad_dec_new(epoch, c0, gamma=0.101):
    ck = c0 / (epoch + 1.0) ** gamma
    return ck


@jax.jit
def beta_1_t(epoch, beta_1_0, lamda):
    beta_1_t = beta_1_0 / (epoch + 1) ** lamda
    return beta_1_t


def generator(n_params, iters, random_key):
    rng = jax.random.PRNGKey(random_key)
    x = jax.random.choice(rng, a=jnp.asarray([-1, 1]), shape=(n_params * iters,))

    return x


def spsa_grad(fun, current_params, n_params, ck, Deltak):
    n_params = len(current_params)

    grad = ((fun(current_params + ck * Deltak) -
             fun(current_params - ck * Deltak)) /
            (2 * ck * Deltak))

    return grad



@jax.jit
def adam_grad(epoch, grad, m, v, beta_1, beta_2, epsilon):
    m = beta_1 * m + (1 - beta_1) * grad
    v = beta_2 * v + (1 - beta_2) * jnp.power(grad, 2)
    m_hat = m / (1 - jnp.power(beta_1, epoch + 1))
    v_hat = v / (1 - jnp.power(beta_2, epoch + 1))

    return m_hat / (jnp.sqrt(v_hat) + epsilon), m, v


def min_spsa_v2(
        fun: Callable,
        x0: List[float],
        maxfev: int = 10000,
        a: float = 1.0,
        alpha: float = 0.602,
        c: float = 1.0,
        gamma: float = 0.101,
        lamda: float = 0.4,
        beta_1: float = 0.999,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        random_key: int = 42,
        adam: bool = True) -> OptimizeResult:

    last_elem = np.random.uniform(low=0, high=0.1 )


    if len(x0)>0:
        x0 = x0.at[-1].set(last_elem)

    current_params = jnp.asarray(x0)

    n_params = len(current_params)


   
    maxiter = int(jnp.ceil(maxfev / 2))

    n_fevals = 2*maxiter 

    best_params = current_params

    best_feval = fun(current_params)

    FE_best = 0

    m = 0

    v = 0

    Delta_ks = jnp.array( generator_np_v2(n_params, maxiter, random_key), dtype = jnp.float32 )
    Delta_counter = 0

    for epoch in range(maxiter):

        ak = spsa_lr_dec_new(epoch, a, alpha)
        ck = spsa_grad_dec_new(epoch, c, gamma)



        Delta_k = Delta_ks[epoch,:]


        grad = spsa_grad(fun, current_params, n_params, ck, Delta_k)


        if adam:
            if epoch > 0:
                beta_1t = beta_1_t(epoch, beta_1, lamda)
                a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1t, beta_2, epsilon)

            else:
                a_grad = grad

        else:
            a_grad = grad



        current_params -= ak * a_grad


        current_feval = fun(current_params)



        if current_feval < best_feval:
            best_feval = current_feval
            best_params = jnp.asarray(current_params)
            FE_best = n_fevals

    return OptimizeResult(fun=best_feval,
                           x=best_params,
                           FE_best=FE_best,
                           nit=epoch,
                           nfev=n_fevals)


def min_adam_spsa3(
        fun1: Callable,
        fun2: Callable,
        fun3: Callable,
        x0: List[float],
        maxfev1: int = 3000,
        maxfev2: int = 2000,
        maxfev3: int = 1000,
        a: float = 1.0,
        alpha: float = 0.602,
        c: float = 1.0,
        gamma: float = 0.101,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        random_key: int = 42
) -> OptimizeResult:


    last_elem = np.random.uniform(low=0, high=0.1 )


    if len(x0)>0:
        x0 = x0.at[-1].set(last_elem)

    maxiter1 = int(np.ceil(maxfev1 / 2))

    maxiter2 = int(np.ceil(maxfev2 / 2))

    maxiter3 = int(np.ceil(maxfev3 / 2))

    maxiter = maxiter1 + maxiter2 + maxiter3

    A = 0.01 * maxiter


    current_params = jnp.asarray(x0)

    n_params = len(current_params)



    n_fevals = 2*maxiter


    best_params = current_params

    best_feval = fun1(current_params)

    FE_best = 0

    m = 0

    v = 0

    Delta_ks = jnp.array(generator_np_v2(n_params, maxiter, random_key), dtype=jnp.float32)
    Delta_counter = 0


    for epoch in range(maxiter):

        ak = spsa_lr_dec(epoch, a, A, alpha)
        ck = spsa_grad_dec(epoch, c, gamma)

        if epoch < maxiter1:
            fun = fun1
        elif epoch >= maxiter1 and epoch < (maxiter1 + maxiter2):
            fun = fun2
        elif epoch >= (maxiter1 + maxiter2) and epoch < maxiter:
            fun = fun3

        Delta_k = Delta_ks[epoch, :]

        grad = spsa_grad(fun, current_params, n_params, ck, Delta_k)



        if epoch > 0:
            a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1, beta_2, epsilon)

            current_params -= ak * a_grad

        else:
            current_params -= ak * grad

        current_feval = fun(current_params)



        if current_feval < best_feval:
            best_feval = current_feval
            best_params = current_params
            FE_best = n_fevals

    return OptimizeResult(fun=best_feval,
                          x=best_params,
                          FE_best=FE_best,
                          nit= epoch,
                          nfev=n_fevals)


def min_spsa3_v2(
        fun1: Callable,
        fun2: Callable,
        fun3: Callable,
        x0: List[float],
        maxfev1: int = 2383,
        maxfev2: int = 715,
        maxfev3: int = 238,
        a: float = 1.8658,
        alpha: float = 0.9451,
        c: float =0.046,
        gamma: float = 0.1397,
        lamda: float = 0.0138,
        beta_1: float = 0.9658,
        beta_2: float = 0.8594,
        epsilon: float = 1e-8,
        random_key: int = 42,
        adam: bool = True
) -> OptimizeResult:
    current_params = np.asarray(x0)

    n_params = len(current_params)

    last_elem = np.random.uniform(low=0, high=0.1 )


    if len(x0)>0:
        x0 = x0.at[-1].set(last_elem)

    maxiter1 = int(np.ceil(maxfev1 / 2))

    maxiter2 = int(np.ceil(maxfev2 / 2))

    maxiter3 = int(np.ceil(maxfev3 / 2))

    maxiter = maxiter1 + maxiter2 + maxiter3

    current_params = jnp.asarray(x0)

    n_params = len(current_params)



    n_fevals = 0

    best_params = current_params

    best_feval = fun1(current_params)

    FE_best = 0

    m = 0

    v = 0

    Delta_ks = jnp.array(generator_np_v2(n_params, maxiter, random_key), dtype=jnp.float32)
    Delta_counter = 0

    for epoch in range(maxiter):

        ak = spsa_lr_dec_new(epoch, a, alpha)
        ck = spsa_grad_dec_new(epoch, c, gamma)


        Delta_k = Delta_ks[epoch,:]


        if epoch < maxiter1:
            fun = fun1
        elif epoch >= maxiter1 and epoch < (maxiter1 + maxiter2):
            fun = fun2



        elif epoch >= (maxiter1 + maxiter2) and epoch < maxiter:
            fun = fun3

        grad = spsa_grad(fun, current_params, n_params, ck, Delta_k)

        if adam:
            if epoch > 0:
                beta_1t = beta_1_t(epoch, beta_1, lamda)
                a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1t, beta_2, epsilon)
            else:
                a_grad = grad

        else:
            a_grad = grad


        n_fevals += 2

        current_params -= ak * a_grad

        current_feval = fun(current_params)



        if current_feval < best_feval:
            best_feval = current_feval
            best_params = jnp.asarray(current_params)
            FE_best = n_fevals

    return OptimizeResult(fun=best_feval,
                          x=best_params,
                          FE_best=FE_best,
                          nit=epoch,
                          nfev=n_fevals)

def generator_np(n_params, iters, seed):
    rng = np.random.default_rng(seed)
    x = rng.choice(a=np.array([-1, 1]), size=(n_params * iters,), replace=True)

    return jnp.array(x)


def generator_np_v2(n_params, iters, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.choice(a=np.array([-1, 1]), size=(iters, n_params), replace=True)

    return jnp.array(x)


def spsa_grad(fun, current_params, n_params, ck, Deltak):
    n_params = len(current_params)
    #
    #
    #
    #
    #
    #

    grad = ((fun(current_params + ck * Deltak) -
             fun(current_params - ck * Deltak)) /
            (2 * ck * Deltak))

    return grad


@jax.jit
def imprint_angles(angles, angles_dy, rot_pos):
    angles_dy = angles_dy.at[rot_pos].set(angles)
    return angles_dy


@jax.jit
def get_spsa_grad_params(current_params, ck, Deltak, angles_dy, rot_pos):
    angles_plus = current_params + ck * Deltak
    angles_minus = current_params - ck * Deltak
    spsa_grad_denom = 2 * ck * Deltak
    angles_dy_plus = imprint_angles(angles_plus, angles_dy, rot_pos)
    angles_dy_minus = imprint_angles(angles_minus, angles_dy, rot_pos)
    return angles_dy_plus, angles_dy_minus, spsa_grad_denom

@jax.jit
def get_spsa_grad_params_v2(current_params, ck, Deltak):
    angles_plus = current_params + ck * Deltak
    angles_minus = current_params - ck * Deltak
    spsa_grad_denom = 2 * ck * Deltak
    return angles_plus, angles_minus, spsa_grad_denom

def min_spsa_w_grad_v2(
        fun: Callable,
        fgrad: Callable,
        x0: List[float],
        maxfev: int = 10000,
        maxiter: Optional[int] = None,
        a0: float = 1.0,
        alpha: float = 0.602,
        c0: float = 1.0,
        gamma: float = 0.101,
        lamda: float = 0.4,
        beta_1: float = 0.999,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        random_key: int = 42,
        adam: bool = True) -> OptimizeResult:
    current_params = jnp.asarray(x0)

    n_params = len(current_params)


    if maxiter is None:
        maxiter = int(jnp.ceil(maxfev / 2))

    n_fevals = 0

    best_params = current_params

    best_feval = fun(current_params)

    FE_best = 0

    m = 0

    v = 0

    Delta_ks = generator_np(n_params, maxiter, random_key)
    Delta_counter = 0

    for epoch in range(maxiter):

        ak = spsa_lr_dec_new(epoch, a0, alpha)
        ck = spsa_grad_dec_new(epoch, c0, gamma)

        Delta_k = Delta_ks[Delta_counter:n_params + Delta_counter]
        Delta_counter += n_params


        grad = fgrad(current_params, ck, Delta_k)

        if adam:
            if epoch > 0:
                beta_1t = beta_1_t(epoch, beta_1, lamda)
                a_grad, m, v = adam_grad(epoch, grad, m, v, beta_1t, beta_2, epsilon)
            else:
                a_grad = grad

        else:
            a_grad = grad


        n_fevals += 2

        current_params -= ak * a_grad

        current_feval = fun(current_params)



        if current_feval < best_feval:
            best_feval = current_feval
            best_params = jnp.asarray(current_params)
            FE_best = n_fevals

    return OptimizeResult(fun=best_feval,
                          x=best_params,
                          FE_best=FE_best,
                          nit=epoch,
                          nfev=n_fevals)


@partial(jit, static_argnums=(1))
def generate_deltak(key, n_params):
    arr = jnp.array([-1, 1])
    axis, n_inputs, n_draws = 0, 2, n_params 
    ind = jax.random.randint(key, (n_params,), 0, n_inputs)
    Delta_k = jnp.take(arr, ind, axis)
    new_key, sub_key = jax.random.split(key)
    return Delta_k, new_key
