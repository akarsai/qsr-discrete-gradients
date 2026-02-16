#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements time discretization methods suitable for
# automatic differentiation via jax
#
# currently, the following methods are implemented:
# - implicit midpoint with linear interpolation of the
#   control
# - a custom discrete gradient method
#

# jax
import jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

# custom imports
from helpers.newton import newton_lineax
from helpers.other import style

def implicit_midpoint(
        f: callable,
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        type = 'forward',
        debug = False,
        ) -> jnp.ndarray:

    """
    uses implicit midpoint method to solve the initial value problem

    z' = f(z,u), z(tt[0]) = z0    (if type == 'forward')

    or

    p' = f(p,u), p(tt[-1]) = p0   (if type == 'backward')

    in the implementation, the control input is linearly interpolated
    to evaluate at midpoints of the time interval.

    :param f: right hand side of ode, f = f(z,u)
    :param tt: timepoints, assumed to be evenly spaced
    :param z0: initial or final value
    :param uu: control input at timepoints, shape = (len(tt), N)
    :param type: 'forward' or 'backward'
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        z[i,:] = z(tt[i])
        p[i,:] = p(tt[i])
    """

    N = len(z0) # system dimension
    nt = len(tt) # number of timepoints
    dt = tt[1] - tt[0] # timestep, assumed to be constant
    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input

    def F_implicit_midpoint(zj, zjm1, uj12):
        return \
            zj \
            - zjm1 \
            - dt*f( 1/2*(zjm1+zj), uj12)

    solver = newton_lineax(F_implicit_midpoint, debug=debug)

    if type == 'forward':

        z = jnp.zeros((nt,N))
        z = z.at[0,:].set(z0)

        # after that bdf method
        def body( j, var ):
            z, uumid = var

            zjm1 = z[j-1,:]
            uj12 = uumid[j-1,:]

            y = solver(zjm1, zjm1, uj12)
            z = z.at[j,:].set(y)

            if debug: jax.debug.print(f'{style.info}timestep number = {{j}}{style.end}', j=j)
            # jax.debug.print( 'iter = {x}', x = i)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,zjm1,zjm2,zjm3,zjm4,uj))) )

            return z, uumid

        z, _ = jax.lax.fori_loop(1, nt, body, (z,uumid))

        return z

    else: # type == 'backward'

        return implicit_midpoint(f, tt[::-1], z0, uu[::-1,:], type='forward')[::-1,:]

def QSR_discrete_gradient(
        f: callable,
        g: callable,
        k: callable,
        ham_eta: callable,
        ell: callable,
        W: callable,
        QSR: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tt: jnp.ndarray,
        z0: jnp.ndarray,
        uu: jnp.ndarray,
        debug: bool = False,
        return_hamiltonian: bool = False,
    ) -> jnp.array:
    """
    computes an approximate solution of

    z' = f(z) + g(z) u,    z(0) = z_0
    y  = h(z) + k(z) u

    on given timesteps in the time horizon [0,T].
    the system is assumed to describe dissipative dynamics
    with a quadratic supply rate given by

    w(u,y) = y^T Q y + 2 y^T S u + u^T R u.

    the functions eta, ell and W are assumed to satisfy the
    hill-moylan conditions

    eta(z)^T f(z) = h(z)^T Q h(z) - ell(z)^T ell(z)
    1/2 eta(z)^T g(z) = h(z)^T (Q D(z) + S) - ell(z)^T W(z)
    W(z)^T W(z) = R + k(z)^T S + S^T k(z) + k(z)^T Q k(z)

    where eta(z) = \nabla hamiltonian(z) and hamiltonian(z)
    is a storage function of the system.

    in the method, Q D(z) + S is assumed to be invertible.
    hence, h is not needed for the algorithm, as it can be
    recovered using

    h(z) = (Q k(z) + S)^{-T} (1/2 g(z)^T eta(z) + W(z)^T ell(z))

    ---
    the call signature of eta is assumed to be

    ham_eta: z -> hamiltonian(z), eta(z)

    to minimize function calls to hamiltonian in case the
    computation is expensive.
    ---

    the method uses the gonzalez discrete gradient

    :param f: function f in the system dynamics
    :param g: function g in the system dynamics
    :param k: function k in the system dynamics
    :param ham_eta: function returning (hamiltonian(z), eta(z)) for given z
    :param ell: function ell in the hill-moylan conditions
    :param W: function W in the hill-moylan conditions
    :param QSR: tuple of matrices Q, S, R in the supply rate
    :param tt: array of timepoints to be used
    :param z0: initial condition
    :param uu: value of control input at timepoints
    :param debug: (optional) debug flag
    :param return_hamiltonian: (optional) if True, also the hamiltonian values are returned
    :return: values of solution at time points [t_0, t_1, ... ]
    """

    nt = tt.shape[0]
    nsys = z0.shape[0]
    Delta_t = tt[1] - tt[0] # assumed to be constant

    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input
    Q, S, R = QSR

    def get_uncontrolled_discrete_dynamics(z, zhat, ham_z):
        # tries to minimize function calls to ham_eta

        _, eta_mid = ham_eta(1/2 * (z + zhat)) # eta(1/2*(z+zhat))

        def get_eta_bar():
            # computes eta_bar

            # _, ham_z, _ = ham_eta(zhat) # hamiltonian(z)
            ham_zhat, _ = ham_eta(zhat) # hamiltonian(zhat)

            alpha1 = ham_zhat - ham_z - eta_mid.T @ (zhat - z)
            alpha2 = (zhat - z).T @ (zhat - z)

            def true_fun():
                return eta_mid

            def false_fun():
                return eta_mid + alpha1/alpha2 * (zhat - z)

            return jax.lax.cond(jnp.allclose(alpha2, 0.0), true_fun, false_fun)

        eta_bar = get_eta_bar()
        norm_eta_bar = jnp.linalg.norm(eta_bar)
        eta_bar_rescaled = eta_bar.reshape((-1,1)) / norm_eta_bar # has norm 1 and shape (n,1) - without the reshape, this does not work!
        projection = jnp.eye(nsys) - eta_bar_rescaled @ eta_bar_rescaled.T # projects onto span(eta)^\perp
        f_mid = f(1/2 * (z + zhat))
        g_mid = g(1/2 * (z + zhat))
        k_mid = k(1/2 * (z + zhat))
        ell_mid = ell(1/2 * (z + zhat))
        W_mid = W(1/2 * (z + zhat))
        h_mid = jnp.linalg.solve(Q @ k_mid + S, 1/2 * g_mid.T @ eta_bar + W_mid.T @ ell_mid)

        def get_gamma():
            """
            computes gamma according to formula

            gamma = (h_mid^T Q h_mid - ell_mid^T ell_mid)/(eta_bar^T eta_bar)
            """
            def true_fun():
                return 0.0

            def false_fun():
                return (h_mid.T @ Q @ h_mid - ell_mid.T @ ell_mid) / (norm_eta_bar**2)

            return jax.lax.cond(jnp.allclose(norm_eta_bar, 0.0), true_fun, false_fun)

        gamma = get_gamma()

        uncontrolled_discrete_dynamics = (
                gamma * eta_bar
                + projection @ f_mid
            )

        return uncontrolled_discrete_dynamics


    def F(zip1, zi, ham_zi, u_mid):

        g_mid = g(1/2 * (zi + zip1))

        return (
            zip1
            - zi
            - Delta_t * (
                get_uncontrolled_discrete_dynamics(zi, zip1, ham_zi)
                + g_mid @ u_mid
            )
        )

    solver = newton_lineax(F, debug=debug)

    # set initial condition and hamiltonian array
    z = jnp.zeros((nt,nsys)).at[0,:].set( z0 )
    ham_z0, _ = ham_eta(z0)
    ham = jnp.zeros((nt,)).at[0].set(ham_z0)

    # loop
    def body( i, var ):
        z, ham, uumid = var

        zk = z[i,:]
        ham_zk = ham[i]
        umidk = uumid[i,:]

        y = solver(zk, zk, ham_zk, umidk)
        z = z.at[i+1,:].set(y)
        ham_zkp1, _ = ham_eta(y) # hamiltonian(zkp1)
        ham = ham.at[i+1].set(ham_zkp1)

        # jax.debug.print('timestep number = {k}', k=k)
        if debug: jax.debug.print('timestep number = {i}', i=i)

        return z, ham, uumid

    z, ham, _ = jax.lax.fori_loop(0, nt-1, body, (z, ham, uumid))

    if return_hamiltonian:
        return z, ham

    return z