#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements some visualization helper functions
#
#

# jax
import jax
import jax.numpy as jnp
from jax import jit, jacobian
import jax.lax

# matplotlib
import matplotlib.pyplot as plt

# custom imports
from helpers.other import mpl_settings

def QSR_discrete_gradient_powerbalance(
        f: callable,
        g: callable,
        k: callable,
        ham_eta: callable,
        ell: callable,
        W: callable,
        QSR: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tt: jnp.ndarray,
        zz_ham: tuple[jnp.ndarray, jnp.ndarray],
        uu: jnp.ndarray,
        debug: bool = False,
        relative_powerbalance: bool = True,
        relative_supplybalance: bool = False,
        title: str = None,
        savepath: str = None,
    ) -> tuple[jnp.array, jnp.ndarray, jnp.ndarray]:
    """
    computes the error in the energy balance of the
    QSR discrete gradient method.

    nt = tt.shape[0]
    nsys = z0.shape[0]
    Delta_t = tt[1] - tt[0] # assumed to be constant

    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input
    Q, S, R = QSR

    :param f: function f in the system dynamics
    :param g: function g in the system dynamics
    :param k: function k in the system dynamics
    :param ham_eta: function returning (hamiltonian(z), eta(z)) for given z
    :param ell: function ell in the hill-moylan conditions
    :param W: function W in the hill-moylan conditions
    :param QSR: tuple of matrices Q, S, R in the supply rate
    :param tt: array of timepoints to be used
    :param zz_ham: solution of the QSR discrete gradient method and associated hamiltonian values
    :param uu: value of control input at timepoints
    :param debug: (optional) debug flag
    :param relative_powerbalance: (optional) whether a relative error should be used in the power balance
    :param relative_supplybalance: (optional) whether a relative error should be used in the supply balance
    :param title: (optional) title for the plot
    :return: values of solution at time points [t_0, t_1, ... ]
    """

    nt = tt.shape[0]
    Delta_t = tt[1] - tt[0] # assumed to be constant
    uumid = 1/2 * (uu[1:,:] + uu[:-1,:]) # linear interpolation of control input

    Q, S, R = QSR

    def get_eta_bar(z, zhat, ham_z, ham_zhat):
        # computes eta_bar
        # tries to minimize function calls to ham_eta

        # _, ham_z, _ = ham_eta(zhat) # hamiltonian(z)
        # _, ham_zhat, _ = ham_eta(zhat) # hamiltonian(zhat)
        _, eta_mid = ham_eta(1/2 * (z + zhat)) # eta(1/2*(z+zhat))

        alpha1 = ham_zhat - ham_z - eta_mid.T @ (zhat - z)
        alpha2 = (zhat - z).T @ (zhat - z)

        def true_fun():
            return eta_mid

        def false_fun():
            return eta_mid + alpha1/alpha2 * (zhat - z)

        return jax.lax.cond(jnp.allclose(alpha2, 0.0), true_fun, false_fun)

    # power balance reads as
    #
    # y_i^T Q y_i + 2 y_i^T Q u_i + u_i^T R u_i
    #  -  (H(z_{i+1}) - H(z_{i})) / (t_{i+1} - t_{i})
    # = (ell_i + W_i u_i)^T (ell_i + W_i u_i)
    #

    zz, ham = zz_ham
    zi = zz[0,:]
    errors_1 = jnp.zeros((nt-1,)) # error in powerbalance
    errors_2 = jnp.zeros((nt-1,)) # error in supply balance
    lhss_1 = jnp.zeros((nt-1,))
    lhss_2 = jnp.zeros((nt-1,))

    # jax.lax.fori implementation
    def body_fun(i, tup):

        zi, ham, errors_1, errors_2, lhss_1, lhss_2 = tup

        u_mid = uumid[i, :]
        zip1 = zz[i+1,:]
        ham_zi = ham[i]
        ham_zip1 = ham[i+1]
        eta_bar = get_eta_bar(zi, zip1, ham_zi, ham_zip1)

        # calculate energy supply
        g_mid = g(1/2 * (zi + zip1))
        k_mid = k(1/2 * (zi + zip1))
        ell_mid = ell(1/2 * (zi + zip1))
        W_mid = W(1/2 * (zi + zip1))
        h_mid = jnp.linalg.solve(Q @ k_mid + S, 1/2 * g_mid.T @ eta_bar + W_mid.T @ ell_mid)
        y_mid = h_mid + k_mid @ u_mid
        supply = y_mid.T @ Q @ y_mid + 2 * y_mid.T @ S @ u_mid + u_mid.T @ R @ u_mid

        # error in power balance
        lhs_1 = (ham_zip1 - ham_zi)/Delta_t
        rhs_1 = h_mid.T @ Q @ h_mid - ell_mid.T @ ell_mid + eta_bar.T @ g_mid @ u_mid
        error_1 = jnp.abs(lhs_1-rhs_1)
        errors_1 = errors_1.at[i].set(error_1)
        lhss_1 = lhss_1.at[i].set(lhs_1)

        # error in supply balance
        lhs_2 = supply - (ham_zip1 - ham_zi)/Delta_t
        rhs_2 = (ell_mid + W_mid @ u_mid).T @ (ell_mid + W_mid @ u_mid)
        error_2 = jnp.abs(lhs_2-rhs_2)
        errors_2 = errors_2.at[i].set(error_2)
        lhss_2 = lhss_2.at[i].set(lhs_2)

        zi = zip1

        return zi, ham, errors_1, errors_2, lhss_1, lhss_2

    _, ham, errors_1, errors_2, lhss_1, lhss_2 = jax.lax.fori_loop(0, nt-1, body_fun, (zi, ham, errors_1, errors_2, lhss_1, lhss_2))

    if relative_powerbalance:
        errors_1 = errors_1 / jnp.max(jnp.abs(lhss_1))

    if relative_supplybalance:
        errors_2 = errors_2 / jnp.max(jnp.abs(lhss_2))

    if title is not None:
        visualize_errors(
            [
                (tt, errors_1, r'relative error in~\eqref{eq:discrete-powerbalance}'),
                (tt, errors_2, r'absolute error in~\eqref{eq:discrete-QSR-dissipativity}')
                ],
            title=title,
            savepath=savepath,
            )

    return ham, errors_1, errors_2

def visualize_hamiltonian(
        tuplist: list[tuple[jnp.ndarray, jnp.ndarray, str]],
        axis_type: str = 'linear',
        title: str = None,
        ylabeltext = r'$\mathcal{H}(z(t))$',
        savepath: str = None,
        ) -> None:

    if axis_type == 'linear':
        plot_function = plt.plot
    elif axis_type == 'semilogy':
        plot_function = plt.semilogy
    else:
        raise NotImplementedError(f'axis type {axis_type} not implemented')

    for tt, ham, label in tuplist:
        plot_function(tt, ham, linewidth=3.0, alpha=0.6, label=label)

    plt.legend()
    plt.xlabel('time $t$')
    plt.ylabel(ylabeltext)

    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    plt.title(r'\tiny '+title)
    plt.tight_layout()
    plt.show()

def visualize_errors(
        tuplist: list[tuple[jnp.ndarray, jnp.ndarray, str]],
        title: str = None,
        ylabeltext = r'$\errorenergy$',
        savepath: str = None,
        ) -> None:

    for tt, errors, label in tuplist:
        plt.semilogy(tt[:-1], errors, linewidth=3.0, alpha=0.6, label=label)

    plt.legend()
    plt.xlabel('time $t$')
    plt.ylabel(ylabeltext)
    plt.ylim([.5e-17, .5e-4])

    if savepath is not None:
        plt.tight_layout()
        plt.savefig(savepath + '.pgf') # save as pgf
        plt.savefig(savepath + '.png') # save as png
        print(f'figure saved under savepath {savepath} (as pgf and png)')

    plt.title(r'\tiny '+title)
    plt.tight_layout()
    plt.show()

