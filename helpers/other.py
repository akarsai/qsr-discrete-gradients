#
#                        author:
#                     attila karsai
#                karsai@math.tu-berlin.de
#
# this file implements a helper class to style print output using
# ansi codes and a helper function to prepare matplotlib figures
# for publication.
#


import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental import sparse
import re

def mpl_fontsize(
        fontsize: int = None,
        bigger_axis_labels: bool = True,
        ):
    
    if fontsize is not None:
        plt.rcParams.update({"font.size": fontsize})
        
        # make legend font size smaller
        plt.rcParams.update({
            "legend.fontsize": fontsize - 6,
            })
    
        # bigger axis labels if needed
        if bigger_axis_labels:
            plt.rcParams.update({
                "axes.labelsize": fontsize + 2,
                "axes.titlesize": fontsize + 2,
                })
            
    return
    
def mpl_settings(
        figsize: tuple = (5.5,4),
        backend: str = None,
        latex_font: str = 'computer modern',
        fontsize: int = None,
        bigger_axis_labels: bool = True,
        dpi: int = 500,
        ) -> None:
    """
    sets matplotlib settings for latex

    :return: None
    """

    plt.rcParams['figure.dpi'] = dpi
    # default for paper: (5.5,4)
    plt.rcParams['figure.figsize'] = figsize
    plt.rc('text', usetex=True)
    
    preamble = '\n'.join([
                        r'\usepackage{amsmath,amssymb}',
                        r'\newcommand{\projnodes}{s_{\Pi}}',
                        r'\newcommand{\quadnodes}{s_Q}',
                        r'\newcommand{\error}{\mathcal{E}}',
                        r'\newcommand{\errorenergy}{\error_{\mathrm{energy}}}',
                        r'\newcommand{\errorenergyabs}{\error_{\mathrm{energy}}^{\mathrm{abs}}}',
                        r'\newcommand{\errorstate}{\error_{\mathrm{state}}}',
                        r'\newcommand{\errorstatenodal}{\error_{\mathrm{state,nodal}}}',
                        r'\newcommand{\hamc}{\hat{\mathcal{H}}}',
                        r'\newcommand{\zc}{\hat{z}}',
                        r'\newcommand{\zekf}{\overline{z}}',
                        r'\newcommand{\norm}[1]{\Vert #1 \Vert}',
                    ])
    
    plt.rc('text.latex', preamble=preamble)

    plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,      # don't setup fonts from rc parameters
            "pgf.preamble": preamble, # the preamble really need to be defined two times ... i do not know why
            "savefig.transparent": True,
            })
    
    mpl_fontsize(fontsize=fontsize, bigger_axis_labels=bigger_axis_labels)

    if latex_font == 'times':
        plt.rc('font',**{'family':'serif','serif':['Times']})
    elif latex_font == 'computer modern':
        plt.rc('font',**{'family':'serif'})

    plt.rc('axes.formatter', useoffset=False)
    # plt.rcParams['savefig.transparent'] = True

    if backend is not None:
        matplotlib.use(backend)
    if backend == 'macosx':
        plt.rcParams['figure.dpi'] = 140

    return

class style:
    info = '\033[38;5;027m'
    success = '\033[38;5;028m'
    warning = '\033[38;5;208m'
    fail = '\033[38;5;196m'
    #
    bold = '\033[1m'
    underline = '\033[4m'
    italic = '\033[3m'
    end = '\033[0m'