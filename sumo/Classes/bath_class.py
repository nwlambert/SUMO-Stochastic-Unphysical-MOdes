"""
This module provides utilities for describing baths 
"""

import enum

import numpy as np
from scipy.linalg import eigvalsh
from scipy.integrate import quad

from qutip.core import data as _data
from qutip.core.qobj import Qobj
from qutip.core.superoperator import spre, spost

__all__ = [
    "BathExponent",
     "BathGeneralCorr",
    "Bath",
    "BosonicBath",
    "UnderDampedBath",
]

def integrate(function,*args,**kwargs):
    if 'x_i' not in kwargs:
          raise Exception('Missing x_i, {} instead'.format(kwargs.keys()))
    if 'x_f' not in kwargs:
          raise Exception('Missing x_f, {} instead'.format(kwargs.keys()))
    x_i = kwargs['x_i']
    x_f = kwargs['x_f']
    limit = kwargs['limit']
    def function_real(x,*args):
        return np.real(function(x,*args))
    def function_imag(x,*args):
        return np.imag(function(x,*args))
        
    quadv = np.vectorize(quad)

    return quadv(function_real, x_i, x_f,args=args,limit=limit)[0] + 1j * quadv(function_imag, x_i, x_f,args=args,limit=limit)[0]


def _isequal(Q1, Q2, tol):
    """ Return true if Q1 and Q2 are equal to within the given tolerance. """
    return _data.iszero(_data.sub(Q1.data, Q2.data), tol=tol)


class BathExponent:
    """
    Represents a single exponent (naively, an excitation mode) within the
    decomposition of the correlation functions of a bath.

    Parameters
    ----------
    type : {"R", "I"} or BathExponent.ExponentType
        The type of bath exponent.

        "R" and "I" are bosonic bath exponents that appear in the real and
        imaginary parts of the correlation expansion.

    Q : Qobj
        The coupling operator for this excitation mode.

    vk : complex
        The frequency of the exponent of the excitation term.

    ck : complex
        The coefficient of the excitation term.

    tag : optional, str, tuple or any other object
        A label for the exponent (often the name of the bath). It
        defaults to None.

    Attributes
    ----------
 

    All of the parameters are also available as attributes.
    """
    types = enum.Enum("ExponentType", ["R", "I"])


    def __init__(
            self, type,  Q, ck, vk, tag=None,
    ):
        if not isinstance(type, self.types):
            type = self.types[type]
        self.type = type
        self.Q = Q
        self.ck = ck
        self.vk = vk
        self.tag = tag

    def __repr__(self):
        dims = getattr(self.Q, "dims", None)
        return (
            f"<{self.__class__.__name__} type={self.type.name}"
            f" dim={self.dim!r}"
            f" Q.dims={dims!r}"
            f" ck={self.ck!r} vk={self.vk!r}"
            f" tag={self.tag!r}>"
        )

class BathGeneralCorr:
    """
    Represents a time dependent cotinous function for the bath correlation function 
    not explicitly represented by analytic exponentials.
    
    This can be used for fitting, or for classical noise terms.

    Parameters
    ----------
    type : {"R", "I"} or BathExponent.ExponentType
        The type of bath exponent.

        "R" and "I" are bosonic bath exponents that appear in the real and
        imaginary parts of the correlation expansion.

    Q : Qobj
        The coupling operator for this excitation mode.

    C : function

    tag : optional, str, tuple or any other object
        A label for the exponent (often the name of the bath). It
        defaults to None.

    Attributes
    ----------
 

    All of the parameters are also available as attributes.
    """
    types = enum.Enum("ExponentType", ["R", "I"])


    def __init__(
            self, type,  Q,  C, tag=None,
    ):
        if not isinstance(type, self.types):
            type = self.types[type]
        self.type = type
        self.Q = Q
        self.C = C
        self.tag = tag

    def __repr__(self):
        dims = getattr(self.Q, "dims", None)
        return (
            f"<{self.__class__.__name__} type={self.type.name}"
            f" dim={self.dim!r}"
            f" Q.dims={dims!r}"
            f" C={self.C!r}"
            f" tag={self.tag!r}>"
        )
    
class Bath:
    """
    Represents a list of bath expansion exponents
    and general functions.

    Parameters
    ----------
    exponents : list of BathExponent
        The exponents of the correlation function describing the bath.

    Attributes
    ----------

    All of the parameters are available as attributes.
    """
    def __init__(self, exponents, functions):
        self.exponents = exponents
        self.functions = functions


class BosonicBath(Bath):
    """
    A helper class for constructing a bosonic bath from 
    real and imaginary parts of
    the bath correlation function.

    If the correlation functions ``C(t)`` is split into real and imaginary
    parts::

        C(t) = C_real(t) + i * C_imag(t)

    then::

        C_real(t) = sum(ck_real * exp(- vk_real * t)) + C_real_rest(t)
        C_imag(t) = sum(ck_imag * exp(- vk_imag * t)) + C_imag_rest(t)

    Defines the coefficients ``ck`` and the frequencies ``vk``
    and any general functions not decomposible directly.

    Note that the ``ck`` and ``vk`` may be complex, even through ``C_real(t)``
    and ``C_imag(t)`` (i.e. the sum) is real.



    Parameters
    ----------
    Q : Qobj
        The coupling operator for the bath.

    ck_real : list of complex
        The coefficients of the expansion terms for the real part of the
        correlation function. The corresponding frequencies are passed as
        vk_real.

    vk_real : list of complex
        The frequencies (exponents) of the expansion terms for the real part of
        the correlation function. The corresponding ceofficients are passed as
        ck_real.

    ck_imag : list of complex
        The coefficients of the expansion terms in the imaginary part of the
        correlation function. The corresponding frequencies are passed as
        vk_imag.

    vk_imag : list of complex
        The frequencies (exponents) of the expansion terms for the imaginary
        part of the correlation function. The corresponding ceofficients are
        passed as ck_imag.

    C_real_rest : function

    C_imag_rest : function    

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    """
    def _check_cks_and_vks(self, ck_real, vk_real, ck_imag, vk_imag):
        if len(ck_real) != len(vk_real) or len(ck_imag) != len(vk_imag):
            raise ValueError(
                "The bath exponent lists ck_real and vk_real, and ck_imag and"
                " vk_imag must be the same length."
            )

    def _check_coup_op(self, Q):
        if not isinstance(Q, Qobj):
            raise ValueError("The coupling operator Q must be a Qobj.")

    def __init__(
            self, Q, ck_real, vk_real, ck_imag, vk_imag, C_real_rest,
            C_imag_rest, tag=None,
    ):
        self._check_cks_and_vks(ck_real, vk_real, ck_imag, vk_imag)
        self._check_coup_op(Q)

        exponents = []
        exponents.extend(
            BathExponent("R", Q, ck, vk, tag=tag)
            for ck, vk in zip(ck_real, vk_real)
        )
        exponents.extend(
            BathExponent("I", Q, ck, vk, tag=tag)
            for ck, vk in zip(ck_imag, vk_imag)
        )

        functions = []
        functions.extend(
            BathGeneralCorr("R", C_real_rest, tag=tag)
        )
        functions.extend(
            BathGeneralCorr("I", C_imag_rest, tag=tag)
        )

        super().__init__(exponents, functions)

  


class UnderDampedBath(BosonicBath):
    """
    A helper class for constructing an under-damped bosonic bath from the
    bath parameters (see parameters below).

    Parameters
    ----------
    Q : Qobj
        Operator describing the coupling between system and bath.

    lam : float
        Coupling strength.

    gamma : float
        Bath spectral density cutoff frequency.

    w0 : float
        Bath spectral density resonance frequency.

    T : float
        Bath temperature.

    Nk : int
        Number of exponential terms used to approximate the bath correlation
        functions.

    tag : optional, str, tuple or any other object
        A label for the bath exponents (for example, the name of the
        bath). It defaults to None but can be set to help identify which
        bath an exponent is from.
    """
    def __init__(
        self, Q, lam, gamma, w0, T, Nk, tag=None,
    ):
        ck_real, vk_real, ck_imag, vk_imag, C_real_rest, C_imag_rest = self._matsubara_params(
            lam=lam,
            gamma=gamma,
            w0=w0,
            T=T,
            Nk=Nk,
        )

        super().__init__(
            Q, ck_real, vk_real, ck_imag, vk_imag, C_real_rest, C_imag_rest, tag=tag,
        )

    def _matsubara_params(self, lam, gamma, w0, T, Nk):
        """ Calculate the Matsubara coefficents and frequencies. """
        if T!=0:
            beta = 1/T
        else: 
            beta = 'inf'

        Om = np.sqrt(w0**2 - (gamma/2)**2)
        Gamma = gamma/2.
        if beta == 'inf':
            ck_real = [lam**2 / (4 * Om), lam**2 / (4*Om)]
            vk_real = [-1.0j * Om + Gamma, 1.0j * Om + Gamma]
            ck_imag = [1.0j * lam**2 / (4 * Om), -1.0j * lam**2 / (4 * Om)]
            vk_imag = [-1.0j * Om + Gamma, 1.0j * Om + Gamma]
            def C_real_rest(t):
                W_max = 10 * Om
                a = Omega + 1j * Gamma
                aa = Omega - 1j * Gamma
                def W(t):
                    return - lam**2 * gamma / np.pi * w * np.exp(-w * t) / ((a**2 + w**2) * (aa**2 + w**2))
                return integrate(W, x_i=0, x_f=W_max,limit=1000)    

            def C_imag_rest(t):
                return 0
            
        else:
            ck_real = ([
                (lam**2 / (4 * Om))
                * (1 / np.tanh(beta * (Om + 1.0j * Gamma) / 2)),
                (lam**2 / (4*Om))
                * (1 / np.tanh(beta * (Om - 1.0j * Gamma) / 2)),
            ])

            ck_real.extend([
                (-2 * lam**2 * gamma / beta) * (2 * np.pi * k / beta)
                / (
                    ((Om + 1.0j * Gamma)**2 + (2 * np.pi * k/beta)**2)
                    * ((Om - 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2)
                )
                for k in range(1, Nk + 1)
            ])

            vk_real = [-1.0j * Om + Gamma, 1.0j * Om + Gamma]
            vk_real.extend([
                2 * np.pi * k * T
                for k in range(1, Nk + 1)
            ])

            ck_imag = [
                1.0j * lam**2 / (4 * Om),
                -1.0j * lam**2 / (4 * Om),
            ]

            vk_imag = [-1.0j * Om + Gamma, 1.0j * Om + Gamma]

            def C_real_rest(t):
                return 0
            
            def C_imag_rest(t):
                return 0

        return ck_real, vk_real, ck_imag, vk_imag, C_real_rest, C_imag_rest
