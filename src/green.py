"""Funcion de Green, truncamiento y operador de punto fijo T_lambda.

El problema

    -u'' = lambda f(u) en (0,1),   u(0) = u(1) = 0

es equivalente a la ecuacion de punto fijo u = T_lambda u en C([0,1]), con

    (T_lambda u)(t) = lambda * int_0^1 G(t, s) f_tilde(u(s)) ds,

donde G es la funcion de Green de -d^2/dt^2 con condiciones de Dirichlet y
f_tilde es la truncada de f fuera de [0,1]. El truncamiento es lo que vuelve
al integrando acotado y globalmente Lipschitz: entonces T_lambda tiene imagen
acotada en C^1, es compacto por Arzela-Ascoli, y Schauder se aplica en
cualquier bola grande de C([0,1]). Las cotas a priori 0 <= u <= 1 (principio
del maximo) garantizan despues que toda solucion del problema truncado es
solucion del original.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
ScalarField = Callable[[Array], Array]


def green_kernel(t: Array | float, s: Array | float) -> Array | float:
    """Evalua el nucleo de Green G(t, s) del operador -d^2/dt^2 con Dirichlet en (0,1).

    Explicitamente

        G(t, s) = s (1 - t)   si 0 <= s <= t <= 1,
        G(t, s) = t (1 - s)   si 0 <= t <= s <= 1,

    o de forma compacta G(t, s) = min(t, s) * (1 - max(t, s)). Propiedades que
    los tests verifican y que se usan en las pruebas: G >= 0 (positividad, de
    donde sale que T_lambda preserva el cono de funciones no negativas),
    G(t, s) = G(s, t) (simetria, el operador es autoadjunto), max G = 1/4 en
    t = s = 1/2, e int_0^1 G(t, s) ds = t (1 - t) / 2.

    Parametros
    ----------
    t : float o ndarray
        Punto o puntos de evaluacion en [0, 1]. Se admite difusion por
        broadcasting contra `s` (por ejemplo t de forma (n, 1) y s de forma
        (m,) para obtener la matriz (n, m) del nucleo).
    s : float o ndarray
        Punto o puntos de integracion en [0, 1].

    Devuelve
    --------
    float o ndarray
        G(t, s), con la forma que resulte del broadcasting de `t` y `s`.
    """
    raise NotImplementedError


def truncate(
    f: ScalarField,
    lo: float = 0.0,
    hi: float = 1.0,
) -> ScalarField:
    """Devuelve la truncada f_tilde de una no linealidad fuera del intervalo [lo, hi].

    La truncada se define extendiendo f por su valor en los extremos:

        f_tilde(u) = f(lo)  si u < lo,
        f_tilde(u) = f(u)   si lo <= u <= hi,
        f_tilde(u) = f(hi)  si u > hi,

    es decir f_tilde = f o clip(., lo, hi). Para la reaccion de Allee, como
    f(0) = f(1) = 0, esto significa que f_tilde se anula identicamente fuera de
    [0, 1]; ese es justamente el hecho que hace funcionar el principio del
    maximo (donde u < 0 la ecuacion truncada da -u'' = 0, luego u es afin y se
    anula, contradiccion).

    El truncamiento reemplaza una f polinomial no acotada por una f_tilde
    acotada y globalmente Lipschitz, sin cambiar el conjunto de soluciones del
    problema gracias a las cotas a priori.

    Parametros
    ----------
    f : callable
        No linealidad de una sola variable, ya evaluada en el alpha de interes
        (por ejemplo `lambda u: nonlinearity.f(u, alpha)`). Debe aceptar
        arrays de numpy y devolver arrays de la misma forma.
    lo : float
        Extremo inferior del intervalo donde f se deja intacta. Por defecto 0.
    hi : float
        Extremo superior del intervalo donde f se deja intacta. Por defecto 1.

    Devuelve
    --------
    callable
        La funcion truncada f_tilde, con la misma convencion de entrada y
        salida que `f`.

    Levanta
    -------
    ValueError
        Si lo >= hi.
    """
    raise NotImplementedError


def fixed_point_operator(
    u: Array,
    lam: float,
    alpha: float,
    t: Optional[Array] = None,
    truncated: bool = True,
) -> Array:
    """Aplica el operador de punto fijo T_lambda a una funcion dada por sus valores en una grilla.

    Calcula, por cuadratura sobre la grilla,

        (T_lambda u)(t_i) = lambda * int_0^1 G(t_i, s) f_tilde(u(s)) ds,

    Se sugiere armar la matriz del nucleo G(t_i, s_j) con `green_kernel` por
    broadcasting e integrar con la regla del trapecio (`numpy.trapezoid`) sobre
    el eje de `s`; con una grilla uniforme de unos cientos de puntos alcanza
    para todos los notebooks. Por construccion el resultado se anula en t = 0 y
    t = 1, ya que G(0, s) = G(1, s) = 0.

    Los puntos fijos de esta aplicacion son exactamente las soluciones del BVP.
    Iterandola desde la supersolucion se obtiene la sucesion monotona de
    `monotone.monotone_iteration`.

    Parametros
    ----------
    u : ndarray de forma (n,)
        Valores de la funcion en los nodos de la grilla `t`.
    lam : float
        Parametro lambda > 0.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    t : ndarray de forma (n,), opcional
        Grilla de nodos en [0, 1], creciente, con t[0] = 0 y t[-1] = 1. Si es
        None se usa una grilla uniforme `numpy.linspace(0, 1, len(u))`.
    truncated : bool
        Si es True (por defecto) se integra f_tilde (la truncada de `truncate`);
        si es False se integra f sin truncar. La version sin truncar solo sirve
        para comparar en el notebook 02 y puede divergir al iterar.

    Devuelve
    --------
    ndarray de forma (n,)
        Valores de T_lambda u en los mismos nodos, con ceros en los extremos.

    Levanta
    -------
    ValueError
        Si `u` y `t` no tienen la misma longitud, o si `t` no es creciente, o
        si no cubre [0, 1].
    """
    raise NotImplementedError
