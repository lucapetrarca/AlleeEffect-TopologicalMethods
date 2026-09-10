"""Iteracion monotona entre una subsolucion y una supersolucion.

Si phi <= psi son sub y supersolucion del problema, la sucesion

    u_0 = psi,     u_{n+1} = T_lambda u_n

es DECRECIENTE y converge uniformemente a la solucion MAXIMAL del intervalo de
orden [phi, psi]; arrancando en u_0 = phi la sucesion es CRECIENTE y converge a
la solucion minimal. La monotonia sale de que T_lambda es un operador creciente
sobre el intervalo de orden -consecuencia de G >= 0 y, si f no fuera creciente,
del truco estandar de agregar y restar un termino k u con k grande- y la
convergencia, del teorema de Dini mas la compacidad de T_lambda.

En este problema psi == 1 es supersolucion y phi es la de `subsuper`, asi que
iterando desde 1 se obtiene la rama maximal: es la version constructiva del
resultado 3, y la que se grafica en el notebook 06.

El modulo devuelve la sucesion COMPLETA, no solo el limite, porque lo que se
quiere mostrar es el descenso monotono iteracion por iteracion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SucesionMonotona:
    """Sucesion generada por la iteracion de punto fijo.

    Atributos
    ---------
    t : ndarray de forma (n,)
        Grilla en [0, 1].
    iterates : ndarray de forma (m + 1, n)
        Las iteradas u_0, u_1, ..., u_m evaluadas en la grilla. La fila 0 es el
        dato inicial. Se guardan todas para poder graficar el descenso.
    diffs : ndarray de forma (m,)
        Normas del supremo ||u_{k+1} - u_k||, que deberian decrecer.
    converged : bool
        True si se alcanzo la tolerancia antes del maximo de iteraciones.
    monotone : bool
        True si la sucesion resulto efectivamente monotona a lo largo de todas
        las iteraciones (decreciente si se arranco en la supersolucion,
        creciente si se arranco en la subsolucion). Es un control de sanidad:
        si sale False, hay un error de signo o la grilla es demasiado gruesa.
    decreasing : bool
        Sentido de la monotonia esperada, deducido del dato inicial.
    lam : float
        Parametro lambda.
    alpha : float
        Umbral de Allee.
    """

    t: Array
    iterates: Array
    diffs: Array
    converged: bool
    monotone: bool
    decreasing: bool
    lam: float
    alpha: float

    @property
    def limit(self) -> Array:
        """Ultima iterada, es decir la aproximacion de la solucion limite.

        Devuelve
        --------
        ndarray de forma (n,)
            La fila final de `iterates`.
        """
        raise NotImplementedError


def monotone_iteration(
    u0: Array,
    lam: float,
    alpha: float,
    t: Optional[Array] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    check_monotone: bool = True,
) -> SucesionMonotona:
    """Itera u_{n+1} = T_lambda u_n desde u0 y devuelve la sucesion completa.

    Aplica repetidamente `green.fixed_point_operator` hasta que
    ||u_{n+1} - u_n||_infinito <= tol o hasta agotar `max_iter`, guardando cada
    iterada. Si `check_monotone` es True, compara cada iterada con la anterior
    punto a punto y registra en el resultado si la monotonia se mantuvo; el
    sentido esperado se deduce del dato inicial (decreciente si u0 es la
    supersolucion psi == 1, creciente si u0 es una subsolucion).

    Casos de uso en el notebook 06:

    * u0 == 1 (la supersolucion): la sucesion baja hasta la solucion MAXIMAL.
    * u0 = la subsolucion de `subsuper.build_subsolution`: la sucesion sube
      hasta la solucion MINIMAL del intervalo de orden [phi, 1], que para
      lambda > lambda* coincide con la maximal salvo que phi este por debajo de
      la solucion intermedia.
    * u0 un poco por debajo de la solucion intermedia (inestable): la sucesion
      colapsa a 0. Es la visualizacion del umbral de extincion, y explica por
      que la solucion de indice -1 no se puede obtener por iteracion monotona
      y hace falta el grado.

    Parametros
    ----------
    u0 : ndarray de forma (n,)
        Dato inicial en la grilla, con u0[0] = u0[-1] = 0 si se quiere respetar
        la condicion de borde (la supersolucion psi == 1 no la cumple, y esta
        bien: es supersolucion, no solucion, y la primera iterada ya se anula
        en los extremos).
    lam : float
        Parametro lambda > 0.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    t : ndarray de forma (n,), opcional
        Grilla en [0, 1]. Si es None se usa `numpy.linspace(0, 1, len(u0))`.
    max_iter : int
        Tope de iteraciones.
    tol : float
        Tolerancia en norma del supremo entre iteradas consecutivas.
    check_monotone : bool
        Si es True se verifica la monotonia iterada a iterada y se reporta en
        el campo `monotone`.

    Devuelve
    --------
    SucesionMonotona
        Con todas las iteradas, las diferencias sucesivas y las banderas de
        convergencia y monotonia.

    Levanta
    -------
    ValueError
        Si `u0` y `t` no tienen la misma longitud, o si lam <= 0.
    """
    raise NotImplementedError
