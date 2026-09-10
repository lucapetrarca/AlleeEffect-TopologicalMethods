"""Indice de Leray-Schauder de cada solucion y verificacion de la suma de indices.

Si u es solucion aislada y no degenerada de u = T_lambda u, su indice es

    i(u) = deg(I - T_lambda, B_eps(u), 0) = (-1)^m,

donde m es la cantidad de autovalores de la derivada compacta T'_lambda(u) que
son estrictamente mayores que 1, contados con multiplicidad. En este problema
T'_lambda(u) v = lambda K (f'(u) v) con K el operador de Green, y hay una
traduccion comoda: mu es autovalor del problema linealizado

    -v'' - lambda f'(u) v = mu v,   v(0) = v(1) = 0,

si y solo si el autovalor correspondiente de T'_lambda(u) supera 1 exactamente
cuando mu < 0. Es decir

    i(u) = (-1)^{ #{ autovalores NEGATIVOS del linealizado } }.

Consecuencias que se verifican en el notebook 07:

* u == 0: el linealizado es -v'' + lambda alpha v (porque f'(0) = -alpha < 0),
  con autovalores k^2 pi^2 + lambda alpha, todos positivos. Luego m = 0 e
  i(0) = +1, para todo lambda > 0.
* Solucion maximal (rama superior): mu_1 > 0, ningun autovalor negativo,
  i = +1. Es la solucion estable.
* Solucion intermedia (rama inferior): exactamente un autovalor negativo,
  i = -1. Es el umbral de extincion.

Y la identidad global: como T_lambda tiene imagen acotada (gracias al
truncamiento), es homotopico a la aplicacion nula en toda bola grande, asi que

    deg(I - T_lambda, B_R, 0) = 1

para R suficientemente grande. Por la propiedad de aditividad, la suma de los
indices de TODAS las soluciones contenidas en B_R vale 1:

    i(0) + i(u_inestable) + i(u_maximal) = 1 + (-1) + 1 = 1.

Esa cuenta es la que fuerza la existencia de la tercera solucion: si solo
hubiera u == 0 y la maximal, la suma daria 2 y no 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ReporteIndice:
    """Indice de una solucion junto con el espectro que lo justifica.

    Atributos
    ---------
    index : int
        El indice de Leray-Schauder, +1 o -1.
    n_negative : int
        Cantidad de autovalores negativos del linealizado; el indice es
        (-1) elevado a esta cantidad.
    eigenvalues : ndarray
        Los primeros autovalores calculados, en orden creciente, para poder
        mostrarlos en el notebook.
    degenerate : bool
        True si algun autovalor esta a distancia menor que la tolerancia de 0.
        En ese caso la solucion es DEGENERADA, el indice no esta definido por
        esta formula y hay que reportarlo en lugar de devolver un numero
        cualquiera. Ocurre exactamente en el pliegue, lambda = lambda*.
    norm : float
        ||u||_infinito, para identificar de que solucion se trata.
    """

    index: int
    n_negative: int
    eigenvalues: Array
    degenerate: bool
    norm: float


@dataclass(frozen=True)
class ReporteGrado:
    """Verificacion de que la suma de los indices reproduce el grado total.

    Atributos
    ---------
    indices : tuple[ReporteIndice, ...]
        Un reporte por solucion, en el mismo orden en que se pasaron.
    suma : int
        Suma de los indices.
    grado_total : int
        El grado deg(I - T_lambda, B_R, 0), que vale 1 por el argumento de
        homotopia.
    coincide : bool
        True si `suma == grado_total`. Si da False, o falta alguna solucion en
        la lista, o alguna es degenerada, o hay un error numerico: en los tres
        casos es informacion util y no debe silenciarse.
    incluye_trivial : bool
        True si la solucion trivial u == 0 estaba en la lista. La identidad
        solo cierra si se la incluye, porque tambien ella aporta su +1.
    """

    indices: tuple[ReporteIndice, ...]
    suma: int
    grado_total: int
    coincide: bool
    incluye_trivial: bool


def leray_schauder_index(
    u: Array,
    lam: float,
    alpha: float,
    t: Optional[Array] = None,
    k: int = 10,
    tol: float = 1e-6,
) -> ReporteIndice:
    """Calcula el indice de Leray-Schauder de una solucion contando autovalores del linealizado.

    Procedimiento: obtener los primeros k autovalores del problema linealizado
    con `continuation.linearized_spectrum`, contar cuantos son estrictamente
    negativos (mas alla de `tol`) y devolver (-1) elevado a esa cantidad. Si
    algun autovalor cae dentro de `tol` de 0 hay que marcar `degenerate=True`:
    en un punto de pliegue el indice no esta definido por esta via.

    Como los autovalores crecen como k^2 pi^2 - lambda * max f', con k = 10
    alcanza para no perder ninguno negativo en el rango de lambda de los
    notebooks; conviene igual chequear que el ultimo autovalor calculado sea
    holgadamente positivo, y avisar si no lo es.

    Parametros
    ----------
    u : ndarray de forma (n,)
        Solucion evaluada en la grilla. Se admite la trivial u == 0.
    lam : float
        Parametro lambda > 0.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    t : ndarray de forma (n,), opcional
        Grilla uniforme en [0, 1]; None para `numpy.linspace(0, 1, len(u))`.
    k : int
        Cantidad de autovalores a calcular.
    tol : float
        Umbral por debajo del cual un autovalor se considera nulo (solucion
        degenerada).

    Devuelve
    --------
    ReporteIndice
        Con el indice, el conteo de autovalores negativos y el espectro.
    """
    raise NotImplementedError


def verify_degree_sum(
    solutions: Sequence[Array],
    lam: float,
    alpha: float,
    t: Optional[Array] = None,
    k: int = 10,
    tol: float = 1e-6,
    include_trivial: bool = True,
) -> ReporteGrado:
    """Verifica que la suma de los indices de las soluciones reproduzca el grado total.

    Calcula `leray_schauder_index` para cada solucion de la lista (agregando la
    trivial u == 0 si `include_trivial` es True y no estaba) y compara la suma
    con el grado total deg(I - T_lambda, B_R, 0) = 1.

    Para lambda > lambda*(alpha) la cuenta esperada es

        i(0) + i(u_inestable) + i(u_maximal) = (+1) + (-1) + (+1) = 1,

    y para lambda < lambda*(alpha), donde la unica solucion es la trivial,

        i(0) = +1.

    Si la suma no da 1, la interpretacion correcta no es "el teorema falla"
    sino "falta alguna solucion en la lista, o alguna es degenerada": la
    identidad es un teorema, y aca funciona como test de completitud del
    buscador de soluciones. Esa es exactamente la logica del argumento de
    existencia de la tercera solucion.

    Parametros
    ----------
    solutions : secuencia de ndarray de forma (n,)
        Las soluciones halladas (por ejemplo con `shooting.find_all_solutions`),
        todas en la misma grilla.
    lam : float
        Parametro lambda > 0.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    t : ndarray de forma (n,), opcional
        Grilla uniforme en [0, 1].
    k : int
        Autovalores a calcular por solucion.
    tol : float
        Umbral de degeneracion.
    include_trivial : bool
        Si es True se agrega la solucion trivial u == 0 en caso de que no
        figure ya en `solutions`.

    Devuelve
    --------
    ReporteGrado
        Con los indices individuales, la suma, el grado total y el veredicto.

    Levanta
    -------
    ValueError
        Si las soluciones no comparten la misma longitud de grilla.
    """
    raise NotImplementedError
