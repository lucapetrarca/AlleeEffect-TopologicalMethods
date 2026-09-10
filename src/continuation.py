"""Continuacion de la rama de soluciones y primer autovalor del linealizado.

El teorema de la funcion implicita en espacios de Banach, aplicado a

    Phi(lambda, u) = u'' + lambda f(u),   Phi : R x X -> Y,
    X = {u en C^2([0,1]) : u(0) = u(1) = 0},   Y = C([0,1]),

da una rama local C^1 de soluciones lambda -> u_lambda mientras
D_u Phi(lambda, u) v = v'' + lambda f'(u) v sea isomorfismo, o sea mientras 0
no sea autovalor del problema linealizado

    L_lambda v = -v'' - lambda f'(u_lambda) v = mu v,   v(0) = v(1) = 0.

Derivando la ecuacion respecto de lambda se obtiene la ecuacion de la rama
L_lambda (du/dlambda) = f(u_lambda).

El punto donde el primer autovalor mu_1 se anula es el PLIEGUE (fold) del
diagrama de bifurcacion: alli la rama tiene tangente vertical en el plano
(lambda, ||u||_infinito), la funcion implicita deja de aplicarse y la
parametrizacion por lambda se rompe. La salida estandar es reparametrizar por
LONGITUD DE ARCO: se agrega lambda como incognita y una ecuacion escalar que
fija el paso a lo largo de la curva, y el sistema ampliado sigue siendo
regular en el pliegue. Eso es lo que hace `pseudo_arclength`, y es lo que
permite trazar de un tirar la S completa del notebook 05.

Sobre la rama superior mu_1 > 0 (solucion estable, indice +1); sobre la rama
inferior mu_1 < 0 (solucion inestable, indice -1). El cambio de signo ocurre
exactamente en el pliegue, en lambda = lambda*(alpha).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Rama:
    """Rama de soluciones trazada por continuacion.

    Atributos
    ---------
    t : ndarray de forma (n,)
        Grilla espacial en [0, 1].
    lambdas : ndarray de forma (m,)
        Valor de lambda en cada punto de la rama, en el orden en que fueron
        calculados. NO es monotono: decrece al pasar el pliegue y vuelve a
        crecer, que es justamente lo que hace visible la forma de S.
    solutions : ndarray de forma (m, n)
        Las soluciones u correspondientes, evaluadas en la grilla.
    norms : ndarray de forma (m,)
        ||u||_infinito de cada solucion, la ordenada del diagrama de
        bifurcacion.
    first_eigenvalues : ndarray de forma (m,)
        Primer autovalor mu_1 del linealizado en cada punto de la rama. Cambia
        de signo exactamente una vez, en el pliegue.
    fold_index : int or None
        Indice del punto de la rama mas cercano al pliegue (donde mu_1 cambia
        de signo), o None si la rama calculada no lo contiene.
    alpha : float
        Umbral de Allee.
    """

    t: Array
    lambdas: Array
    solutions: Array
    norms: Array
    first_eigenvalues: Array
    fold_index: Optional[int]
    alpha: float


def linearized_first_eigenvalue(
    u: Array,
    lam: float,
    alpha: float,
    t: Optional[Array] = None,
) -> float:
    """Devuelve el primer autovalor mu_1 del problema linealizado en u.

    Resuelve el problema de autovalores de Dirichlet

        -v'' - lambda f'(u(t)) v = mu v,    v(0) = v(1) = 0,

    discretizando -d^2/dt^2 por diferencias finitas de segundo orden en los
    nodos interiores (matriz tridiagonal (2, -1) / h^2) y restando la diagonal
    lambda f'(u). El menor autovalor de esa matriz simetrica -que se obtiene
    con `scipy.linalg.eigh_tridiagonal` pidiendo solo el primero, o con
    `numpy.linalg.eigvalsh` si la grilla es chica- aproxima mu_1 con error
    O(h^2).

    Significado del signo:

    * mu_1 > 0: la solucion es linealmente estable como estado estacionario de
      la ecuacion parabolica asociada, el operador L_lambda es invertible con
      inversa positiva, la funcion implicita se aplica y el indice de
      Leray-Schauder vale +1. Es la rama maximal.
    * mu_1 < 0: solucion inestable; sobre la rama inferior hay exactamente un
      autovalor negativo, luego el indice vale -1.
    * mu_1 = 0: punto de pliegue. La funcion implicita no se aplica y hay que
      pasar a pseudo-longitud de arco.

    Parametros
    ----------
    u : ndarray de forma (n,)
        Solucion (o candidata) evaluada en la grilla.
    lam : float
        Parametro lambda > 0.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    t : ndarray de forma (n,), opcional
        Grilla en [0, 1]. Si es None se usa `numpy.linspace(0, 1, len(u))`. Se
        supone uniforme; con grilla no uniforme hay que cambiar el esquema.

    Devuelve
    --------
    float
        La aproximacion de mu_1.

    Levanta
    -------
    ValueError
        Si `u` y `t` no tienen la misma longitud o si la grilla tiene menos de
        tres puntos.
    """
    raise NotImplementedError


def linearized_spectrum(
    u: Array,
    lam: float,
    alpha: float,
    t: Optional[Array] = None,
    k: int = 10,
) -> Array:
    """Devuelve los primeros k autovalores del problema linealizado en u.

    Misma discretizacion que `linearized_first_eigenvalue`, pero devolviendo
    varios autovalores. La cantidad de autovalores NEGATIVOS es la que define
    el indice de Leray-Schauder en `indices.leray_schauder_index`, asi que k
    tiene que ser suficientemente grande como para capturarlos a todos; como
    los autovalores del linealizado crecen como k^2 pi^2, con k del orden de 10
    alcanza sobradamente para los lambda de los notebooks.

    Parametros
    ----------
    u : ndarray de forma (n,)
        Solucion evaluada en la grilla.
    lam : float
        Parametro lambda > 0.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    t : ndarray de forma (n,), opcional
        Grilla uniforme en [0, 1].
    k : int
        Cantidad de autovalores a devolver.

    Devuelve
    --------
    ndarray de forma (k,)
        Los k autovalores mas chicos, en orden creciente.
    """
    raise NotImplementedError


def pseudo_arclength(
    lam0: float,
    u0: Array,
    alpha: float,
    t: Optional[Array] = None,
    ds: float = 0.05,
    n_steps: int = 400,
    tol: float = 1e-10,
    max_newton: int = 20,
) -> Rama:
    """Traza la rama completa de soluciones, pliegue incluido, por pseudo-longitud de arco.

    Discretizando el problema en la grilla se obtiene un sistema
    H(u, lambda) = 0 con u en R^n y lambda en R: una ecuacion menos que
    incognitas, o sea una curva. La continuacion natural (fijar lambda y
    resolver por Newton en u) se rompe en el pliegue, donde la jacobiana
    respecto de u es singular. La pseudo-longitud de arco agrega la ecuacion
    escalar

        (u - u_k) . du_k + (lambda - lambda_k) dlambda_k - ds = 0,

    donde (du_k, dlambda_k) es el vector tangente unitario en el punto k. El
    sistema ampliado (n + 1) x (n + 1) es regular en el pliegue -su jacobiana
    tiene la forma de una orlada- asi que Newton converge tambien alli.

    Esquema de cada paso:

    1. Calcular el tangente resolviendo el sistema ampliado con lado derecho
       (0, ..., 0, 1), y normalizarlo; elegir el signo que continue la
       direccion del paso anterior (para no volver sobre lo andado al doblar
       en el pliegue).
    2. Predecir: (u, lambda) = (u_k, lambda_k) + ds * tangente.
    3. Corregir con Newton sobre el sistema ampliado hasta `tol`.
    4. Guardar el punto, evaluar `linearized_first_eigenvalue` y seguir.

    Arrancar desde un punto conocido de la rama superior -por ejemplo la
    solucion que da `monotone.monotone_iteration` desde psi == 1 para un lambda
    grande- y recorrer hacia lambda decreciente: la rama sube, dobla en el
    pliegue lambda = lambda*(alpha) y vuelve hacia lambda creciente por la rama
    inferior, dibujando la S. El pliegue calculado asi debe coincidir con
    `timemap.lambda_star(alpha)`, y esa coincidencia es el mejor control
    cruzado de todo el repositorio: son dos metodos numericos sin nada en
    comun.

    Parametros
    ----------
    lam0 : float
        Valor inicial de lambda, sobre la rama, tipicamente bastante mayor que
        lambda*(alpha).
    u0 : ndarray de forma (n,)
        Solucion inicial correspondiente a lam0, ya convergida.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    t : ndarray de forma (n,), opcional
        Grilla uniforme en [0, 1]. Si es None se usa
        `numpy.linspace(0, 1, len(u0))`.
    ds : float
        Paso de arco. Chico cerca del pliegue; conviene adaptarlo segun la
        cantidad de iteraciones que consuma Newton.
    n_steps : int
        Cantidad maxima de pasos de continuacion.
    tol : float
        Tolerancia del corrector de Newton.
    max_newton : int
        Tope de iteraciones de Newton por paso. Si se agota, lo razonable es
        reducir `ds` y reintentar antes de abandonar.

    Devuelve
    --------
    Rama
        Con los lambda, las soluciones, sus normas, los primeros autovalores y
        el indice del pliegue.

    Levanta
    -------
    ValueError
        Si `u0` y `t` no tienen la misma longitud, o si ds <= 0.
    RuntimeError
        Si Newton no converge ni siquiera reduciendo el paso.
    """
    raise NotImplementedError
