"""Subsolucion de tres tramos y su verificacion.

La supersolucion es trivial: psi == 1 cumple -psi'' = 0 = lambda f(1) y
psi >= 0 en el borde. Todo el trabajo esta en la subsolucion.

Se busca phi con

    -phi'' <= lambda f(phi) en (0,1),   phi(0) <= 0,   phi(1) <= 0,
    0 <= phi <= 1,   phi no identicamente nula,

para lambda suficientemente grande. La construccion es simetrica respecto de
t = 1/2 y tiene tres tramos en [0, 1/2] (espejados en [1/2, 1]):

  Tramo 1, CONVEXO, de 0 a t1, con phi subiendo de 0 hasta alpha + eta.
      Aca phi < alpha en casi todo el tramo, luego f(phi) < 0, y la desigualdad
      -phi'' <= lambda f(phi) < 0 OBLIGA a phi'' > 0. La subsolucion tiene que
      subir convexamente y rapido. Eleccion explicita: parabola con curvatura
      constante phi'' = lambda C1, donde

          C1 = max_{0 <= u <= alpha} (-f(u)) > 0,

      con lo cual phi'' = lambda C1 >= -lambda f(phi) se cumple en todo el
      tramo por definicion de C1. Da phi(t) = (lambda C1 / 2) t^2 y
      t1 = sqrt(2 (alpha + eta) / (lambda C1)), que tiende a 0 cuando
      lambda -> infinito: es la capa limite junto al borde letal.

  Tramo 2, CONCAVO, de t1 a t2, con phi subiendo de alpha + eta hasta beta.
      Aca f(phi) >= m2 := min_{alpha + eta <= u <= beta} f(u) > 0, y hay margen
      para curvar hacia abajo: alcanza con phi'' = -lambda m2. La derecha eta
      es indispensable, porque f(alpha) = 0 y sin ese margen no habria ninguna
      concavidad admisible pegada al umbral.

      Con curvatura constante -lambda m2, la derivada baja de
      phi'(t1) = sqrt(2 (alpha + eta) lambda C1) hasta 0 en un tiempo
      Delta t = phi'(t1) / (lambda m2), subiendo una altura
      phi'(t1)^2 / (2 lambda m2) = (alpha + eta) C1 / m2. Eso fija

          beta = (alpha + eta) * (1 + C1 / m2),

      condicion que NO depende de lambda: la construccion es factible si y solo
      si se pueden elegir eta > 0 y beta < 1 compatibles. Y Delta t es de orden
      lambda^(-1/2), asi que t2 -> 0: para lambda grande los dos primeros
      tramos entran comodos en [0, 1/2].

  Tramo 3, MESETA, de t2 a 1/2, con phi == beta constante.
      Alli -phi'' = 0 <= lambda f(beta), que vale porque alpha < beta < 1.

ADVERTENCIA (el error clasico, y la razon de ser de este modulo)
---------------------------------------------------------------
La construccion ingenua -una rampa lineal que sube desde 0 y se pega
directamente a la meseta- NO es una subsolucion, y falla por dos motivos
independientes:

  1. En el empalme con la meseta la derivada salta HACIA ABAJO (de un valor
     positivo a 0). Una esquina concava introduce una delta de Dirac
     NEGATIVA en phi'', luego -phi'' tiene una delta POSITIVA que ninguna
     cota lambda f(phi) puede dominar: la desigualdad se viola en sentido
     distribucional exactamente en el punto de empalme. Solo se admiten
     esquinas CONVEXAS (saltos de derivada hacia arriba), porque esas dan
     deltas negativas en -phi'' y la desigualdad sigue valiendo.
     Por eso el empalme con la meseta debe ser C^1: la derivada tiene que
     llegar a cero de manera continua, y para eso hace falta el tramo concavo
     intermedio.

  2. Sobre la rampa phi'' = 0, asi que -phi'' = 0, pero f(phi) < 0 mientras
     phi < alpha. La desigualdad -phi'' <= lambda f(phi) pide 0 <= (negativo)
     y falla en todo el tramo inicial, no solo en un punto.

`build_naive_trapezoid` construye precisamente ese objeto invalido, y
`verify_subsolution` lo tiene que rechazar: es el contraejemplo del notebook
04 y esta cubierto por los tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class SubsolucionNoFactible(RuntimeError):
    """La construccion de tres tramos no entra en el intervalo para ese lambda.

    Se levanta cuando t2 >= 1/2, es decir cuando el tramo convexo mas el
    concavo no caben en [0, 1/2]. Como ambos son de orden lambda^(-1/2), eso
    ocurre exactamente para lambda por debajo del lambda_0(alpha) del
    enunciado del resultado 3, y no es un error del llamador sino el contenido
    de la hipotesis "lambda suficientemente grande".

    Es una excepcion propia y no un RuntimeError pelado para poder testearla
    sin ambiguedad: NotImplementedError tambien hereda de RuntimeError.
    """


@dataclass(frozen=True)
class ParametrosSubsolucion:
    """Parametros libres de la construccion de tres tramos.

    Atributos
    ---------
    eta : float, opcional
        Margen por encima de alpha hasta donde llega el tramo convexo. Debe
        ser > 0: sin margen, f(alpha) = 0 y el tramo concavo no tendria
        curvatura admisible.

        Si es None (por defecto) se elige automaticamente el eta que MINIMIZA
        la altura resultante beta = (alpha + eta) (1 + C1 / m2), es decir el
        que deja mas margen respecto de 1. Esa eleccion automatica importa: con
        un eta fijo chico la construccion deja de ser factible ya para alpha en
        torno a 0.35, porque m2 = f(alpha + eta) se hace demasiado chico y beta
        se dispara por encima de 1.
    beta : float, opcional
        Altura de la meseta, en (alpha + eta, 1). Si es None se toma el valor
        que sale del empalme C^1, beta = (alpha + eta) (1 + C1 / m2), que es el
        minimo compatible con la construccion.
    """

    eta: Optional[float] = None
    beta: Optional[float] = None


@dataclass(frozen=True)
class Subsolucion:
    """Subsolucion de tres tramos, simetrica respecto de t = 1/2.

    Se la representa por sus puntos de quiebre y por los coeficientes de cada
    tramo, de modo que `verify_subsolution` pueda evaluar phi, phi' y phi'' de
    manera exacta en cada tramo (sin derivar numericamente) y examinar los
    empalmes uno por uno.

    Atributos
    ---------
    breakpoints : tuple[float, ...]
        Puntos de quiebre 0 = s0 < s1 < ... < sk = 1, incluidos los espejados.
        Para la construccion de tres tramos son (0, t1, t2, 1 - t2, 1 - t1, 1).
    lam : float
        Parametro lambda para el que se construyo.
    alpha : float
        Umbral de Allee.
    beta : float
        Altura de la meseta.
    eta : float
        Margen usado por encima de alpha.
    curvatures : tuple[float, ...]
        Curvatura constante phi'' de cada tramo, en el mismo orden que los
        intervalos definidos por `breakpoints`. Para la construccion nominal:
        (+lambda C1, -lambda m2, 0, -lambda m2, +lambda C1).
    valid_by_construction : bool
        True para la construccion de tres tramos, False para el trapecio
        ingenuo de `build_naive_trapezoid`. Es solo un rotulo informativo:
        `verify_subsolution` no lo mira y decide por su cuenta.
    """

    breakpoints: tuple[float, ...]
    lam: float
    alpha: float
    beta: float
    eta: float
    curvatures: tuple[float, ...]
    valid_by_construction: bool = True
    extra: dict = field(default_factory=dict)

    def __call__(self, t: Array | float) -> Array | float:
        """Evalua phi(t) tramo por tramo.

        Parametros
        ----------
        t : float o ndarray
            Puntos de [0, 1] donde evaluar.

        Devuelve
        --------
        float o ndarray
            phi(t), de la misma forma que `t`. Debe cumplir phi(0) = phi(1) = 0
            y 0 <= phi <= beta <= 1.
        """
        raise NotImplementedError

    def derivative(self, t: Array | float) -> Array | float:
        """Evalua phi'(t) tramo por tramo.

        En los puntos de quiebre devuelve la derivada por derecha; para
        detectar saltos hay que comparar los limites laterales, que es lo que
        hace `verify_subsolution`.

        Parametros
        ----------
        t : float o ndarray
            Puntos de [0, 1] donde evaluar.

        Devuelve
        --------
        float o ndarray
            phi'(t), de la misma forma que `t`.
        """
        raise NotImplementedError

    def second_derivative(self, t: Array | float) -> Array | float:
        """Evalua phi''(t) en el interior de cada tramo (curvatura constante).

        No incluye las eventuales deltas de Dirac de los puntos de quiebre: esa
        parte singular se analiza por separado mirando los saltos de phi'. La
        distincion es justamente el nucleo de la advertencia del modulo.

        Parametros
        ----------
        t : float o ndarray
            Puntos de [0, 1] donde evaluar, preferentemente interiores a los
            tramos.

        Devuelve
        --------
        float o ndarray
            phi''(t), de la misma forma que `t`.
        """
        raise NotImplementedError

    def derivative_jumps(self) -> tuple[float, ...]:
        """Devuelve los saltos phi'(s+) - phi'(s-) en cada punto de quiebre interior.

        Un salto positivo es una esquina convexa y es ADMISIBLE (aporta una
        delta positiva a phi'', o sea negativa a -phi''). Un salto negativo es
        una esquina concava y INVALIDA la subsolucion. La construccion de tres
        tramos tiene todos los saltos nulos (empalmes C^1).

        Devuelve
        --------
        tuple[float, ...]
            Un salto por cada punto de quiebre interior, en orden creciente de
            t. Longitud len(breakpoints) - 2.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ReporteSubsolucion:
    """Resultado de verificar una candidata a subsolucion.

    Atributos
    ---------
    es_subsolucion : bool
        Veredicto global: True solo si se cumplen a la vez la desigualdad
        diferencial en el interior de todos los tramos, las condiciones de
        borde, las cotas 0 <= phi <= 1 y la ausencia de esquinas concavas.
    residuo_maximo : float
        max sobre t de (-phi''(t) - lambda f(phi(t))). Debe ser <= tol para que
        la desigualdad diferencial valga. Un valor positivo grande senala en
        que tramo falla.
    residuo_por_tramo : tuple[float, ...]
        El mismo residuo maximo, desagregado por tramo, para saber cual es el
        que rompe.
    saltos_derivada : tuple[float, ...]
        Los saltos de phi' en los quiebres interiores (ver
        `Subsolucion.derivative_jumps`).
    salto_concavo : bool
        True si algun salto es negativo mas alla de la tolerancia, es decir si
        hay una esquina concava. Es el diagnostico del error clasico.
    cumple_borde : bool
        True si phi(0) <= 0 y phi(1) <= 0.
    cumple_cotas : bool
        True si 0 <= phi <= 1 en toda la grilla.
    detalle : dict
        Informacion auxiliar para los graficos del notebook 04: la grilla
        usada, el residuo punto a punto, los puntos de quiebre.
    """

    es_subsolucion: bool
    residuo_maximo: float
    residuo_por_tramo: tuple[float, ...]
    saltos_derivada: tuple[float, ...]
    salto_concavo: bool
    cumple_borde: bool
    cumple_cotas: bool
    detalle: dict


def build_subsolution(
    lam: float,
    alpha: float,
    params: Optional[ParametrosSubsolucion] = None,
) -> Subsolucion:
    """Construye la subsolucion de tres tramos empalmada en C^1.

    Sigue la receta descrita en el docstring del modulo:

    1. Calcular C1 = max_{[0, alpha]} (-f) y m2 = min_{[alpha + eta, beta]} f,
       ambas constantes que solo dependen de alpha (y de eta, beta).
    2. Tramo convexo en [0, t1]: phi(t) = (lambda C1 / 2) t^2, hasta la altura
       alpha + eta, con t1 = sqrt(2 (alpha + eta) / (lambda C1)).
    3. Tramo concavo en [t1, t2]: parabola con phi'' = -lambda m2, que lleva la
       derivada de phi'(t1) = lambda C1 t1 hasta 0 subiendo exactamente
       (alpha + eta) C1 / m2, de donde beta = (alpha + eta) (1 + C1 / m2) si el
       usuario no fijo beta.
    4. Meseta en [t2, 1/2] a altura beta, y todo espejado en [1/2, 1].

    Los empalmes son C^1 por construccion: en t1 las dos parabolas comparten
    valor y derivada, y en t2 la derivada llega a 0 de forma continua, que es
    justo lo que la meseta necesita. NUNCA hay que pegar la meseta a un tramo
    con derivada positiva (ver la advertencia del modulo).

    La construccion existe solo si lambda es suficientemente grande: hace falta
    t2 < 1/2, y como t1 y t2 - t1 son ambos de orden lambda^(-1/2), eso se
    cumple para todo lambda >= lambda_0(alpha). Ese lambda_0 es el que aparece
    en el enunciado del resultado 3.

    Parametros
    ----------
    lam : float
        Parametro lambda > 0. Debe ser grande; ver `Levanta`.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    params : ParametrosSubsolucion, opcional
        Parametros libres eta y beta. Si es None se usan los valores por
        defecto de la dataclass, es decir eta y beta elegidos automaticamente.

    Devuelve
    --------
    Subsolucion
        Con `valid_by_construction=True`. Deberia pasar `verify_subsolution`.

    Levanta
    -------
    ValueError
        Si alpha esta fuera de (0, 1/2); si eta <= 0; o si el beta resultante
        (o el pedido) no cae en (alpha + eta, 1). Este ultimo caso NO es un
        error del llamador sino un limite real de la receta parabolica: para
        alpha suficientemente cerca de 1/2 ningun eta da beta < 1, porque
        min_x [ x + C1 / ((x - alpha)(1 - x)) ] > 1. Ahi hay que reemplazar la
        parabola del tramo concavo por la solucion exacta de la EDO, que gasta
        mucha menos altura para frenar la derivada. El mensaje de error tiene
        que decir esto, no solo 'beta invalido'.
    SubsolucionNoFactible
        Si lambda es demasiado chico y t2 >= 1/2, es decir si los dos primeros
        tramos no entran en la mitad del intervalo. El mensaje deberia informar
        el lambda_0 minimo que si funcionaria.
    """
    raise NotImplementedError


def build_naive_trapezoid(
    lam: float,
    alpha: float,
    beta: float = 0.6,
    t1: float = 0.1,
) -> Subsolucion:
    """Construye el trapecio ingenuo: rampa lineal pegada a una meseta. NO es subsolucion.

    Es decir phi lineal de 0 a beta en [0, t1], constante igual a beta en
    [t1, 1 - t1], y lineal de beta a 0 en [1 - t1, 1]. Existe unicamente para
    exhibir el contraejemplo del notebook 04 y para que el test lo rechace.

    Falla por dos motivos independientes (ver la advertencia del modulo):
    la esquina concava en t1, donde phi' salta hacia abajo y aparece una delta
    de Dirac que rompe la desigualdad en sentido distribucional; y el tramo
    inicial, donde phi'' = 0 pero f(phi) < 0 mientras phi < alpha.

    Parametros
    ----------
    lam : float
        Parametro lambda > 0 (solo se guarda; el objeto es invalido para
        cualquier lambda).
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    beta : float
        Altura de la meseta, en (alpha, 1).
    t1 : float
        Ancho de la rampa, en (0, 1/2).

    Devuelve
    --------
    Subsolucion
        Con `valid_by_construction=False`. `verify_subsolution` debe devolver
        `es_subsolucion=False` con `salto_concavo=True`.
    """
    raise NotImplementedError


def verify_subsolution(
    u: Subsolucion,
    lam: float,
    alpha: float,
    n_points: int = 2001,
    tol: float = 1e-8,
) -> ReporteSubsolucion:
    """Verifica que una candidata sea efectivamente subsolucion, tramo por tramo.

    Chequea, por separado y sin usar el rotulo `valid_by_construction`:

    1. **Desigualdad diferencial** en el interior de cada tramo: sobre una
       grilla que EXCLUYE los puntos de quiebre (para no mezclar la parte
       regular con la singular), evaluar el residuo
       r(t) = -phi''(t) - lambda f(phi(t)) y pedir r <= tol. Se reporta el
       maximo global y el maximo por tramo.
    2. **Continuidad de la derivada en los empalmes**: calcular los saltos
       phi'(s+) - phi'(s-) en cada quiebre interior. Un salto negativo (esquina
       concava) invalida la subsolucion aunque el residuo interior sea perfecto,
       porque la delta de Dirac correspondiente rompe la desigualdad
       distribucional. Un salto positivo (esquina convexa) es admisible.
    3. **Condiciones de borde**: phi(0) <= 0 y phi(1) <= 0.
    4. **Cotas**: 0 <= phi <= 1, necesarias para que phi <= psi == 1 y el
       intervalo de orden [phi, 1] no sea vacio.

    El veredicto `es_subsolucion` es la conjuncion de los cuatro. Que el punto 2
    sea un chequeo aparte, y no un subproducto del punto 1, es exactamente la
    leccion de este modulo: una discretizacion ciega del residuo sobre una
    grilla uniforme no ve la delta de Dirac y aprueba una candidata invalida.

    Parametros
    ----------
    u : Subsolucion
        La candidata a verificar.
    lam : float
        Parametro lambda > 0 con el que se verifica la desigualdad. Deberia
        coincidir con `u.lam`; si no coincide, se usa este y se deja constancia
        en `detalle`.
    alpha : float
        Umbral de Allee, 0 < alpha < 1/2.
    n_points : int
        Cantidad total de puntos de la grilla de verificacion, repartidos entre
        los tramos.
    tol : float
        Tolerancia con la que se aceptan el residuo y los saltos de derivada.

    Devuelve
    --------
    ReporteSubsolucion
        Con el veredicto y todo el diagnostico desagregado.
    """
    raise NotImplementedError
