"""Paquete de soporte para el estudio del problema de Dirichlet con efecto Allee.

    -u''(t) = lambda * u(t) * (u(t) - alpha) * (1 - u(t)),   t en (0, 1),
     u(0) = u(1) = 0,

con lambda > 0 y 0 < alpha < 1/2.

Modulos
-------
nonlinearity : la reaccion f, su derivada, su primitiva F y las constantes
               M(alpha) y 4*pi^2/(1-alpha)^2.
green        : nucleo de Green del operador -d^2/dt^2 con Dirichlet, el
               operador de punto fijo T_lambda y el truncamiento de f.
shooting     : metodo de disparo para localizar todas las soluciones del BVP.
timemap      : mapa del tiempo T(rho) y el umbral exacto lambda*(alpha).
subsuper     : construccion y verificacion de la subsolucion de tres tramos.
monotone     : iteracion monotona desde la supersolucion.
continuation : continuacion por pseudo-longitud de arco y primer autovalor
               del problema linealizado.
indices      : indice de Leray-Schauder de cada solucion y suma = grado.
plotting     : estilo comun de las figuras.

Todo el paquete esta en estado de esqueleto: las firmas y los docstrings son
definitivos, los cuerpos levantan NotImplementedError.
"""

__all__ = [
    "nonlinearity",
    "green",
    "shooting",
    "timemap",
    "subsuper",
    "monotone",
    "continuation",
    "indices",
    "plotting",
]
