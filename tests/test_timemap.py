"""Especificacion ejecutable del mapa del tiempo y del umbral exacto lambda*.

Fija el contrato de theta(alpha), de T(rho) y de lambda*(alpha), incluyendo la
relacion exacta lambda = T(rho)^2 (contrastada contra el shooting, que es un
metodo numerico completamente independiente) y la desigualdad
lambda*(alpha) > 4 pi^2 / (1 - alpha)^2, que muestra que la cota variacional
del resultado 2 es estricta y no optima.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ALPHAS_VALIDOS
from src import nonlinearity as nl, shooting, timemap


def theta_formula_cerrada(a: float) -> float:
    """Raiz menor de 3 u^2 - 4 (1 + alpha) u + 6 alpha, que es theta(alpha)."""
    return (2.0 * (1.0 + a) - np.sqrt(4.0 * (1.0 + a) ** 2 - 18.0 * a)) / 3.0


# --------------------------------------------------------------------------
# theta(alpha): el cero no trivial de F
# --------------------------------------------------------------------------


def test_theta_anula_a_F(alpha):
    """F(theta(alpha)) = 0."""
    th = timemap.theta(alpha)
    assert nl.F(th, alpha) == pytest.approx(0.0, abs=1e-12)


def test_theta_esta_entre_el_umbral_y_la_capacidad_de_carga(alpha):
    """alpha < theta(alpha) < 1: F solo puede anularse donde ya es creciente."""
    th = timemap.theta(alpha)
    assert alpha < th < 1.0


def test_theta_coincide_con_la_formula_cerrada(alpha):
    """Control contra la resolvente de la cuadratica asociada a F."""
    assert timemap.theta(alpha) == pytest.approx(theta_formula_cerrada(alpha), rel=1e-10)


def test_theta_crece_con_alpha():
    """Cuanto mas alto el umbral de Allee, mas alto tiene que ser el pico."""
    valores = [timemap.theta(a) for a in ALPHAS_VALIDOS]
    assert all(x < y for x, y in zip(valores, valores[1:]))


@pytest.mark.parametrize("a", [-0.1, 0.0, 0.5, 0.7])
def test_theta_valida_alpha(a):
    """Fuera de (0, 1/2) no hay theta que tenga sentido."""
    with pytest.raises(ValueError):
        timemap.theta(a)


# --------------------------------------------------------------------------
# El mapa del tiempo T(rho)
# --------------------------------------------------------------------------


def test_time_map_es_positivo_en_el_rango_admisible(alpha):
    """T(rho) > 0 para todo rho en (theta, 1)."""
    th = timemap.theta(alpha)
    for rho in np.linspace(th, 1.0, 12)[1:-1]:
        assert timemap.time_map(float(rho), alpha) > 0.0


def test_time_map_rechaza_rho_fuera_del_rango(alpha):
    """Para rho <= theta o rho >= 1 el radicando F(rho) - F(u) no es positivo."""
    th = timemap.theta(alpha)
    for rho in (0.5 * th, th, 1.0, 1.2):
        with pytest.raises(ValueError):
            timemap.time_map(float(rho), alpha)


def test_time_map_diverge_al_acercarse_a_theta(alpha):
    """T -> +infinito cuando rho -> theta^+: la singularidad en u = 0 deja de ser integrable."""
    th = timemap.theta(alpha)
    valores = [timemap.time_map(th + eps, alpha) for eps in (1e-3, 1e-5, 1e-7)]
    assert all(x < y for x, y in zip(valores, valores[1:]))


def test_time_map_diverge_al_acercarse_a_uno(alpha):
    """T -> +infinito cuando rho -> 1^-: f(1) = 0 y la integral diverge logaritmicamente."""
    valores = [timemap.time_map(1.0 - eps, alpha) for eps in (1e-3, 1e-5, 1e-7)]
    assert all(x < y for x, y in zip(valores, valores[1:]))


def test_time_map_tiene_un_minimo_interior(alpha):
    """Con T infinito en ambos extremos, el minimo esta estrictamente adentro."""
    th = timemap.theta(alpha)
    rhos = np.linspace(th, 1.0, 200)[1:-1]
    valores = timemap.time_map_array(rhos, alpha)
    i = int(np.argmin(valores))
    assert 0 < i < len(rhos) - 1


def test_time_map_array_coincide_con_la_version_escalar(alpha):
    """La version vectorizada es exactamente la escalar aplicada elemento a elemento."""
    th = timemap.theta(alpha)
    rhos = np.linspace(th, 1.0, 15)[1:-1]
    esperado = np.array([timemap.time_map(float(r), alpha) for r in rhos])
    np.testing.assert_allclose(timemap.time_map_array(rhos, alpha), esperado, rtol=1e-10)


@pytest.mark.parametrize("frac", [0.25, 0.5, 0.75])
def test_la_relacion_lambda_igual_T_al_cuadrado(alpha, frac):
    """Control cruzado con el shooting: si lambda = T(rho)^2 hay solucion con maximo rho.

    Se integra el PVI con la pendiente que da la identidad de energia,
    u'(0) = sqrt(2 lambda F(rho)), y se verifica que la trayectoria vuelva a
    cero en t = 1 y alcance exactamente rho en t = 1/2. Son dos metodos sin
    nada en comun: si coinciden, la normalizacion de T es la correcta.
    """
    th = timemap.theta(alpha)
    rho = th + frac * (1.0 - th)
    lam = timemap.time_map(rho, alpha) ** 2

    pendiente = float(np.sqrt(2.0 * lam * nl.F(rho, alpha)))
    tray = shooting.solve_ivp_from_slope(pendiente, lam=lam, alpha=alpha, n_points=2001)

    assert tray.u[-1] == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(tray.u)) == pytest.approx(rho, rel=1e-5)


# --------------------------------------------------------------------------
# lambda*(alpha): el umbral exacto
# --------------------------------------------------------------------------


def test_lambda_star_es_el_cuadrado_del_minimo(alpha):
    """lambda* = T_min^2 y rho* esta en el rango admisible."""
    res = timemap.lambda_star(alpha)
    assert res.lambda_star == pytest.approx(res.t_min**2, rel=1e-10)
    assert res.alpha == alpha
    assert timemap.theta(alpha) < res.rho_star < 1.0


def test_lambda_star_minimiza_efectivamente_el_mapa_del_tiempo(alpha):
    """Ningun rho del rango da un T menor que T_min."""
    res = timemap.lambda_star(alpha)
    th = timemap.theta(alpha)
    rhos = np.linspace(th, 1.0, 500)[1:-1]
    valores = timemap.time_map_array(rhos, alpha)
    assert res.t_min <= float(np.min(valores)) + 1e-8
    assert timemap.time_map(res.rho_star, alpha) == pytest.approx(res.t_min, rel=1e-8)


def test_lambda_star_supera_estrictamente_la_cota_variacional(alpha):
    """lambda*(alpha) > 4 pi^2 / (1 - alpha)^2.

    La cota del resultado 2 es necesaria pero no suficiente: entre ella y
    lambda* hay una franja de lambda donde tampoco existe solucion positiva,
    pese a que el argumento con sin(pi t) no lo detecta.
    """
    res = timemap.lambda_star(alpha)
    assert res.lambda_star > nl.lambda_lower_bound(alpha)


def test_lambda_star_crece_con_alpha():
    """Un umbral de Allee mas alto exige un habitat mas grande."""
    valores = [timemap.lambda_star(a).lambda_star for a in ALPHAS_VALIDOS]
    assert all(x < y for x, y in zip(valores, valores[1:]))


@pytest.mark.parametrize("a", [-0.1, 0.0, 0.5, 0.7])
def test_lambda_star_valida_alpha(a):
    """Fuera de (0, 1/2) no hay umbral que calcular."""
    with pytest.raises(ValueError):
        timemap.lambda_star(a)


# --------------------------------------------------------------------------
# Inversion: cuantas soluciones hay para cada lambda
# --------------------------------------------------------------------------


def test_sin_soluciones_por_debajo_del_umbral(alpha):
    """Para lambda < lambda* no hay ningun rho admisible: la poblacion se extingue."""
    res = timemap.lambda_star(alpha)
    assert timemap.rho_from_lambda(0.5 * res.lambda_star, alpha) == []
    assert timemap.rho_from_lambda(0.99 * res.lambda_star, alpha) == []


@pytest.mark.parametrize("factor", [1.05, 1.5, 3.0])
def test_dos_soluciones_por_encima_del_umbral(alpha, factor):
    """Para lambda > lambda* hay exactamente dos rho, uno a cada lado del pliegue."""
    res = timemap.lambda_star(alpha)
    lam = factor * res.lambda_star
    rhos = timemap.rho_from_lambda(lam, alpha)

    assert len(rhos) == 2
    assert rhos[0] < res.rho_star < rhos[1]
    for rho in rhos:
        assert timemap.time_map(rho, alpha) ** 2 == pytest.approx(lam, rel=1e-6)


def test_las_dos_soluciones_se_juntan_en_el_pliegue(alpha):
    """Al bajar lambda hacia lambda*, los dos rho colapsan en rho*."""
    res = timemap.lambda_star(alpha)
    separaciones = []
    for factor in (1.5, 1.1, 1.01):
        rhos = timemap.rho_from_lambda(factor * res.lambda_star, alpha)
        assert len(rhos) == 2
        separaciones.append(rhos[1] - rhos[0])
    assert all(x > y for x, y in zip(separaciones, separaciones[1:]))


@pytest.mark.parametrize("factor", [1.5, 3.0])
def test_el_shooting_encuentra_las_mismas_dos_soluciones(alpha, factor):
    """Control cruzado final: shooting y mapa del tiempo dan los mismos maximos."""
    res = timemap.lambda_star(alpha)
    lam = factor * res.lambda_star

    esperados = timemap.rho_from_lambda(lam, alpha)
    hallados = sorted(sol.rho for sol in shooting.find_all_solutions(lam=lam, alpha=alpha))

    assert len(hallados) == len(esperados) == 2
    np.testing.assert_allclose(hallados, esperados, rtol=1e-4)
