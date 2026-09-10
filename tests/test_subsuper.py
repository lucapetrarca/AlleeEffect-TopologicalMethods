"""Especificacion ejecutable de la subsolucion de tres tramos.

Dos cosas se fijan aca:

* que la construccion de `build_subsolution` sea efectivamente una subsolucion
  (desigualdad diferencial tramo por tramo, empalmes C^1, borde y cotas);
* que la construccion INGENUA -rampa lineal pegada a una meseta- sea
  RECHAZADA, por su esquina concava y por el tramo inicial con curvatura nula
  donde f < 0. Ese rechazo es el contenido pedagogico central del modulo.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import nonlinearity as nl, subsuper


# La receta parabolica del tramo concavo gasta altura para frenar la derivada,
# y esa altura crece con alpha: para alpha cerca de 1/2 ningun eta da beta < 1
# y hace falta resolver el tramo intermedio con la EDO exacta. Los tests de
# construccion se quedan en el rango donde la receta parabolica es factible.
ALPHAS_FACTIBLES = [0.05, 0.15, 0.25, 0.35]

# lambda suficientemente grande para que los dos primeros tramos entren en
# [0, 1/2]: ambos son de orden lambda^(-1/2).
LAMBDAS_GRANDES = [5_000.0, 20_000.0]


# --------------------------------------------------------------------------
# La supersolucion
# --------------------------------------------------------------------------


def test_la_constante_uno_es_supersolucion(alpha):
    """psi == 1 cumple -psi'' = 0 >= lambda f(1) = 0 y psi >= 0 en el borde."""
    assert nl.f(1.0, alpha) == pytest.approx(0.0, abs=1e-14)


# --------------------------------------------------------------------------
# Construccion de la subsolucion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_subsolucion_se_anula_en_el_borde(a, lam):
    """phi(0) = phi(1) = 0, con lo cual phi(0) <= 0 y phi(1) <= 0."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    assert phi(0.0) == pytest.approx(0.0, abs=1e-12)
    assert phi(1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_subsolucion_esta_bajo_la_supersolucion(a, lam):
    """0 <= phi <= beta < 1 = psi, para que el intervalo de orden no sea vacio."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    t = np.linspace(0.0, 1.0, 4001)
    valores = np.asarray(phi(t))
    assert np.all(valores >= -1e-12)
    assert np.all(valores <= 1.0 + 1e-12)
    assert phi.beta < 1.0
    assert float(np.max(valores)) == pytest.approx(phi.beta, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_subsolucion_no_es_identicamente_nula(a, lam):
    """Una subsolucion nula no sirve de nada: la meseta esta sobre el umbral."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    assert phi.beta > a
    assert float(np.max(np.asarray(phi(np.linspace(0.0, 1.0, 1001))))) > a


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_subsolucion_es_simetrica_y_creciente_hasta_la_mitad(a, lam):
    """phi(t) = phi(1-t), y sube de 0 a beta en [0, 1/2]."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    t = np.linspace(0.0, 0.5, 1001)
    izq = np.asarray(phi(t))
    der = np.asarray(phi(1.0 - t))
    np.testing.assert_allclose(izq, der, rtol=1e-10, atol=1e-12)
    assert np.all(np.diff(izq) >= -1e-12)


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_los_tramos_tienen_las_curvaturas_del_esquema(a, lam):
    """Convexo, concavo, meseta, concavo, convexo: signos (+, -, 0, -, +)."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    assert len(phi.breakpoints) == 6
    assert list(phi.breakpoints) == sorted(phi.breakpoints)
    assert phi.breakpoints[0] == pytest.approx(0.0, abs=1e-14)
    assert phi.breakpoints[-1] == pytest.approx(1.0, abs=1e-14)
    assert phi.breakpoints[2] < 0.5 < phi.breakpoints[3]

    c = phi.curvatures
    assert len(c) == 5
    assert c[0] > 0.0
    assert c[1] < 0.0
    assert c[2] == pytest.approx(0.0, abs=1e-14)
    assert c[3] < 0.0
    assert c[4] > 0.0


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_los_empalmes_son_c1(a, lam):
    """Ningun salto de derivada: la construccion es C^1 en todos los quiebres.

    Es la condicion que la construccion ingenua viola.
    """
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    saltos = phi.derivative_jumps()
    assert len(saltos) == len(phi.breakpoints) - 2
    assert np.allclose(saltos, 0.0, atol=1e-8)


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_derivada_se_anula_al_entrar_en_la_meseta(a, lam):
    """phi'(t2) = 0: la derivada llega a cero de manera continua, no de golpe."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    t2 = phi.breakpoints[2]
    assert phi.derivative(t2 - 1e-9) == pytest.approx(0.0, abs=1e-5)
    assert phi.derivative(0.5 * (t2 + phi.breakpoints[3])) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Verificacion de la subsolucion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_construccion_pasa_la_verificacion(a, lam):
    """El veredicto global y cada uno de sus cuatro ingredientes."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    rep = subsuper.verify_subsolution(phi, lam=lam, alpha=a)

    assert rep.es_subsolucion is True
    assert rep.residuo_maximo <= 1e-8
    assert rep.salto_concavo is False
    assert rep.cumple_borde is True
    assert rep.cumple_cotas is True


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_desigualdad_vale_en_cada_tramo_por_separado(a, lam):
    """-phi'' <= lambda f(phi) en el interior de los cinco tramos."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    rep = subsuper.verify_subsolution(phi, lam=lam, alpha=a)
    assert len(rep.residuo_por_tramo) == 5
    assert all(r <= 1e-8 for r in rep.residuo_por_tramo)


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
@pytest.mark.parametrize("lam", LAMBDAS_GRANDES)
def test_la_desigualdad_vale_tambien_calculada_a_mano(a, lam):
    """Recalculo independiente del residuo, sin pasar por verify_subsolution."""
    phi = subsuper.build_subsolution(lam=lam, alpha=a)
    quiebres = np.asarray(phi.breakpoints)
    for izq, der in zip(quiebres[:-1], quiebres[1:]):
        t = np.linspace(izq, der, 400)[1:-1]  # interior del tramo
        residuo = -np.asarray(phi.second_derivative(t)) - lam * nl.f(np.asarray(phi(t)), a)
        assert float(np.max(residuo)) <= 1e-8


# --------------------------------------------------------------------------
# El contraejemplo: la construccion ingenua NO es subsolucion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
def test_el_trapecio_ingenuo_es_rechazado(a):
    """La rampa pegada a la meseta no es subsolucion, para ningun lambda."""
    for lam in (100.0, 5_000.0, 100_000.0):
        malo = subsuper.build_naive_trapezoid(lam=lam, alpha=a)
        rep = subsuper.verify_subsolution(malo, lam=lam, alpha=a)
        assert rep.es_subsolucion is False


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
def test_el_trapecio_ingenuo_tiene_una_esquina_concava(a):
    """Motivo 1 del rechazo: phi' salta hacia abajo al entrar en la meseta.

    Ese salto negativo es una delta de Dirac negativa en phi'', o sea positiva
    en -phi'', y ninguna cota lambda f(phi) la domina.
    """
    lam = 5_000.0
    malo = subsuper.build_naive_trapezoid(lam=lam, alpha=a)
    rep = subsuper.verify_subsolution(malo, lam=lam, alpha=a)

    assert rep.salto_concavo is True
    assert min(rep.saltos_derivada) < -1e-6


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
def test_el_trapecio_ingenuo_viola_la_desigualdad_en_la_rampa(a):
    """Motivo 2 del rechazo: en la rampa phi'' = 0 pero f(phi) < 0 si phi < alpha."""
    lam = 5_000.0
    malo = subsuper.build_naive_trapezoid(lam=lam, alpha=a)
    rep = subsuper.verify_subsolution(malo, lam=lam, alpha=a)

    assert rep.residuo_maximo > 0.0
    assert rep.residuo_por_tramo[0] > 0.0


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
def test_el_trapecio_ingenuo_si_cumple_borde_y_cotas(a):
    """El trapecio falla SOLO por la desigualdad y la esquina, no por lo demas.

    Es lo que lo vuelve un contraejemplo instructivo: a simple vista parece una
    subsolucion perfectamente razonable.
    """
    lam = 5_000.0
    malo = subsuper.build_naive_trapezoid(lam=lam, alpha=a)
    rep = subsuper.verify_subsolution(malo, lam=lam, alpha=a)

    assert rep.cumple_borde is True
    assert rep.cumple_cotas is True
    assert malo.valid_by_construction is False


# --------------------------------------------------------------------------
# Limites de la construccion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
def test_lambda_chico_no_admite_la_construccion(a):
    """Si lambda es chico los dos primeros tramos no entran en [0, 1/2].

    Es el lambda_0(alpha) del enunciado: la subsolucion existe solo para
    lambda suficientemente grande, que es exactamente lo que dice el teorema.
    """
    with pytest.raises(subsuper.SubsolucionNoFactible):
        subsuper.build_subsolution(lam=1.0, alpha=a)


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
def test_eta_no_positivo_es_invalido(a):
    """Sin margen sobre alpha no hay curvatura concava admisible, porque f(alpha) = 0."""
    with pytest.raises(ValueError):
        subsuper.build_subsolution(
            lam=5_000.0, alpha=a, params=subsuper.ParametrosSubsolucion(eta=0.0)
        )


@pytest.mark.parametrize("a", [-0.1, 0.0, 0.5, 0.8])
def test_alpha_invalido_es_rechazado(a):
    """La construccion valida alpha antes de calcular nada."""
    with pytest.raises(ValueError):
        subsuper.build_subsolution(lam=5_000.0, alpha=a)


@pytest.mark.parametrize("a", ALPHAS_FACTIBLES)
def test_beta_fuera_de_rango_es_invalido(a):
    """Una meseta a altura >= 1 no sirve: alli f(beta) <= 0."""
    with pytest.raises(ValueError):
        subsuper.build_subsolution(
            lam=5_000.0, alpha=a, params=subsuper.ParametrosSubsolucion(eta=0.05, beta=1.5)
        )
