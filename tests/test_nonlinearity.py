"""Especificacion ejecutable de `src.nonlinearity`.

Fija el contrato de f, f', F y de las dos constantes M(alpha) y
4 pi^2 / (1 - alpha)^2. Estos tests fallan hasta que las funciones esten
implementadas: eso es lo esperado en el estado de esqueleto del repositorio.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ALPHAS_INVALIDOS, ALPHAS_VALIDOS
from src import nonlinearity as nl


# --------------------------------------------------------------------------
# Validacion de alpha
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", ALPHAS_VALIDOS)
def test_validate_alpha_acepta_el_rango_admisible(a):
    """Para 0 < alpha < 1/2 la validacion pasa sin levantar nada."""
    assert nl.validate_alpha(a) is None


@pytest.mark.parametrize("a", ALPHAS_INVALIDOS)
def test_validate_alpha_rechaza_fuera_del_rango(a):
    """alpha <= 0 y alpha >= 1/2 son inadmisibles y deben levantar ValueError."""
    with pytest.raises(ValueError):
        nl.validate_alpha(a)


def test_alpha_max_es_un_medio():
    """El limite superior documentado es 1/2, el que hace F(1) > 0."""
    assert nl.ALPHA_MAX == 0.5


# --------------------------------------------------------------------------
# La no linealidad f
# --------------------------------------------------------------------------


def test_f_se_anula_en_sus_tres_ceros(alpha):
    """f(u) = u (u - alpha) (1 - u) tiene ceros exactamente en 0, alpha y 1."""
    for cero in (0.0, alpha, 1.0):
        assert nl.f(cero, alpha) == pytest.approx(0.0, abs=1e-14)


def test_f_tiene_el_signo_del_efecto_allee(alpha):
    """f < 0 por debajo del umbral, f > 0 entre el umbral y la capacidad de carga."""
    bajo_umbral = np.linspace(0.0, alpha, 50)[1:-1]
    sobre_umbral = np.linspace(alpha, 1.0, 50)[1:-1]
    sobre_capacidad = np.linspace(1.0, 1.5, 20)[1:]

    assert np.all(nl.f(bajo_umbral, alpha) < 0.0)
    assert np.all(nl.f(sobre_umbral, alpha) > 0.0)
    assert np.all(nl.f(sobre_capacidad, alpha) < 0.0)


def test_f_coincide_con_la_forma_expandida(alpha):
    """f(u) = -u^3 + (1 + alpha) u^2 - alpha u."""
    u = np.linspace(-0.5, 1.5, 101)
    esperado = -(u**3) + (1.0 + alpha) * u**2 - alpha * u
    np.testing.assert_allclose(nl.f(u, alpha), esperado, rtol=1e-12, atol=1e-14)


def test_f_preserva_la_forma_del_array(alpha):
    """f acepta arrays y devuelve algo de la misma forma; y acepta escalares."""
    u = np.linspace(0.0, 1.0, 7).reshape(7, 1)
    assert np.shape(nl.f(u, alpha)) == (7, 1)
    assert np.isscalar(nl.f(0.5, alpha)) or np.ndim(nl.f(0.5, alpha)) == 0


def test_f_alcanza_su_maximo_donde_corresponde(alpha):
    """El maximo de f en [alpha, 1] es interior y positivo."""
    u = np.linspace(alpha, 1.0, 2001)
    valores = nl.f(u, alpha)
    i = int(np.argmax(valores))
    assert 0 < i < len(u) - 1
    assert valores[i] > 0.0


# --------------------------------------------------------------------------
# La derivada f'
# --------------------------------------------------------------------------


def test_f_prime_coincide_con_la_forma_expandida(alpha):
    """f'(u) = -3 u^2 + 2 (1 + alpha) u - alpha."""
    u = np.linspace(-0.5, 1.5, 101)
    esperado = -3.0 * u**2 + 2.0 * (1.0 + alpha) * u - alpha
    np.testing.assert_allclose(nl.f_prime(u, alpha), esperado, rtol=1e-12, atol=1e-14)


def test_f_prime_es_la_derivada_numerica_de_f(alpha):
    """Control cruzado: f' coincide con diferencias centradas de f."""
    u = np.linspace(-0.3, 1.3, 81)
    h = 1e-6
    numerica = (nl.f(u + h, alpha) - nl.f(u - h, alpha)) / (2.0 * h)
    np.testing.assert_allclose(nl.f_prime(u, alpha), numerica, rtol=1e-6, atol=1e-8)


def test_f_prime_en_los_extremos(alpha):
    """f'(0) = -alpha < 0 (de ahi el indice +1 de la trivial) y f'(1) = alpha - 1 < 0."""
    assert nl.f_prime(0.0, alpha) == pytest.approx(-alpha, abs=1e-14)
    assert nl.f_prime(1.0, alpha) == pytest.approx(alpha - 1.0, abs=1e-14)
    assert nl.f_prime(0.0, alpha) < 0.0
    assert nl.f_prime(1.0, alpha) < 0.0


# --------------------------------------------------------------------------
# La primitiva F
# --------------------------------------------------------------------------


def test_F_se_anula_en_cero(alpha):
    """F esta normalizada con F(0) = 0."""
    assert nl.F(0.0, alpha) == pytest.approx(0.0, abs=1e-14)


def test_F_coincide_con_la_forma_expandida(alpha):
    """F(u) = -u^4/4 + (1 + alpha) u^3 / 3 - alpha u^2 / 2."""
    u = np.linspace(-0.5, 1.5, 101)
    esperado = -(u**4) / 4.0 + (1.0 + alpha) * u**3 / 3.0 - alpha * u**2 / 2.0
    np.testing.assert_allclose(nl.F(u, alpha), esperado, rtol=1e-12, atol=1e-14)


def test_F_es_primitiva_de_f(alpha):
    """Control cruzado: F' = f por diferencias centradas."""
    u = np.linspace(-0.3, 1.3, 81)
    h = 1e-6
    numerica = (nl.F(u + h, alpha) - nl.F(u - h, alpha)) / (2.0 * h)
    np.testing.assert_allclose(nl.f(u, alpha), numerica, rtol=1e-6, atol=1e-8)


def test_F_en_uno_vale_un_doceavo_de_uno_menos_dos_alpha(alpha):
    """La identidad clave: F(1) = (1 - 2 alpha) / 12.

    Su positividad es exactamente la condicion alpha < 1/2.
    """
    assert nl.F(1.0, alpha) == pytest.approx((1.0 - 2.0 * alpha) / 12.0, rel=1e-12, abs=1e-14)
    assert nl.F(1.0, alpha) > 0.0


def test_F_decrece_bajo_el_umbral_y_crece_por_encima(alpha):
    """F es decreciente en (0, alpha) y creciente en (alpha, 1), porque F' = f."""
    bajo = np.linspace(0.0, alpha, 60)
    sobre = np.linspace(alpha, 1.0, 60)
    assert np.all(np.diff(nl.F(bajo, alpha)) < 0.0)
    assert np.all(np.diff(nl.F(sobre, alpha)) > 0.0)


def test_F_tiene_un_unico_cero_no_trivial_en_el_intervalo(alpha):
    """Hay exactamente un theta en (alpha, 1) con F(theta) = 0, y F < 0 antes."""
    u = np.linspace(1e-9, 1.0, 4001)
    valores = nl.F(u, alpha)
    cambios = np.flatnonzero(np.sign(valores[:-1]) * np.sign(valores[1:]) < 0)
    assert len(cambios) == 1
    theta_aprox = u[cambios[0]]
    assert alpha < theta_aprox < 1.0


# --------------------------------------------------------------------------
# Las constantes M(alpha) y la cota inferior de lambda
# --------------------------------------------------------------------------


def test_M_constant_formula(alpha):
    """M(alpha) = (1 - alpha)^2 / 4."""
    assert nl.M_constant(alpha) == pytest.approx((1.0 - alpha) ** 2 / 4.0, rel=1e-12)


def test_M_constant_es_el_maximo_de_f_sobre_u(alpha):
    """M(alpha) = max_{0 < u <= 1} f(u)/u, alcanzado en u = (1 + alpha)/2."""
    u = np.linspace(1e-8, 1.0, 200001)
    maximo_numerico = float(np.max(nl.f(u, alpha) / u))
    assert nl.M_constant(alpha) == pytest.approx(maximo_numerico, rel=1e-6)

    argmax = (1.0 + alpha) / 2.0
    assert nl.f(argmax, alpha) / argmax == pytest.approx(nl.M_constant(alpha), rel=1e-12)


def test_M_constant_domina_linealmente_a_f(alpha):
    """f(u) <= M(alpha) u en [0, 1]: la desigualdad que se usa contra sin(pi t)."""
    u = np.linspace(0.0, 1.0, 5001)
    assert np.all(nl.f(u, alpha) <= nl.M_constant(alpha) * u + 1e-12)


@pytest.mark.parametrize("a", ALPHAS_INVALIDOS)
def test_M_constant_valida_alpha(a):
    """M_constant valida su argumento antes de calcular."""
    with pytest.raises(ValueError):
        nl.M_constant(a)


def test_lambda_lower_bound_formula(alpha):
    """La cota es 4 pi^2 / (1 - alpha)^2."""
    esperado = 4.0 * np.pi**2 / (1.0 - alpha) ** 2
    assert nl.lambda_lower_bound(alpha) == pytest.approx(esperado, rel=1e-12)


def test_lambda_lower_bound_es_pi_cuadrado_sobre_M(alpha):
    """La cota es el primer autovalor de Dirichlet dividido por M(alpha)."""
    assert nl.lambda_lower_bound(alpha) == pytest.approx(
        np.pi**2 / nl.M_constant(alpha), rel=1e-12
    )


def test_lambda_lower_bound_crece_con_alpha():
    """Cuanto mas alto el umbral de Allee, mas grande debe ser el habitat."""
    valores = [nl.lambda_lower_bound(a) for a in ALPHAS_VALIDOS]
    assert all(x < y for x, y in zip(valores, valores[1:]))


@pytest.mark.parametrize("a", ALPHAS_INVALIDOS)
def test_lambda_lower_bound_valida_alpha(a):
    """lambda_lower_bound valida su argumento antes de calcular."""
    with pytest.raises(ValueError):
        nl.lambda_lower_bound(a)
