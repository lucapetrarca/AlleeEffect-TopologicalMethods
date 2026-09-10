"""Especificacion ejecutable de las cotas a priori y de la formulacion de punto fijo.

Cubre `src.green` (nucleo de Green, truncamiento, operador T_lambda) y la
consecuencia central del resultado 1: toda solucion cumple 0 <= u <= 1.
Incluye ademas el chequeo de por que se pide alpha < 1/2.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import green, nonlinearity as nl, shooting


# --------------------------------------------------------------------------
# Nucleo de Green
# --------------------------------------------------------------------------


def test_green_kernel_formula_por_tramos():
    """G(t,s) = s(1-t) si s <= t, y t(1-s) si t <= s."""
    t = np.array([0.2, 0.5, 0.5, 0.9])
    s = np.array([0.1, 0.5, 0.8, 0.3])
    esperado = np.where(s <= t, s * (1.0 - t), t * (1.0 - s))
    np.testing.assert_allclose(green.green_kernel(t, s), esperado, rtol=1e-12, atol=1e-14)


def test_green_kernel_forma_compacta():
    """G(t,s) = min(t,s) * (1 - max(t,s)) en toda la matriz."""
    t = np.linspace(0.0, 1.0, 41).reshape(-1, 1)
    s = np.linspace(0.0, 1.0, 37).reshape(1, -1)
    esperado = np.minimum(t, s) * (1.0 - np.maximum(t, s))
    np.testing.assert_allclose(green.green_kernel(t, s), esperado, rtol=1e-12, atol=1e-14)


def test_green_kernel_es_simetrico():
    """G(t,s) = G(s,t): el operador de Green es autoadjunto."""
    t = np.linspace(0.0, 1.0, 33).reshape(-1, 1)
    s = np.linspace(0.0, 1.0, 33).reshape(1, -1)
    matriz = np.asarray(green.green_kernel(t, s))
    np.testing.assert_allclose(matriz, matriz.T, rtol=1e-12, atol=1e-14)


def test_green_kernel_es_no_negativo():
    """G >= 0: es lo que hace que T_lambda preserve el orden."""
    t = np.linspace(0.0, 1.0, 51).reshape(-1, 1)
    s = np.linspace(0.0, 1.0, 51).reshape(1, -1)
    assert np.all(np.asarray(green.green_kernel(t, s)) >= 0.0)


def test_green_kernel_se_anula_en_el_borde():
    """G(0,s) = G(1,s) = 0, de donde T_lambda u cumple la condicion de Dirichlet."""
    s = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(green.green_kernel(0.0, s), 0.0, atol=1e-14)
    np.testing.assert_allclose(green.green_kernel(1.0, s), 0.0, atol=1e-14)


def test_green_kernel_maximo_es_un_cuarto():
    """max G = 1/4, alcanzado en t = s = 1/2."""
    t = np.linspace(0.0, 1.0, 401).reshape(-1, 1)
    s = np.linspace(0.0, 1.0, 401).reshape(1, -1)
    assert float(np.max(np.asarray(green.green_kernel(t, s)))) == pytest.approx(0.25, rel=1e-10)
    assert green.green_kernel(0.5, 0.5) == pytest.approx(0.25, rel=1e-12)


def test_integral_del_nucleo_de_green():
    """int_0^1 G(t,s) ds = t(1-t)/2, la solucion de -w'' = 1 con Dirichlet."""
    s = np.linspace(0.0, 1.0, 20001)
    for t in (0.1, 0.25, 0.5, 0.73, 0.9):
        integral = np.trapezoid(np.asarray(green.green_kernel(t, s)), s)
        assert integral == pytest.approx(t * (1.0 - t) / 2.0, rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------
# Truncamiento de la no linealidad
# --------------------------------------------------------------------------


def test_truncate_coincide_con_f_dentro_de_cero_uno(alpha):
    """f_tilde = f en [0, 1], que es donde viven las soluciones."""
    f_tilde = green.truncate(lambda u: nl.f(u, alpha))
    u = np.linspace(0.0, 1.0, 501)
    np.testing.assert_allclose(f_tilde(u), nl.f(u, alpha), rtol=1e-12, atol=1e-14)


def test_truncate_se_anula_fuera_de_cero_uno(alpha):
    """Como f(0) = f(1) = 0, la truncada es identicamente nula fuera de [0, 1].

    Ese es justamente el hecho que hace funcionar el principio del maximo:
    donde u < 0 la ecuacion truncada da -u'' = 0.
    """
    f_tilde = green.truncate(lambda u: nl.f(u, alpha))
    izquierda = np.linspace(-3.0, -1e-6, 200)
    derecha = np.linspace(1.0 + 1e-6, 4.0, 200)
    np.testing.assert_allclose(f_tilde(izquierda), 0.0, atol=1e-14)
    np.testing.assert_allclose(f_tilde(derecha), 0.0, atol=1e-14)


def test_truncate_es_acotada(alpha):
    """La truncada es acotada, a diferencia de la f cubica original."""
    f_tilde = green.truncate(lambda u: nl.f(u, alpha))
    u = np.linspace(-50.0, 50.0, 5001)
    cota = float(np.max(np.abs(nl.f(np.linspace(0.0, 1.0, 5001), alpha))))
    assert np.all(np.abs(f_tilde(u)) <= cota + 1e-12)


def test_truncate_rechaza_intervalo_degenerado(alpha):
    """lo >= hi no define ningun intervalo de truncamiento."""
    with pytest.raises(ValueError):
        green.truncate(lambda u: nl.f(u, alpha), lo=1.0, hi=0.0)


# --------------------------------------------------------------------------
# Operador de punto fijo T_lambda
# --------------------------------------------------------------------------


def test_operador_se_anula_en_el_borde(alpha):
    """T_lambda u cumple siempre la condicion de Dirichlet, sea cual sea u."""
    t = np.linspace(0.0, 1.0, 401)
    u = 0.7 * np.sin(np.pi * t)
    salida = green.fixed_point_operator(u, lam=50.0, alpha=alpha, t=t)
    assert salida[0] == pytest.approx(0.0, abs=1e-12)
    assert salida[-1] == pytest.approx(0.0, abs=1e-12)


def test_la_funcion_nula_es_punto_fijo(alpha):
    """u == 0 es solucion del problema: T_lambda 0 = 0 porque f(0) = 0."""
    t = np.linspace(0.0, 1.0, 401)
    salida = green.fixed_point_operator(np.zeros_like(t), lam=80.0, alpha=alpha, t=t)
    np.testing.assert_allclose(salida, 0.0, atol=1e-12)


def test_la_supersolucion_constante_uno_va_a_cero(alpha):
    """T_lambda 1 = 0 porque f(1) = 0; por eso psi == 1 es supersolucion."""
    t = np.linspace(0.0, 1.0, 401)
    salida = green.fixed_point_operator(np.ones_like(t), lam=80.0, alpha=alpha, t=t)
    np.testing.assert_allclose(salida, 0.0, atol=1e-12)


def test_operador_sobre_una_constante_da_la_parabola_exacta(alpha):
    """Para u == c constante, T_lambda u = lambda f(c) t(1-t)/2.

    Es consecuencia exacta de int_0^1 G(t,s) ds = t(1-t)/2, y controla a la vez
    el nucleo, la cuadratura y el factor lambda.
    """
    t = np.linspace(0.0, 1.0, 2001)
    c = (1.0 + alpha) / 2.0  # estrictamente entre alpha y 1, con f(c) > 0
    lam = 30.0
    salida = green.fixed_point_operator(np.full_like(t, c), lam=lam, alpha=alpha, t=t)
    esperado = lam * nl.f(c, alpha) * t * (1.0 - t) / 2.0
    np.testing.assert_allclose(salida, esperado, rtol=1e-4, atol=1e-7)


def test_operador_es_lineal_en_lambda(alpha):
    """T_{c lambda} u = c T_lambda u: lambda entra como factor."""
    t = np.linspace(0.0, 1.0, 401)
    u = 0.8 * np.sin(np.pi * t)
    a = green.fixed_point_operator(u, lam=10.0, alpha=alpha, t=t)
    b = green.fixed_point_operator(u, lam=30.0, alpha=alpha, t=t)
    np.testing.assert_allclose(b, 3.0 * a, rtol=1e-10, atol=1e-12)


def test_operador_es_monotono_en_el_rango_donde_f_crece(alpha):
    """Si alpha <= u <= v <= (1+alpha)/2 entonces T_lambda u <= T_lambda v.

    G >= 0 y f creciente en ese rango: es el ingrediente de la iteracion
    monotona.
    """
    t = np.linspace(0.0, 1.0, 801)
    c = (1.0 + alpha) / 2.0
    u = np.full_like(t, alpha + 0.25 * (c - alpha))
    v = np.full_like(t, alpha + 0.75 * (c - alpha))
    tu = green.fixed_point_operator(u, lam=40.0, alpha=alpha, t=t)
    tv = green.fixed_point_operator(v, lam=40.0, alpha=alpha, t=t)
    assert np.all(tu <= tv + 1e-12)


def test_operador_rechaza_grillas_incompatibles(alpha):
    """u y t tienen que tener la misma longitud."""
    with pytest.raises(ValueError):
        green.fixed_point_operator(
            np.zeros(10), lam=1.0, alpha=alpha, t=np.linspace(0.0, 1.0, 11)
        )


# --------------------------------------------------------------------------
# Cotas a priori sobre las soluciones (resultado 1)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [200.0, 400.0, 800.0])
def test_toda_solucion_esta_entre_cero_y_uno(alpha, lam):
    """Cota a priori: toda solucion del BVP cumple 0 <= u <= 1.

    Es la conclusion del principio del maximo, y lo que permite reemplazar f
    por su truncada sin cambiar el conjunto de soluciones.
    """
    soluciones = shooting.find_all_solutions(lam=lam, alpha=alpha)
    for sol in soluciones:
        assert np.min(sol.u) >= -1e-8
        assert np.max(sol.u) <= 1.0 + 1e-8


@pytest.mark.parametrize("lam", [200.0, 400.0, 800.0])
def test_las_soluciones_son_estrictamente_positivas_en_el_interior(alpha, lam):
    """Las soluciones no triviales halladas son positivas dentro de (0, 1)."""
    soluciones = shooting.find_all_solutions(lam=lam, alpha=alpha)
    for sol in soluciones:
        interior = sol.u[1:-1]
        assert np.all(interior > -1e-8)
        assert sol.rho > 0.0


@pytest.mark.parametrize("lam", [200.0, 400.0, 800.0])
def test_el_maximo_de_toda_solucion_supera_el_umbral_de_allee(alpha, lam):
    """El pico de una solucion positiva supera theta(alpha) > alpha.

    Sale de la identidad de energia: hace falta F(rho) > 0.
    """
    soluciones = shooting.find_all_solutions(lam=lam, alpha=alpha)
    for sol in soluciones:
        assert sol.rho > alpha
        assert nl.F(sol.rho, alpha) > 0.0


@pytest.mark.parametrize("lam", [200.0, 400.0])
def test_las_soluciones_son_puntos_fijos_del_operador(alpha, lam):
    """Control cruzado entre shooting y punto fijo: u = T_lambda u."""
    soluciones = shooting.find_all_solutions(lam=lam, alpha=alpha, n_points=801)
    for sol in soluciones:
        imagen = green.fixed_point_operator(sol.u, lam=lam, alpha=alpha, t=sol.t)
        assert float(np.max(np.abs(imagen - sol.u))) < 1e-3


# --------------------------------------------------------------------------
# Por que se pide alpha < 1/2
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.5, 0.55, 0.7, 0.9])
def test_sin_energia_positiva_no_hay_solucion_positiva(a):
    """Si alpha >= 1/2 entonces F <= 0 en todo (0, 1] y no hay solucion positiva.

    La identidad de energia (1/2)(u')^2 + lambda F(u) = lambda F(rho) exige
    F(rho) > F(0) = 0 para el maximo rho de una solucion positiva; si F <= 0 en
    (0, 1], junto con la cota a priori rho <= 1, eso es imposible para todo
    lambda > 0.
    """
    u = np.linspace(1e-9, 1.0, 5001)
    assert np.all(nl.F(u, a) <= 1e-12)
    assert nl.F(1.0, a) == pytest.approx((1.0 - 2.0 * a) / 12.0, abs=1e-14)
    assert nl.F(1.0, a) <= 0.0


@pytest.mark.parametrize("a", [0.5, 0.55, 0.7])
def test_el_shooting_no_encuentra_soluciones_si_alpha_es_grande(a):
    """Corolario operativo: con alpha >= 1/2 el barrido no devuelve nada.

    Se toma un lambda muy grande a proposito: la ausencia de soluciones no es
    cuestion de tamano de habitat sino de la forma de la reaccion.
    """
    assert shooting.find_all_solutions(lam=5000.0, alpha=a) == []
