"""Configuracion comun de los tests.

Agrega la raiz del repositorio al sys.path para que `from src import ...`
funcione sin necesidad de instalar el paquete, y define los valores de alpha
sobre los que se parametriza toda la bateria.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Valores de alpha admisibles (0 < alpha < 1/2) que barren el rango util:
# umbral de Allee muy bajo, moderado y cercano al limite superior.
ALPHAS_VALIDOS = [0.05, 0.15, 0.25, 0.35, 0.45]

# Valores inadmisibles: alpha <= 0 (sin efecto Allee) y alpha >= 1/2 (donde
# F <= 0 en (0,1] y no puede haber solucion positiva para ningun lambda).
ALPHAS_INVALIDOS = [-0.5, -1e-12, 0.0, 0.5, 0.6, 1.0, 2.0]


@pytest.fixture(params=ALPHAS_VALIDOS)
def alpha(request) -> float:
    """Un alpha admisible; los tests que la usan corren una vez por valor."""
    return request.param
