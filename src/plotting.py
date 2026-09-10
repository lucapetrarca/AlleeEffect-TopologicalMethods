"""Estilo comun de las figuras y guardado en `figuras/`.

Todos los notebooks arrancan llamando a `set_style()` y terminan cada figura
con `save_figure(fig, nombre)`, de modo que las salidas sean homogeneas y
reproducibles. El directorio `figuras/` esta ignorado por git: las figuras se
regeneran corriendo los notebooks, no se versionan.

Recordatorio: las figuras ILUSTRAN los teoremas, no los demuestran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.figure

# Directorio de salida, relativo a la raiz del repositorio.
FIGURES_DIR: Path = Path(__file__).resolve().parent.parent / "figuras"


def set_style(context: str = "notebook") -> None:
    """Fija el estilo de matplotlib comun a todas las figuras del trabajo.

    Debe dejar los graficos legibles tanto en pantalla como impresos en el
    informe: tamano de fuente razonable, grilla suave de fondo, lineas de
    grosor uniforme, colores que se distingan tambien en escala de grises
    (porque las tres soluciones del notebook 06 se imprimen en el mismo eje) y
    `figure.dpi` alto. Conviene tambien habilitar `text.usetex` solo si esta
    disponible, y en caso contrario usar mathtext, para que las etiquetas con
    lambda y alpha se vean bien sin romper en maquinas sin LaTeX.

    Parametros
    ----------
    context : str
        Perfil de tamanos. 'notebook' para trabajar, 'paper' para las figuras
        que van al informe (fuentes algo mas chicas, lineas mas finas).

    Devuelve
    --------
    None
        Modifica `matplotlib.rcParams` en el lugar.

    Levanta
    -------
    ValueError
        Si `context` no es uno de los perfiles previstos.
    """
    raise NotImplementedError


def save_figure(
    fig: matplotlib.figure.Figure,
    name: str,
    directory: Optional[Path] = None,
    dpi: int = 200,
    formats: tuple[str, ...] = ("png",),
) -> tuple[Path, ...]:
    """Guarda una figura en `figuras/` con nombre y formato uniformes.

    Crea el directorio si no existe, aplica `tight_layout` o `bbox_inches`
    ajustado para que no se corten las etiquetas, y guarda con el `dpi` pedido.
    El nombre no debe incluir extension: se agrega una por cada formato de
    `formats`.

    Parametros
    ----------
    fig : matplotlib.figure.Figure
        La figura a guardar.
    name : str
        Nombre base del archivo, sin extension. Convencion del repositorio:
        `NN_tema` con NN el numero del notebook que la genera, por ejemplo
        `05_diagrama_bifurcacion_S`.
    directory : Path, opcional
        Directorio de salida. Si es None se usa `FIGURES_DIR`.
    dpi : int
        Resolucion de salida.
    formats : tuple[str, ...]
        Formatos a generar, por ejemplo ('png',) o ('png', 'pdf').

    Devuelve
    --------
    tuple[Path, ...]
        Las rutas de los archivos efectivamente escritos, en el orden de
        `formats`.

    Levanta
    -------
    ValueError
        Si `name` viene con extension o esta vacio, o si `formats` es vacio.
    """
    raise NotImplementedError
