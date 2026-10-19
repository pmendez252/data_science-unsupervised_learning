"""Versiones Python de las funciones MATLAB del laboratorio.

Requiere NumPy y, para visualizar imágenes, Matplotlib.
Las dimensiones ``dim`` se numeran desde cero, como en NumPy.
"""

import numpy as np


def nanmean(x, dim=None):
    """Media ignorando NaN; por defecto usa la primera dimensión no unitaria.

    Conserva la dimensión reducida, igual que MATLAB. Una entrada vacía o
    un grupo sin valores válidos produce NaN.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if x.ndim == 0:
        return x.copy()
    if dim is None:
        dim = next((i for i, size in enumerate(x.shape) if size != 1), 0)
    valid = ~np.isnan(x)
    count = np.sum(valid, axis=dim, keepdims=True)
    total = np.sum(np.where(valid, x, 0), axis=dim, keepdims=True)
    result = np.full_like(total, np.nan, dtype=float)
    np.divide(total, count, out=result, where=count != 0)
    return result


def handle_missing(data):
    """Devuelve una copia con los NaN sustituidos por la media de su columna.

    Las columnas completamente vacías permanecen como NaN.
    """
    data = np.array(data, dtype=float, copy=True)
    if data.ndim != 2:
        raise ValueError("data debe ser una matriz bidimensional.")
    if data.size == 0:
        return data
    return np.where(np.isnan(data), nanmean(data, dim=0), data)


def kMeansInitCentroids(X, K, random_state=None):
    """Selecciona K filas al azar, sin reemplazo, como centroides iniciales.

    Usa random_state para reproducir una selección.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz bidimensional.")
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)):
        raise ValueError("K debe ser un entero.")
    if not 1 <= K <= X.shape[0]:
        raise ValueError("K debe estar entre 1 y el número de filas de X.")
    rng = np.random.default_rng(random_state)
    return X[rng.permutation(X.shape[0])[:K]].copy()


def displayData(X, example_width=None):
    """Muestra las imágenes de X en una cuadrícula y devuelve (h, matriz).

    Cada fila es una imagen aplanada en orden de columnas, como en MATLAB.
    ``h`` es el objeto AxesImage de Matplotlib. En scripts, utiliza
    ``matplotlib.pyplot.show()`` para mantener la ventana abierta.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X debe ser una matriz bidimensional no vacía.")
    m, n = X.shape
    if example_width is None:
        example_width = int(np.floor(np.sqrt(n) + 0.5))
    if (
        isinstance(example_width, (bool, np.bool_))
        or not isinstance(example_width, (int, np.integer))
        or example_width <= 0
        or n % example_width != 0
    ):
        raise ValueError("example_width debe ser un entero positivo que divida las columnas de X.")
    example_height = n // example_width
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    pad = 1
    display_array = -np.ones(
        (pad + display_rows * (example_height + pad),
         pad + display_cols * (example_width + pad))
    )
    for index, example in enumerate(X):
        row, col = divmod(index, display_cols)
        top = pad + row * (example_height + pad)
        left = pad + col * (example_width + pad)
        max_val = np.max(np.abs(example))
        patch = example.reshape((example_height, example_width), order="F")
        if max_val != 0:
            patch = patch / max_val
        display_array[top:top + example_height, left:left + example_width] = patch
    _, ax = plt.subplots()
    h = ax.imshow(display_array, cmap="gray", vmin=-1, vmax=1,
                  aspect="equal", interpolation="nearest")
    ax.axis("off")
    ax.figure.canvas.draw_idle()
    return h, display_array


# El archivo MATLAB se llama displayImageData.m, pero declara displayData.
displayImageData = displayData
