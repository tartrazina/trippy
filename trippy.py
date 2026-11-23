"""Efectos de distorsión sobre imágenes cuadradas, exportados como GIF.

Todas las imágenes se cuantizan a una paleta fija y los efectos trabajan
sobre los índices de la paleta, así que los colores nunca se mezclan.

Uso desde línea de comandos, por ejemplo:
    python trippy.py cycle imagen.png --colors 16
    python trippy.py hosc imagen.png --interval 0.012 --distortion 10
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def _load_quantized(filename, colors):
    """Abre una imagen cuadrada y la reduce a `colors` colores (modo 'P')."""
    with Image.open(filename) as src:
        if src.width != src.height:
            raise ValueError(
                f"{filename} debe ser cuadrada, pero mide {src.width}x{src.height}"
            )
        rgb = src.convert("RGB")
    # Sin dithering: mantiene bloques de color limpios, como en pixel art.
    return rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.NONE)


def _output_path(filename, effect, out_dir=None):
    src = Path(filename)
    folder = Path(out_dir) if out_dir else src.parent
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{src.stem}-{effect}.gif"


def _roll_rows(arr, shifts):
    """Rota cada fila de `arr` hacia la derecha según `shifts[fila]`."""
    n, length = arr.shape
    cols = (np.arange(length)[None, :] - shifts[:, None]) % length
    return arr[np.arange(n)[:, None], cols]


def _write_gif(index_frames, palette_img, path, duration):
    """Guarda una secuencia de matrices de índices usando la paleta de `palette_img`."""
    palette = palette_img.getpalette()
    frames = []
    for idx in index_frames:
        frame = palette_img.copy()
        frame.frombytes(np.ascontiguousarray(idx, dtype=np.uint8).tobytes())
        frame.putpalette(palette)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    return path


def paletteCycling(filename, colors=16, out_dir=None, duration=100):
    """Rota las entradas de la paleta. Un ciclo completo son `colors` frames."""
    base = _load_quantized(filename, colors)
    palette = np.array(base.getpalette()[:3 * colors], dtype=np.uint8).reshape(colors, 3)
    frames = []
    for k in range(colors):
        frame = base.copy()
        frame.putpalette(np.roll(palette, -k, axis=0).flatten().tolist())
        frames.append(frame)
    path = _output_path(filename, f"cycle{colors}", out_dir)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    return path


def _oscillate(filename, horizontal, interval, distortion, iterations,
               out_dir, duration):
    base = _load_quantized(filename, 256)
    arr = np.array(base)
    if not horizontal:
        arr = arr.T  # las columnas pasan a ser "filas" para reutilizar _roll_rows
    n = arr.shape[0]
    state = arr
    index_frames = []
    for i in range(iterations):
        # Fase continua por línea y por frame, igual que el contador sin_count original.
        phase = (i * n + np.arange(n)) * interval
        shifts = np.floor(distortion * np.sin(phase)).astype(int)
        state = _roll_rows(state, shifts)  # cada frame parte del anterior
        index_frames.append(state if horizontal else state.T)
    effect = "hosc" if horizontal else "vosc"
    path = _output_path(filename, effect, out_dir)
    return _write_gif(index_frames, base, path, duration)


def horizontalOscillation(filename, interval=0.012, distortion=10, iterations=256,
                          out_dir=None, duration=100):
    """Desplaza cada fila con un seno: movimiento horizontal."""
    return _oscillate(filename, True, interval, distortion, iterations,
                      out_dir, duration)


def verticalOscillation(filename, interval=0.012, distortion=10, iterations=256,
                        out_dir=None, duration=100):
    """Desplaza cada columna con un seno: movimiento vertical."""
    return _oscillate(filename, False, interval, distortion, iterations,
                      out_dir, duration)


def _randomize(filename, horizontal, iterations, seed, out_dir, duration):
    base = _load_quantized(filename, 256)
    arr = np.array(base)
    if not horizontal:
        arr = arr.T
    n, length = arr.shape
    rng = random.Random(seed)  # generador propio: no toca el estado global de random
    state = arr
    index_frames = []
    for _ in range(iterations):
        shifts = np.array([rng.randrange(length) for _ in range(n)])
        state = _roll_rows(state, shifts)
        index_frames.append(state if horizontal else state.T)
    effect = "rand-h" if horizontal else "rand-v"
    path = _output_path(filename, effect, out_dir)
    return _write_gif(index_frames, base, path, duration)


def randomizeHorizontal(filename, iterations=256, seed=None, out_dir=None, duration=100):
    """Desplaza cada fila un valor aleatorio por frame."""
    return _randomize(filename, True, iterations, seed, out_dir, duration)


def randomizeVertical(filename, iterations=256, seed=None, out_dir=None, duration=100):
    """Desplaza cada columna un valor aleatorio por frame."""
    return _randomize(filename, False, iterations, seed, out_dir, duration)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Genera GIFs con efectos sobre imágenes cuadradas.")
    parser.add_argument("image", help="imagen cuadrada de entrada")
    parser.add_argument("--out-dir", help="carpeta de salida (por defecto, la de la imagen)")
    parser.add_argument("--duration", type=int, default=100, help="ms por frame")
    sub = parser.add_subparsers(dest="effect", required=True)

    cycle = sub.add_parser("cycle", help="rotación de paleta")
    cycle.add_argument("--colors", type=int, default=16, help="colores de la paleta (1-256)")

    for name, helptext in (("hosc", "oscilación horizontal"), ("vosc", "oscilación vertical")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--interval", type=float, default=0.012)
        p.add_argument("--distortion", type=float, default=10)
        p.add_argument("--iterations", type=int, default=256)

    for name, helptext in (("rand-h", "aleatorio horizontal"), ("rand-v", "aleatorio vertical")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--iterations", type=int, default=256)
        p.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(argv)
    common = dict(out_dir=args.out_dir, duration=args.duration)
    if args.effect == "cycle":
        path = paletteCycling(args.image, colors=args.colors, **common)
    elif args.effect == "hosc":
        path = horizontalOscillation(args.image, args.interval, args.distortion,
                                     args.iterations, **common)
    elif args.effect == "vosc":
        path = verticalOscillation(args.image, args.interval, args.distortion,
                                   args.iterations, **common)
    elif args.effect == "rand-h":
        path = randomizeHorizontal(args.image, args.iterations, args.seed, **common)
    else:
        path = randomizeVertical(args.image, args.iterations, args.seed, **common)
    print(path)
    return path


if __name__ == "__main__":
    main()
