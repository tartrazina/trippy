import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

import trippy


def _make_square(path, size=32):
    """Imagen sintética con pocos colores y patrón no simétrico."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 6, size=(size, size, 3), dtype=np.uint8) * 40
    Image.fromarray(arr).save(path)
    return path


def _encode(rgb):
    """Convierte un array (H, W, 3) en enteros (H, W) para comparar píxeles."""
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _read_frames(gif_path):
    """Lee cada frame del GIF como colores reales (no índices de paleta)."""
    frames = []
    with Image.open(gif_path) as im:
        for frame in ImageSequence.Iterator(im):
            frames.append(_encode(np.array(frame.convert("RGB"))))
    return frames


def _rows(a):
    """Contenido de cada fila como multiconjunto (ignora el orden dentro de la fila)."""
    return sorted(tuple(sorted(r)) for r in a)


def _cols(a):
    """Contenido de cada columna como multiconjunto (ignora el orden dentro de la columna)."""
    return sorted(tuple(sorted(c)) for c in a.T)


class TrippyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.src = _make_square(self.dir / "img.png")
        self.base = _encode(np.array(trippy._load_quantized(self.src, 256).convert("RGB")))

    def tearDown(self):
        self._tmp.cleanup()

    def test_palette_cycling_produces_colors_frames(self):
        colors = 8
        out = trippy.paletteCycling(self.src, colors=colors, out_dir=self.dir)
        frames = _read_frames(out)
        self.assertEqual(len(frames), colors)
        # Frame k debe ser la imagen de 8 colores con la paleta rotada k posiciones.
        q = trippy._load_quantized(self.src, colors)
        idx = np.array(q)
        pal = np.array(q.getpalette()[: 3 * colors], dtype=np.uint8).reshape(colors, 3)
        for k, frame in enumerate(frames):
            expected = _encode(np.roll(pal, -k, axis=0)[idx])
            np.testing.assert_array_equal(frame, expected)

    def test_horizontal_oscillation_keeps_row_contents(self):
        out = trippy.horizontalOscillation(self.src, iterations=6, out_dir=self.dir)
        frames = _read_frames(out)
        self.assertEqual(len(frames), 6)
        for idx in frames:
            self.assertEqual(_rows(idx), _rows(self.base))

    def test_vertical_oscillation_keeps_column_contents(self):
        out = trippy.verticalOscillation(self.src, iterations=6, out_dir=self.dir)
        frames = _read_frames(out)
        self.assertEqual(len(frames), 6)
        for idx in frames:
            self.assertEqual(_cols(idx), _cols(self.base))

    def test_randomize_vertical_shifts_columns_not_rows(self):
        # Regresión: la versión original rotaba filas también en el modo vertical.
        out = trippy.randomizeVertical(self.src, iterations=5, seed=1, out_dir=self.dir)
        frames = _read_frames(out)
        for idx in frames:
            self.assertEqual(_cols(idx), _cols(self.base))
        self.assertNotEqual(_rows(frames[-1]), _rows(self.base))

    def test_randomize_horizontal_keeps_row_contents(self):
        out = trippy.randomizeHorizontal(self.src, iterations=5, seed=1, out_dir=self.dir)
        frames = _read_frames(out)
        for idx in frames:
            self.assertEqual(_rows(idx), _rows(self.base))

    def test_iterations_sets_frame_count(self):
        out = trippy.randomizeHorizontal(self.src, iterations=10, seed=3, out_dir=self.dir)
        frames = _read_frames(out)
        self.assertEqual(len(frames), 10)

    def test_same_seed_same_output_different_seed_differs(self):
        a = trippy.randomizeHorizontal(self.src, 4, seed=7, out_dir=self.dir / "a")
        b = trippy.randomizeHorizontal(self.src, 4, seed=7, out_dir=self.dir / "b")
        c = trippy.randomizeHorizontal(self.src, 4, seed=8, out_dir=self.dir / "c")
        ia = _read_frames(a)
        ib = _read_frames(b)
        ic = _read_frames(c)
        for x, y in zip(ia, ib):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(any(not np.array_equal(x, y) for x, y in zip(ia, ic)))

    def test_non_square_image_is_rejected(self):
        wide = self.dir / "wide.png"
        Image.new("RGB", (40, 20)).save(wide)
        with self.assertRaises(ValueError):
            trippy.paletteCycling(wide, out_dir=self.dir)

    def test_dotted_directory_and_repeated_calls(self):
        # Regresión: split('.') rompía rutas con puntos y os.chdir rompía la segunda llamada.
        folder = self.dir / "v1.2"
        folder.mkdir()
        src = _make_square(folder / "a.png")
        cwd_before = os.getcwd()
        first = trippy.horizontalOscillation(src, iterations=3)
        second = trippy.horizontalOscillation(src, iterations=3)
        self.assertEqual(os.getcwd(), cwd_before)
        self.assertEqual(first.parent, folder)
        self.assertEqual(second.parent, folder)
        self.assertTrue(first.exists() and second.exists())


if __name__ == "__main__":
    unittest.main()
