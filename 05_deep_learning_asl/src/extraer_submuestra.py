"""
Extrae una submuestra estratificada del dataset ASL Alphabet (Kaggle) directamente
desde el .zip descargado, sin descomprimir las ~87,000 imágenes completas.

Motivo de la submuestra (ver informe, sección de preprocesamiento):
 - El dataset trae 3,000 imágenes por clase x 29 clases = 87,000 imágenes de 200x200 a color.
 - Las imágenes de cada clase son frames consecutivos de un mismo video de una persona
   haciendo la seña (ver EDA), por lo que hay altísima redundancia entre frames vecinos.
 - Se usa muestreo aleatorio (semilla fija) en vez de tomar los primeros N frames, para no
   quedarnos con una sola franja temporal del video (mismo ángulo/mismo instante) y capturar
   algo más de variabilidad de pose dentro de cada clase.
"""
import random
import zipfile
from pathlib import Path

ZIP_PATH = Path(r"D:\Downloads\archive (1).zip")
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "asl_alphabet"
TRAIN_PREFIX = "asl_alphabet_train/asl_alphabet_train/"
TEST_PREFIX = "asl_alphabet_test/asl_alphabet_test/"

N_POR_CLASE = 600  # dentro del rango 500-800 sugerido en el enunciado
SEED = 42

CLASES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["space", "del", "nothing"]


def main():
    random.seed(SEED)
    out_train = OUT_DIR / "train"
    out_test = OUT_DIR / "test"
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        nombres = zf.namelist()

        # --- submuestra de train, estratificada por clase ---
        por_clase = {c: [] for c in CLASES}
        for n in nombres:
            if not n.startswith(TRAIN_PREFIX) or n.endswith("/"):
                continue
            resto = n[len(TRAIN_PREFIX):]
            clase = resto.split("/")[0]
            if clase in por_clase:
                por_clase[clase].append(n)

        total_extraidas = 0
        for clase, archivos in por_clase.items():
            archivos.sort()
            k = min(N_POR_CLASE, len(archivos))
            elegidos = random.sample(archivos, k)
            dest_dir = out_train / clase
            dest_dir.mkdir(parents=True, exist_ok=True)
            for nombre_zip in elegidos:
                fname = Path(nombre_zip).name
                dest = dest_dir / fname
                if dest.exists():
                    continue
                with zf.open(nombre_zip) as src, open(dest, "wb") as f:
                    f.write(src.read())
                total_extraidas += 1
            print(f"{clase:>8}: {k} imágenes -> {dest_dir}")

        print(f"\nTotal train extraídas: {total_extraidas}")

        # --- set de prueba oficial de kaggle (muy pocas imágenes, se extrae completo) ---
        test_files = [n for n in nombres if n.startswith(TEST_PREFIX) and not n.endswith("/")]
        for nombre_zip in test_files:
            fname = Path(nombre_zip).name
            dest = out_test / fname
            with zf.open(nombre_zip) as src, open(dest, "wb") as f:
                f.write(src.read())
        print(f"Total test (oficial kaggle) extraídas: {len(test_files)}")


if __name__ == "__main__":
    main()
