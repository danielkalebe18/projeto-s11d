from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(exist_ok=True)

def carregar(nome):
    with rasterio.open(DATA / nome) as img:
        return img.read(1).astype("float32")

def realcar(imagem):
    validos = imagem[np.isfinite(imagem) & (imagem > 0)]
    minimo, maximo = np.percentile(validos, [2, 98])
    return np.clip((imagem - minimo) / (maximo - minimo), 0, 1)

print("Carregando bandas Sentinel-2...")

azul = carregar("blue_S11D.tif")
verde = carregar("green_S11D.tif")
vermelho = carregar("red_S11D.tif")
nir = carregar("nir_S11D.tif")

print("Gerando composição colorida natural...")

natural = np.dstack([
    realcar(vermelho),
    realcar(verde),
    realcar(azul)
])

plt.figure(figsize=(10, 10))
plt.imshow(natural)
plt.title("S11D - Composição Natural Sentinel-2")
plt.axis("off")
plt.savefig(
    RESULTADOS / "s11d_composicao_natural.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Gerando composição infravermelha...")

infravermelho = np.dstack([
    realcar(nir),
    realcar(vermelho),
    realcar(verde)
])

plt.figure(figsize=(10, 10))
plt.imshow(infravermelho)
plt.title("S11D - Composição Infravermelha Sentinel-2")
plt.axis("off")
plt.savefig(
    RESULTADOS / "s11d_composicao_infravermelha.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("\nIMAGENS GERADAS COM SUCESSO!")
print("Confira a pasta resultados.")