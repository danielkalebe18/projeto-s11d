from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "resultados"
OUT.mkdir(exist_ok=True)

print("Carregando bandas...")
nomes = ["blue", "green", "red", "nir"]
imagens = []
perfil = None

for nome in nomes:
    with rasterio.open(DATA / f"{nome}_S11D.tif") as src:
        imagens.append(src.read(1).astype("float32"))
        perfil = src.profile.copy()

blue, green, red, nir = imagens
stack = np.stack(imagens, axis=-1)
altura, largura, _ = stack.shape

def esticar(banda):
    validos = banda[np.isfinite(banda)]
    minimo, maximo = np.percentile(validos, [2, 98])
    return np.clip((banda - minimo) / (maximo - minimo + 1e-9), 0, 1)

rgb = np.dstack([esticar(red), esticar(green), esticar(blue)])

classes = [
    "Cava",
    "Pilha de estéril",
    "Barragem de rejeito",
    "Água",
    "Vegetação",
    "Estruturas"
]

X = []
y = []

print("\nSELEÇÃO DAS AMOSTRAS")
print("Uma janela com a imagem será aberta para cada classe.")
print("Clique em 15 a 20 pontos bem identificados.")
print("Depois pressione ENTER para passar à próxima classe.")

for codigo, nome in enumerate(classes, start=1):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title(
        f"{nome}: clique em 15 a 20 pontos e pressione ENTER"
    )
    ax.axis("off")
    plt.show(block=False)

    pontos = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    quantidade = 0

    for x, y_pixel in pontos:
        coluna = int(round(x))
        linha = int(round(y_pixel))

        if 0 <= linha < altura and 0 <= coluna < largura:
            valores = stack[linha, coluna, :]

            if np.all(np.isfinite(valores)):
                X.append(valores)
                y.append(codigo)
                quantidade += 1

    print(f"{nome}: {quantidade} amostras")

if len(set(y)) < 6:
    raise SystemExit(
        "ERRO: selecione pontos para as seis classes e tente novamente."
    )

X = np.asarray(X)
y = np.asarray(y)

if min(np.bincount(y)[1:]) < 5:
    raise SystemExit(
        "ERRO: cada classe precisa de pelo menos 5 amostras."
    )

print("\nTreinando Random Forest...")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

modelo = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

modelo.fit(X_treino, y_treino)

previsao = modelo.predict(X_teste)

matriz = confusion_matrix(
    y_teste, previsao, labels=[1, 2, 3, 4, 5, 6]
)

kappa = cohen_kappa_score(y_teste, previsao)

relatorio = classification_report(
    y_teste,
    previsao,
    labels=[1, 2, 3, 4, 5, 6],
    target_names=classes,
    zero_division=0
)

print("\nGerando mapa classificado...")

pixels = stack.reshape(-1, 4)
validos = np.all(np.isfinite(pixels), axis=1)

resultado = np.zeros(len(pixels), dtype="uint8")
resultado[validos] = modelo.predict(pixels[validos])
resultado = resultado.reshape(altura, largura)

perfil.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=0,
    compress="lzw"
)

with rasterio.open(OUT / "mapa_classificado_s11d.tif", "w", **perfil) as dst:
    dst.write(resultado, 1)

cores = [
    "#d73027", "#a6611a", "#fdae61",
    "#2166ac", "#1a9850", "#984ea3"
]

plt.figure(figsize=(11, 9))
plt.imshow(resultado, cmap=ListedColormap(cores), vmin=1, vmax=6)
plt.title("Classificação supervisionada - Complexo S11D")
plt.axis("off")

legenda = [
    plt.Line2D([0], [0], marker="s", color="w",
               markerfacecolor=cores[i], label=classes[i],
               markersize=10)
    for i in range(6)
]
plt.legend(handles=legenda, loc="lower right")
plt.savefig(OUT / "mapa_classificado_s11d.png", dpi=200,
            bbox_inches="tight")
plt.close()

np.savetxt(
    OUT / "matriz_confusao.csv",
    matriz,
    delimiter=",",
    fmt="%d"
)

with open(OUT / "relatorio_acuracia.txt", "w", encoding="utf-8") as f:
    f.write("CLASSIFICAÇÃO SUPERVISIONADA - S11D\n\n")
    f.write(relatorio)
    f.write(f"\nKappa: {kappa:.4f}\n")
    f.write(
        "\nAVISO: avaliação feita com divisão aleatória das amostras "
        "selecionadas. Não representa validação independente de campo.\n"
    )

print("\nPROCESSAMENTO CONCLUÍDO!")
print("Mapa: resultados/mapa_classificado_s11d.png")
print("GeoTIFF: resultados/mapa_classificado_s11d.tif")
print("Matriz: resultados/matriz_confusao.csv")
print("Relatório: resultados/relatorio_acuracia.txt")
print(f"Kappa preliminar: {kappa:.4f}")