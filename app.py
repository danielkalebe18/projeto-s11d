from pathlib import Path
from flask import Flask, render_template, jsonify, send_file
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "resultados"

CLASSES = {
    1: ("Cava", "#d73027"),
    2: ("Pilha de estéril", "#a6611a"),
    3: ("Barragem de rejeito", "#fdae61"),
    4: ("Água", "#2166ac"),
    5: ("Vegetação", "#1a9850"),
    6: ("Estruturas", "#984ea3")
}

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/api/info")
def info():
    with rasterio.open(DATA / "red_S11D.tif") as src:
        limites = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds
        )

    return jsonify({
        "nome": "Monitoramento Geoespacial S11D",
        "sensor": "Sentinel-2 L2A",
        "ano": 2024,
        "metodo": "Random Forest",
        "limites": [
            [limites[1], limites[0]],
            [limites[3], limites[2]]
        ],
        "classes": [
            nome for nome, cor in CLASSES.values()
        ]
    })

@app.route("/api/satelite")
def satelite():
    bandas = []

    for nome in ["red", "green", "blue"]:
        with rasterio.open(DATA / f"{nome}_S11D.tif") as src:
            bandas.append(src.read(1).astype(np.float32))

    canais = []

    for banda in bandas:
        minimo, maximo = np.percentile(banda, [2, 98])
        canal = np.clip(
            (banda - minimo) / (maximo - minimo + 1e-9),
            0, 1
        )
        canais.append((canal * 255).astype(np.uint8))

    imagem = np.dstack(canais)

    saida = io.BytesIO()
    Image.fromarray(imagem).save(saida, format="PNG")
    saida.seek(0)

    return send_file(saida, mimetype="image/png")

@app.route("/api/classificacao")
def classificacao():
    arquivo = OUT / "mapa_classificado_s11d.tif"

    if not arquivo.exists():
        return jsonify({
            "erro": "Mapa classificado ainda não foi gerado."
        }), 404

    with rasterio.open(arquivo) as src:
        dados = src.read(1)
        imagem = np.zeros(
            (src.height, src.width, 4),
            dtype=np.uint8
        )

        for codigo, (nome, cor) in CLASSES.items():
            rgb = tuple(
                int(cor[i:i+2], 16)
                for i in (1, 3, 5)
            )
            mascara = dados == codigo
            imagem[mascara, :3] = rgb
            imagem[mascara, 3] = 190

    saida = io.BytesIO()
    Image.fromarray(imagem, "RGBA").save(saida, format="PNG")
    saida.seek(0)

    return send_file(saida, mimetype="image/png")

@app.route("/api/estatisticas")
def estatisticas():
    arquivo = OUT / "mapa_classificado_s11d.tif"

    if not arquivo.exists():
        return jsonify({"disponivel": False})

    with rasterio.open(arquivo) as src:
        dados = src.read(1)

    total = np.count_nonzero(dados)

    resultado = {}

    for codigo, (nome, cor) in CLASSES.items():
        quantidade = int(np.count_nonzero(dados == codigo))

        resultado[nome] = {
            "pixels": quantidade,
            "percentual": round(
                quantidade / total * 100, 2
            ) if total else 0
        }

    return jsonify({
        "disponivel": True,
        "total_pixels": int(total),
        "classes": resultado
    })

if __name__ == "__main__":
    print("SISTEMA S11D INICIADO")
    print("http://127.0.0.1:5000")
    app.run(debug=True)