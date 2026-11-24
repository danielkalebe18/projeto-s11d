from pathlib import Path
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds

BASE = Path(__file__).resolve().parent.parent
PASTA = BASE / "data"
PASTA.mkdir(exist_ok=True)

# Área aproximada do Complexo S11D
BBOX = [-50.08, -6.46, -49.97, -6.37]

print("=" * 55)
print("PROJETO S11D - DOWNLOAD SENTINEL-2 L2A")
print("=" * 55)

try:
    catalogo = Client.open(
        "https://earth-search.aws.element84.com/v1"
    )

    print("\nProcurando imagens de 2024...")

    busca = catalogo.search(
        collections=["sentinel-2-c1-l2a"],
        bbox=BBOX,
        datetime="2024-01-01/2024-12-31",
        query={"eo:cloud_cover": {"lt": 50}},
        max_items=100
    )

    cenas = list(busca.items())

    if not cenas:
        print("Nenhuma cena encontrada.")
        raise SystemExit

    cenas.sort(
        key=lambda cena: cena.properties.get(
            "eo:cloud_cover", 100
        )
    )

    cena = cenas[0]

    print("Cena:", cena.id)
    print("Data:", cena.datetime)
    print(
        "Cobertura de nuvens:",
        cena.properties.get("eo:cloud_cover"),
        "%"
    )

    bandas = ["blue", "green", "red", "nir"]

    for banda in bandas:
        print(f"\nBaixando banda {banda}...")

        url = cena.assets[banda].href

        with rasterio.open(url) as src:
            limites = transform_bounds(
                "EPSG:4326",
                src.crs,
                *BBOX
            )

            janela = from_bounds(
                *limites,
                transform=src.transform
            )

            janela = janela.round_offsets().round_lengths()
            janela = janela.intersection(
                Window(0, 0, src.width, src.height)
            )

            dados = src.read(window=janela)

            perfil = src.profile.copy()
            perfil.update(
                driver="GTiff",
                width=dados.shape[2],
                height=dados.shape[1],
                transform=src.window_transform(janela),
                compress="deflate"
            )

            destino = PASTA / f"{banda}_S11D.tif"

            with rasterio.open(destino, "w", **perfil) as saida:
                saida.write(dados)

        print("Salvo:", destino.name)

    print("\nDOWNLOAD CONCLUÍDO!")
    print("Confira os arquivos na pasta data.")

except SystemExit:
    pass
except Exception as erro:
    print("\nERRO:", erro)
    print("Me envie essa mensagem para corrigirmos.")