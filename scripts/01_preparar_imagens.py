from pathlib import Path
import rasterio

# Caminho principal do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

print("=" * 50)
print("PROJETO S11D - SENTINEL-2 L2A")
print("=" * 50)

# Procurar arquivos de imagem
extensoes = ["*.tif", "*.tiff"]

arquivos = []

for extensao in extensoes:
    arquivos.extend(DATA_DIR.rglob(extensao))

if not arquivos:
    print("\nNenhuma imagem TIFF encontrada.")
    print("Verifique se os dados Sentinel-2 foram baixados.")
else:
    print(f"\nImagens encontradas: {len(arquivos)}")

    for arquivo in arquivos:
        print(f"\nArquivo: {arquivo.name}")

        try:
            with rasterio.open(arquivo) as imagem:
                print("Imagem aberta com sucesso!")
                print(f"Quantidade de bandas: {imagem.count}")
                print(f"Largura: {imagem.width}")
                print(f"Altura: {imagem.height}")
                print(f"CRS: {imagem.crs}")
                print(f"Resolução: {imagem.res}")

        except Exception as erro:
            print(f"Não foi possível abrir: {erro}")

print("\nVerificação concluída.")