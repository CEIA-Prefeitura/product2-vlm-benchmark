import os
import csv
import shutil
from pathlib import Path

# Configuração

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATASET_CSV_PATH = PROJECT_ROOT / "data" / "CSVs" / "300imoveis.csv" # Seu CSV com as classificações
IMAGES_CSV_PATH = PROJECT_ROOT / "data" / "imagens" / "images" # Novo nome de arquivo

def limpar_diretorios_invalidos(caminho_csv, diretorio_pai):
    """
    Extrai os 'id_anuncio' de um arquivo CSV e remove os diretórios
    em um diretório pai que não correspondem a nenhum id extraído.

    Args:
        caminho_csv (str): O caminho para o arquivo .csv.
        diretorio_pai (str): O caminho para o diretório principal (ex: 'images').
    """
    # Passo 1: Extrair todos os 'id_anuncio' do arquivo .csv
    ids_validos = set()
    try:
        with open(caminho_csv, mode='r', encoding='utf-8') as arquivo_csv:
            leitor_csv = csv.DictReader(arquivo_csv)
            for linha in leitor_csv:
                if 'id_anuncio' in linha and linha['id_anuncio']:
                    ids_validos.add(linha['id_anuncio'])
        print(f"Total de {len(ids_validos)} IDs de anúncio válidos extraídos do CSV.")
    except FileNotFoundError:
        print(f"Erro: O arquivo CSV '{caminho_csv}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo CSV: {e}")
        return

    # Passo 2: Verificar os diretórios e excluir os inválidos
    if not os.path.isdir(diretorio_pai):
        print(f"Erro: O diretório pai '{diretorio_pai}' não foi encontrado.")
        return

    total_excluidos = 0
    qtd_dir_iterados = 0
    # Itera sobre os subdiretórios principais (APARTMENT, HOME, etc.)
    for nome_subdiretorio in os.listdir(diretorio_pai):
        caminho_subdiretorio = os.path.join(diretorio_pai, nome_subdiretorio)

        if os.path.isdir(caminho_subdiretorio):
            # Itera sobre as pastas com nome de 'id_anuncio'
            for nome_anuncio_dir in os.listdir(caminho_subdiretorio):
                caminho_anuncio_dir = os.path.join(caminho_subdiretorio, nome_anuncio_dir)

                if os.path.isdir(caminho_anuncio_dir):

                    qtd_dir_iterados += 1
                    
                    # Passo 3: Compara o nome do diretório com a lista de IDs válidos
                    if nome_anuncio_dir not in ids_validos:
                        try:
                            # Passo 4: Exclui o diretório se não for válido
                            shutil.rmtree(caminho_anuncio_dir)
                            print(f"Diretório inválido excluído: {caminho_anuncio_dir}")
                            total_excluidos += 1
                        except OSError as e:
                            print(f"Erro ao excluir o diretório {caminho_anuncio_dir}: {e}")

    if total_excluidos == 0:
        print("\nNenhum diretório inválido foi encontrado.")
        print(f"\nOperação concluída. Total de {qtd_dir_iterados} diretórios analisados.")
    else:
        print(f"\nOperação concluída. Total de {total_excluidos} diretórios inválidos foram excluídos.")
        print(f"\nOperação concluída. Total de {qtd_dir_iterados} diretórios analisados.")

# --- Como usar o script ---

# 1. Altere 'DATASET_CSV_PATH' para o caminho do .csv com os dados anotados

# 2. Altere 'IMAGES_CSV_PATH' para o caminho do diretório principal com as imagens dos dados anotados

# 3. Execute a função.
limpar_diretorios_invalidos(DATASET_CSV_PATH, IMAGES_CSV_PATH)