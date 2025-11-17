# utils/process_results_for_scoring.py

import json
from pathlib import Path
from loguru import logger
import sys
import argparse
from tqdm import tqdm

# Adicionar a raiz do projeto ao sys.path para permitir a importação de 'utils'
try:
    from .scoring import calcular_pontuacao_bic # Import relativo se rodar como parte do pacote
except ImportError:
    # Fallback para importação absoluta
    current_script_path = Path(__file__).resolve()
    project_root_for_import = current_script_path.parent.parent
    if str(project_root_for_import) not in sys.path:
        sys.path.insert(0, str(project_root_for_import))
    from utils.scoring import calcular_pontuacao_bic

# --- Configuração ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "resultados_geracao_gemini" # Onde os resultados da geração estão
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "resultados_pontuados" # Onde salvar os novos JSONs com pontuação

LOG_FILE_PATH_SCORING = PROJECT_ROOT / "logs" / "scoring_process.log"

# --- Configurar Logging ---
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE_PATH_SCORING, rotation="2 MB", retention="3 days", level="DEBUG", encoding="utf-8")


def process_and_score_results(input_file: Path, output_file: Path):
    """
    Lê um arquivo JSON de resultados gerados, calcula a pontuação BIC para cada
    imóvel com base na resposta do modelo, e salva um novo JSON com os resultados
    e as pontuações.

    Args:
        input_file: Caminho para o arquivo 'generation_results.json'.
        output_file: Caminho para o novo arquivo JSON a ser criado com as pontuações.
    """
    logger.info(f"Iniciando processo de pontuação para o arquivo: {input_file.name}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            generation_data = json.load(f)
        logger.info(f"Arquivo de geração carregado com sucesso.")
    except FileNotFoundError:
        logger.error(f"Arquivo de entrada não encontrado: {input_file}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON do arquivo de entrada {input_file}: {e}")
        return
    except Exception as e:
        logger.error(f"Erro inesperado ao ler o arquivo {input_file}: {e}")
        return

    metadata_geracao = generation_data.get("metadata_geracao_run", {})
    resultados_gerados = generation_data.get("resultados_gerados_run", [])
    
    if not resultados_gerados:
        logger.warning(f"Nenhum resultado de imóvel ('resultados_gerados_run') encontrado em {input_file}. Nada a fazer.")
        return
        
    resultados_com_pontuacao = []

    for item in tqdm(resultados_gerados, desc=f"Calculando pontuações para {input_file.name}"):
        parsed_dict = item.get("resposta_modelo_parsed_dict")
        
        # Estrutura base para o novo item de resultado
        novo_item = {
            "property_id": item.get("property_id"),
            "dataset_idx": item.get("dataset_idx"),
            "resposta_modelo_parsed_dict": parsed_dict,
            # Inicializar pontuações como None
            "pontuacao_caracteristicas": None,
            "pontuacao_benfeitorias": None,
            "pontuacao_total": None
        }

        if parsed_dict and isinstance(parsed_dict, dict):
            try:
                # Chamar a função de pontuação usando o dicionário parseado do modelo
                pontuacoes = calcular_pontuacao_bic(parsed_dict)
                novo_item.update(pontuacoes) # Adiciona as chaves de pontuação ao novo_item
            except Exception as e:
                logger.error(f"Erro ao calcular pontuação para property_id {item.get('property_id')}: {e}")
        else:
            logger.warning(f"Nenhuma resposta parseada ('resposta_modelo_parsed_dict') encontrada para property_id {item.get('property_id')}. Pontuação não calculada.")
        
        resultados_com_pontuacao.append(novo_item)

    # Construir o objeto JSON de saída final
    output_final = {
        "metadata_geracao_run": metadata_geracao, # Manter os metadados originais
        "resultados_pontuados": resultados_com_pontuacao # Usar a nova lista com pontuações
    }

    # Salvar o novo arquivo JSON
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_final, f, indent=2, ensure_ascii=False)
        logger.success(f"Arquivo com pontuações salvo com sucesso em: {output_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar o arquivo JSON de saída {output_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processa um arquivo JSON de resultados de geração de VLM e adiciona a pontuação BIC a cada resultado."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Caminho para o arquivo JSON de entrada (ex: 'generation_results.json')."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Diretório para salvar o novo arquivo JSON com pontuações. Padrão: {DEFAULT_OUTPUT_DIR}"
    )
    args = parser.parse_args()

    if not args.input_file.is_file():
        logger.critical(f"Arquivo de entrada especificado não existe: {args.input_file}")
        sys.exit(1)

    # Criar um nome de arquivo de saída baseado no de entrada
    output_filename = f"scored_{args.input_file.name}"
    output_filepath = args.output_dir / output_filename

    process_and_score_results(args.input_file, output_filepath)