"""Script para GERAÇÃO de respostas JSON de modelos VLM OLLAMA - AVALIAÇÃO POR CARACTERÍSTICA INDIVIDUAL.

Este script itera sobre um dataset de imóveis e processa cada uma das 11 características
SEPARADAMENTE. Para cada característica, o script:
1. Processa TODAS as propriedades do dataset
2. Gera um JSON específico com os resultados dessa característica
3. Salva em um arquivo dedicado

Estrutura de saída: Um JSON por característica contendo resultados de todas as propriedades.
"""

import os
import json
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import sys
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm
from loguru import logger

try:
    from utils.helpers import (
        get_images_from_directory,
        _encode_image_to_base64,
    )
    from ollama_connector import ollama_generate, validate_model
except ImportError as e:
    logger.critical(f"Erro de importação: {e}. Verifique ollama_connector.py e utils/helpers.py.")
    raise SystemExit(1)

load_dotenv()

# --- Configuração ---
MODEL_NAME_TO_USE = os.getenv("OLLAMA_BENCHMARK_MODELS", "gemma3:4b").split(',')[0].strip()
TEMPERATURE_TO_USE = float(os.getenv("BENCHMARK_TEMPERATURES", "0.0"))

DATASET_PATH = Path(os.getenv("BENCHMARK_DATASET_PATH", "./data/CSVs/91_casas_terreas_anotadas.csv"))
IMAGES_BASE_DIR = Path(os.getenv("BENCHMARK_IMAGES_DIR", "./data/imagens/images"))
OUTPUT_BASE_DIR = Path(os.getenv("SINGLE_CHAR_OUTPUT_DIR", "./resultados_por_caracteristica"))
PROMPTS_BASE_DIR = Path(os.getenv("BENCHMARK_PROMPTS_DIR_CARACTERISTICAS", "./prompts/prompts_por_caracteristica"))

MODEL = "gemma" # Altere para o modelo desejado

LOG_FILE_PATH = "logs/generation_ollama_single_char.log"

"""# Definição das 11 características a serem avaliadas
CARACTERISTICAS = [
    "Estrutura",
    "Esquadrias",
    "Piso",
    "Forro",
    "Instalacao_Eletrica",
    "Instalacao_Sanitaria",
    "Revestimento_Interno",
    "Acabamento_Interno",
    "Revestimento_Externo",
    "Acabamento_Externo",
    "Cobertura"
]"""

"""# Mapeamento de nomes para display (com acentos/espaços)
CARACTERISTICAS_DISPLAY = {
    "Estrutura": "Estrutura",
    "Esquadrias": "Esquadrias",
    "Piso": "Piso",
    "Forro": "Forro",
    "Instalacao_Eletrica": "Instalação Elétrica",
    "Instalacao_Sanitaria": "Instalação Sanitária",
    "Revestimento_Interno": "Revestimento Interno",
    "Acabamento_Interno": "Acabamento Interno",
    "Revestimento_Externo": "Revestimento Externo",
    "Acabamento_Externo": "Acabamento Externo",
    "Cobertura": "Cobertura"
}"""

# Definição das 11 características a serem avaliadas
CARACTERISTICAS = [
    "Estrutura",
    "Esquadrias",
    "Piso",
    "Forro",
    "Instalacao_Eletrica",
    "Instalacao_Sanitaria",
    "Revestimento_Interno",
    "Acabamento_Interno",
    "Revestimento_Externo",
    "Acabamento_Externo",
    "Cobertura"
]

# Mapeamento de nomes para display (com acentos/espaços)
CARACTERISTICAS_DISPLAY = {
    "Estrutura": "Estrutura",
    "Esquadrias": "Esquadrias",
    "Piso": "Piso",
    "Forro": "Forro",
    "Instalacao_Eletrica": "Instalação Elétrica",
    "Instalacao_Sanitaria": "Instalação Sanitária",
    "Revestimento_Interno": "Revestimento Interno",
    "Acabamento_Interno": "Acabamento Interno",
    "Revestimento_Externo": "Revestimento Externo",
    "Acabamento_Externo": "Acabamento Externo",
    "Cobertura": "Cobertura"
}

# --- Configurar Logging ---
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE_PATH, rotation="10 MB", retention="7 days", level="DEBUG", encoding="utf-8")

def load_prompts_for_characteristic(caracteristica: str) -> Tuple[str, str]:
    """Carrega os prompts system e human para uma característica específica.
    
    Estrutura esperada de pastas:
    prompts/caracteristicas_individuais/
        estrutura/
            system_prompt.txt
            human_prompt.txt
        esquadrias/
            system_prompt.txt
            human_prompt.txt
        ...
    """
    char_dir = PROMPTS_BASE_DIR / MODEL / caracteristica.lower()
    system_file = char_dir / "system_prompt.txt"
    human_file = char_dir / "human_prompt.txt"
    
    try:
        with open(system_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        with open(human_file, 'r', encoding='utf-8') as f:
            human_prompt = f.read()
        
        if not system_prompt or not human_prompt:
            raise ValueError(f"Prompts vazios para {caracteristica}")
        
        logger.info(f"Prompts carregados para '{caracteristica}': {system_file}, {human_file}")
        return system_prompt, human_prompt
    
    except Exception as e:
        logger.error(f"Erro ao carregar prompts para '{caracteristica}': {e}")
        raise

def clean_json_string(json_str: str) -> str:
    """Remove caracteres inválidos mantendo estrutura JSON."""
    cleaned_chars = []
    for char_code in map(ord, json_str):
        if (32 <= char_code <= 126) or char_code in [9, 10, 13, 12, 8] or char_code > 127:
            cleaned_chars.append(chr(char_code))
        elif chr(char_code) == '\\':
            cleaned_chars.append('\\')
    return "".join(cleaned_chars)

def extract_json_from_response(raw_response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extrai e parseia JSON da resposta do modelo."""
    if not raw_response_text:
        return None, "Resposta do modelo vazia."
    
    json_str_candidate = None
    
    # Tenta encontrar bloco ```json ... ```
    match_block = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', raw_response_text, re.DOTALL)
    if match_block:
        json_str_candidate = match_block.group(1)
    else:
        # Busca primeiro { até último }
        first_brace = raw_response_text.find('{')
        last_brace = raw_response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str_candidate = raw_response_text[first_brace : last_brace+1]
        else:
            return None, "Nenhum padrão JSON (bloco ou chaves) encontrado."
    
    if not json_str_candidate:
        return None, "Nenhum candidato a string JSON."
    
    cleaned_json_str = clean_json_string(json_str_candidate)
    
    try:
        return json.loads(cleaned_json_str), None
    except json.JSONDecodeError as e:
        return None, f"Falha parse JSON: {e}. Str: '{cleaned_json_str[:200]}...'"

def _prepare_images_for_ollama(image_dir_path: Path) -> Tuple[List[str], List[str], Optional[str]]:
    """Prepara as imagens de um diretório para serem enviadas à API Ollama."""
    try:
        image_paths_full = get_images_from_directory(str(image_dir_path))
    except ValueError as e:
        return [], [], str(e)
    
    valid_image_paths_obj = []
    for img_path_str in image_paths_full:
        p = Path(img_path_str)
        # Heurística para detectar base64 como path
        if isinstance(img_path_str, str) and len(img_path_str) > 200 and any(
            img_path_str.startswith(prefix) for prefix in ('data:image','UklGR','R0lGO','iVBOR','/9j/')
        ):
            logger.error(f"Pulando imagem - Base64 como path: {img_path_str[:70]}...")
            continue
        
        if not p.is_file():
            logger.warning(f"Pulando path inválido: {p}")
            continue
        valid_image_paths_obj.append(p)
    
    if not valid_image_paths_obj:
        return [], [], "Nenhum path de imagem válido."
    
    encoded_images = []
    valid_image_paths_final_str = []
    
    for p_obj in valid_image_paths_obj:
        try:
            encoded_img_str = _encode_image_to_base64(str(p_obj))
            if not isinstance(encoded_img_str, str):
                logger.error(f"Encode não retornou str para {p_obj}")
                if isinstance(encoded_img_str, bytes):
                    encoded_img_str = encoded_img_str.decode('utf-8')
                else:
                    return [], [str(p) for p in valid_image_paths_obj], f"Encode tipo inesperado: {type(encoded_img_str)}"
            
            encoded_images.append(encoded_img_str)
            valid_image_paths_final_str.append(str(p_obj))
        
        except Exception as e:
            logger.error(f"Erro ao codificar {p_obj}: {e}")
            return [], [str(p) for p in valid_image_paths_obj], f"Falha ao codificar {p_obj}: {e}"
    
    if not encoded_images:
        return [], valid_image_paths_final_str, "Nenhuma imagem codificada."
    return encoded_images, valid_image_paths_final_str, None

def find_property_image_dir(base_dir: Path, property_id_search_term: str, property_type_search_term: str) -> Optional[Path]:
    """Localiza o diretório de imagens de uma propriedade."""
    patterns_to_try = [
        property_id_search_term,
        f"{property_type_search_term}/{property_id_search_term}"
    ]
    
    for pattern_str in patterns_to_try:
        try:
            matches = list(base_dir.glob(pattern_str))
            dir_matches = [m for m in matches if m.is_dir()]
            if dir_matches:
                if len(dir_matches) > 1:
                    logger.warning(f"Múltiplos dirs para '{property_id_search_term}'. Usando {dir_matches[0]}")
                return dir_matches[0]
        except Exception as e:
            logger.error(f"Erro find_property_image_dir: {e}")
            continue
    
    logger.warning(f"Nenhum dir para '{property_id_search_term}' em '{base_dir}'.")
    return None

def process_property_for_characteristic(
    property_dir_path: Path,
    caracteristica: str,
    system_prompt: str,
    human_prompt: str,
    model_name: str,
    temperature: float
) -> Dict[str, Any]:
    """Processa imagens de uma propriedade para avaliar UMA característica específica."""
    process_start_time = time.time()
    
    base_result = {
        "caracteristica_avaliada": caracteristica,
        "image_paths_processed_full": [],
        "image_filenames_processed": [],
        "num_images_found_in_dir": 0,
        "num_images_sent_to_api": 0,
        "raw_model_output": None,
        "parsed_json_object": None,
        "json_parsing_error": None,
        "api_total_duration_s": None,
        "function_total_time_s": 0.0,
        "success_api_call": False,
        "success_json_parsing": False,
        "error_message_processing": None
    }
    
    try:
        # 1. Preparar imagens
        encoded_images, valid_image_paths_str, prep_error = _prepare_images_for_ollama(property_dir_path)
        
        base_result["image_paths_processed_full"] = valid_image_paths_str
        base_result["image_filenames_processed"] = [Path(p).name for p in valid_image_paths_str]
        base_result["num_images_found_in_dir"] = len(valid_image_paths_str)
        
        if prep_error:
            base_result["error_message_processing"] = f"Erro preparação imagens: {prep_error}"
            logger.warning(f"{base_result['error_message_processing']} para {property_dir_path}")
            base_result["function_total_time_s"] = time.time() - process_start_time
            return base_result
        
        base_result["num_images_sent_to_api"] = len(encoded_images)
        
        if not encoded_images:
            base_result["error_message_processing"] = "Nenhuma imagem codificada para API."
            base_result["function_total_time_s"] = time.time() - process_start_time
            return base_result
        
        logger.debug(f"Enviando {len(encoded_images)} imagens de {property_dir_path} para avaliar '{caracteristica}'")
        
        # 2. Chamar API Ollama
        request_options = {"temperature": temperature}
        api_response_data = ollama_generate(
            prompt=human_prompt,
            system=system_prompt,
            model=model_name,
            images=encoded_images,
            options=request_options,
            validate_model_name=False
        )
        
        if "total_duration" in api_response_data:
            base_result["api_total_duration_s"] = api_response_data["total_duration"] / 1_000_000_000.0
        
        if "error" in api_response_data:
            base_result["error_message_processing"] = f"API Ollama: {api_response_data['error']}"
            base_result["raw_model_output"] = api_response_data.get("response", api_response_data.get("response_text"))
            logger.error(f"API Ollama erro para {property_dir_path} ({caracteristica}): {api_response_data['error']}")
        else:
            base_result["success_api_call"] = True
            base_result["raw_model_output"] = api_response_data.get("response")
            
            if base_result["raw_model_output"]:
                parsed_json, parsing_error = extract_json_from_response(base_result["raw_model_output"])
                base_result["parsed_json_object"] = parsed_json
                base_result["json_parsing_error"] = parsing_error
                
                if parsed_json and not parsing_error:
                    base_result["success_json_parsing"] = True
                    logger.info(f"JSON parseado com sucesso para {property_dir_path} ({caracteristica})")
                else:
                    logger.warning(f"Falha no parsing JSON para {property_dir_path} ({caracteristica})")
            else:
                base_result["error_message_processing"] = "API sucesso, mas 'response' vazio."
                base_result["json_parsing_error"] = base_result["error_message_processing"]
                logger.warning(f"{base_result['error_message_processing']} ({property_dir_path}, {caracteristica})")
    
    except ValueError as ve:
        base_result["error_message_processing"] = f"Erro ao preparar dados: {ve}"
        logger.error(f"{base_result['error_message_processing']} para {property_dir_path}")
    except Exception as e:
        base_result["error_message_processing"] = f"Exceção: {type(e).__name__} - {e}"
        logger.exception(f"Exceção em process_property_for_characteristic para {property_dir_path} ({caracteristica})")
    
    base_result["function_total_time_s"] = time.time() - process_start_time
    return base_result

def run_generation_for_single_characteristic(caracteristica: str) -> None:
    """Executa a geração para UMA característica em TODAS as propriedades do dataset."""
    
    char_display = CARACTERISTICAS_DISPLAY.get(caracteristica, caracteristica)
    logger.info("="*80)
    logger.info(f"INICIANDO PROCESSAMENTO DA CARACTERÍSTICA: {char_display}")
    logger.info("="*80)
    
    generation_start_time = datetime.now(timezone.utc)
    
    # Carregar prompts específicos para esta característica
    try:
        system_prompt, human_prompt = load_prompts_for_characteristic(caracteristica)
    except Exception as e:
        logger.critical(f"Não foi possível carregar prompts para '{caracteristica}': {e}")
        return
    
    # Carregar dataset
    if not DATASET_PATH.is_file():
        logger.critical(f"Dataset não encontrado: {DATASET_PATH}")
        return
    
    try:
        df_dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset carregado: {len(df_dataset)} propriedades")
    except Exception as e:
        logger.critical(f"Erro ao carregar dataset: {e}")
        return
    
    # Criar diretório de saída para esta característica
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = MODEL_NAME_TO_USE.replace(':', '_').replace('/', '_').replace('.', 'p')
    temp_safe = str(TEMPERATURE_TO_USE).replace('.', 'p')
    
    output_dir = OUTPUT_BASE_DIR / caracteristica.lower() / f"{model_safe}_temp_{temp_safe}_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{caracteristica.lower()}_results.json"
    logger.info(f"Resultados serão salvos em: {results_file}")
    
    # Processar todas as propriedades
    lista_resultados = []
    
    for idx, row in tqdm(df_dataset.iterrows(), total=len(df_dataset), desc=f"{char_display}"):
        property_id_output = idx + 1
        property_folder_search = str(row.get('id_anuncio'))
        property_type_search = str(row.get('tipo_imovel'))
        
        property_dir = find_property_image_dir(IMAGES_BASE_DIR, property_folder_search, property_type_search)
        
        resultado_item = {
            "property_id": property_id_output,
            "dataset_idx": idx,
            "caracteristica": char_display,
            "modelo": MODEL_NAME_TO_USE,
            "temperatura": TEMPERATURE_TO_USE,
            "search_term_folder": property_folder_search,
            "type_folder": property_type_search,
            "property_directory": None
        }
        
        if not property_dir:
            logger.warning(f"Idx {idx} ({char_display}): dir não encontrado para '{property_folder_search}'")
            resultado_item["status"] = "erro_diretorio_nao_encontrado"
            resultado_item["error_message"] = f"Diretório não encontrado"
        else:
            resultado_item["property_directory"] = str(property_dir)
            
            processing_details = process_property_for_characteristic(
                property_dir,
                caracteristica,
                system_prompt,
                human_prompt,
                MODEL_NAME_TO_USE,
                TEMPERATURE_TO_USE
            )
            
            resultado_item.update({
                "num_imagens_encontradas": processing_details.get("num_images_found_in_dir", 0),
                "num_imagens_enviadas": processing_details.get("num_images_sent_to_api", 0),
                "latencia_api_s": processing_details.get("api_total_duration_s"),
                "latencia_total_s": processing_details.get("function_total_time_s"),
                "resposta_raw": processing_details.get("raw_model_output"),
                "resposta_parsed": processing_details.get("parsed_json_object"),
                "status_api": "sucesso" if processing_details.get("success_api_call") else "falha",
                "status_parsing": "sucesso" if processing_details.get("success_json_parsing") else "falha",
                "erro_parsing": processing_details.get("json_parsing_error"),
                "erro_processamento": processing_details.get("error_message_processing"),
                "imagens_processadas": processing_details.get("image_filenames_processed", [])
            })
            
            if not processing_details.get("success_api_call"):
                resultado_item["status"] = "erro_api"
            elif not processing_details.get("success_json_parsing"):
                resultado_item["status"] = "erro_parsing_json"
            else:
                resultado_item["status"] = "sucesso"
        
        lista_resultados.append(resultado_item)
    
    # Montar JSON final
    generation_end_time = datetime.now(timezone.utc)
    output_data = {
        "metadata": {
            "caracteristica_avaliada": char_display,
            "timestamp_inicio": generation_start_time.isoformat(),
            "timestamp_fim": generation_end_time.isoformat(),
            "duracao_total_s": (generation_end_time - generation_start_time).total_seconds(),
            "modelo_usado": MODEL_NAME_TO_USE,
            "temperatura_usada": TEMPERATURE_TO_USE,
            "total_propriedades": len(df_dataset),
            "total_processadas": len(lista_resultados),
            "dataset_usado": str(DATASET_PATH.name),
            "prompts_dir": str(PROMPTS_BASE_DIR / MODEL / caracteristica.lower())
        },
        "resultados": lista_resultados
    }
    
    # Salvar JSON
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False,
                     default=lambda o: int(o) if isinstance(o, np.integer) else
                                     float(o) if isinstance(o, np.floating) else
                                     o.tolist() if isinstance(o, np.ndarray) else str(o))
        logger.info(f"Resultados salvos em: {results_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar JSON para {caracteristica}: {e}")
    
    logger.info(f"Finalizado processamento de '{char_display}' - Duração: {(generation_end_time - generation_start_time).total_seconds():.2f}s")

def run_all_characteristics() -> None:
    """Executa a geração para TODAS as 11 características sequencialmente."""
    overall_start = datetime.now(timezone.utc)
    
    logger.info("="*80)
    logger.info("INICIANDO PROCESSAMENTO DE TODAS AS CARACTERÍSTICAS")
    logger.info(f"Modelo: {MODEL_NAME_TO_USE}")
    logger.info(f"Temperatura: {TEMPERATURE_TO_USE}")
    logger.info(f"Total de características: {len(CARACTERISTICAS)}")
    logger.info("="*80)
    
    for i, caracteristica in enumerate(CARACTERISTICAS, 1):
        logger.info(f"\n>>> Característica {i}/{len(CARACTERISTICAS)}: {CARACTERISTICAS_DISPLAY[caracteristica]}")
        run_generation_for_single_characteristic(caracteristica)
    
    overall_end = datetime.now(timezone.utc)
    total_duration = (overall_end - overall_start).total_seconds()
    
    logger.info("="*80)
    logger.info("TODAS AS CARACTERÍSTICAS PROCESSADAS COM SUCESSO!")
    logger.info(f"Duração total: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    logger.info("="*80)

# --- Função principal ---
if __name__ == "__main__":
    try:
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Iniciando script de GERAÇÃO POR CARACTERÍSTICA INDIVIDUAL (Ollama)")
        logger.info(f"Modelo: {MODEL_NAME_TO_USE}, Temperatura: {TEMPERATURE_TO_USE}")
        
        # Validar modelo antes de iniciar
        try:
            if not validate_model(MODEL_NAME_TO_USE):
                logger.critical(f"Modelo '{MODEL_NAME_TO_USE}' não está disponível no Ollama.")
                sys.exit(1)
            logger.info(f"Modelo '{MODEL_NAME_TO_USE}' validado com sucesso.")
        except Exception as e:
            logger.critical(f"Erro ao validar modelo: {e}")
            sys.exit(1)
        
        if "--test" in sys.argv or "-t" in sys.argv:
            logger.info("Modo de teste: processando apenas 'Estrutura' com 2 propriedades")
            # Criar versão de teste limitada
            original_dataset = DATASET_PATH
            DATASET_PATH = Path("./data/CSVs/test_2_properties.csv")
            if not DATASET_PATH.exists():
                df_test = pd.read_csv(original_dataset, nrows=2)
                df_test.to_csv(DATASET_PATH, index=False)
            
            run_generation_for_single_characteristic("Estrutura")
            logger.info("Teste concluído!")
        else:
            run_all_characteristics()
        
        logger.info("Script finalizado com sucesso!")
        
    except KeyboardInterrupt:
        logger.warning("Interrompido pelo usuário (KeyboardInterrupt)")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")
        sys.exit(1)