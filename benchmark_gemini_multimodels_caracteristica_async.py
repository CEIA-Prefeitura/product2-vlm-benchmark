"""Script ASSÍNCRONO para GERAÇÃO de respostas JSON de modelos MULTIMODAIS GEMINI.

Este script processa múltiplas propriedades em PARALELO usando asyncio e workers,
acelerando significativamente o processamento de grandes datasets.

Características:
- Processamento assíncrono com controle de concorrência
- Workers configuráveis via variável de ambiente
- Barra de progresso em tempo real
- Tratamento robusto de erros
"""

import os
import json
import time
import re
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import sys
from dotenv import load_dotenv

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from loguru import logger

try:
    from utils.helpers import get_images_from_directory
    from gemini_connector_async import gemini_generate_async, validate_gemini_model
except ImportError as e:
    logger.critical(f"Erro de importação: {e}. Verifique gemini_connector_async.py e utils/helpers.py.")
    raise SystemExit(1)

load_dotenv()

# --- Configuração ---
MODEL_NAME_TO_USE = os.getenv("GEMINI_BENCHMARK_MODELS", "gemini-2.5-flash").strip()
TEMPERATURE_TO_USE = float(os.getenv("BENCHMARK_TEMPERATURES", "0.0"))
MAX_WORKERS = int(os.getenv("GEMINI_MAX_WORKERS", "8"))

DATASET_PATH = Path(os.getenv("BENCHMARK_DATASET_PATH", "./data/CSVs/91_casas_terreas_anotadas.csv"))
IMAGES_BASE_DIR = Path(os.getenv("BENCHMARK_IMAGES_DIR", "./data/imagens/images"))
OUTPUT_BASE_DIR = Path(os.getenv("SINGLE_CHAR_OUTPUT_DIR", "./resultados_por_caracteristica"))
PROMPTS_BASE_DIR = Path(os.getenv("BENCHMARK_PROMPTS_DIR", "./prompts/prompts_por_caracteristica"))

LOG_FILE_PATH = "logs/generation_gemini_single_char_async.log"

# Definição das 11 características a serem avaliadas
CARACTERISTICAS = [
    "Estrutura", "Esquadrias", "Piso", "Forro", "Instalacao_Eletrica",
    "Instalacao_Sanitaria", "Revestimento_Interno", "Acabamento_Interno",
    "Revestimento_Externo", "Acabamento_Externo", "Cobertura"
]

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
    """Carrega os prompts system e human para uma característica específica."""
    char_dir = PROMPTS_BASE_DIR / caracteristica.lower()
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
    
    match_block = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', raw_response_text, re.DOTALL)
    if match_block:
        json_str_candidate = match_block.group(1)
    else:
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
    
    return None

async def process_property_for_characteristic_async(
    property_dir_path: Path,
    caracteristica: str,
    system_prompt: str,
    human_prompt: str,
    model_name: str,
    temperature: float
) -> Dict[str, Any]:
    """Versão ASSÍNCRONA do processamento de uma propriedade para uma característica."""
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
        "api_response_time_s": None,
        "api_usage_metadata": None,
        "function_total_time_s": 0.0,
        "success_api_call": False,
        "success_json_parsing": False,
        "error_message_processing": None
    }
    
    try:
        # 1. Listar imagens (I/O bound - usar thread)
        image_paths_in_dir = await asyncio.to_thread(get_images_from_directory, str(property_dir_path))
        base_result["num_images_found_in_dir"] = len(image_paths_in_dir)
        base_result["image_paths_processed_full"] = image_paths_in_dir
        base_result["image_filenames_processed"] = [Path(p).name for p in image_paths_in_dir]
        
        if not image_paths_in_dir:
            base_result["error_message_processing"] = "Nenhuma imagem encontrada no diretório."
            logger.warning(f"{base_result['error_message_processing']} para {property_dir_path}")
            base_result["function_total_time_s"] = time.time() - process_start_time
            return base_result
        
        base_result["num_images_sent_to_api"] = len(image_paths_in_dir)
        
        # 2. Construir prompt_parts
        prompt_parts_for_gemini = []
        prompt_parts_for_gemini.append(human_prompt)
        for img_path in image_paths_in_dir:
            prompt_parts_for_gemini.append({"type": "image_path", "image_path": img_path})
        
        logger.debug(f"Enviando {len(image_paths_in_dir)} imagens de {property_dir_path} para avaliar '{caracteristica}'")
        
        # 3. Chamar API Gemini ASSINCRONAMENTE
        api_response_data = await gemini_generate_async(
            prompt_parts=prompt_parts_for_gemini,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature
        )
        
        base_result["api_response_time_s"] = api_response_data.get("response_time_s")
        base_result["api_usage_metadata"] = api_response_data.get("usage_metadata")
        
        if api_response_data.get("error"):
            base_result["error_message_processing"] = f"API Gemini: {api_response_data['error']}"
            base_result["raw_model_output"] = api_response_data.get("response")
            logger.error(f"API Gemini erro para {property_dir_path} ({caracteristica}): {api_response_data['error']}")
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
    
    except ValueError as ve:
        base_result["error_message_processing"] = f"Erro ao preparar dados: {ve}"
        logger.error(f"{base_result['error_message_processing']} para {property_dir_path}")
    except Exception as e:
        base_result["error_message_processing"] = f"Exceção: {type(e).__name__} - {e}"
        logger.exception(f"Exceção em process_property_for_characteristic_async para {property_dir_path} ({caracteristica})")
    
    base_result["function_total_time_s"] = time.time() - process_start_time
    return base_result

async def run_generation_for_single_characteristic_async(caracteristica: str, df_dataset: pd.DataFrame) -> None:
    """Versão ASSÍNCRONA da geração para uma característica em TODAS as propriedades."""
    
    char_display = CARACTERISTICAS_DISPLAY.get(caracteristica, caracteristica)
    logger.info("="*80)
    logger.info(f"INICIANDO PROCESSAMENTO DA CARACTERÍSTICA: {char_display}")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info("="*80)
    
    generation_start_time = datetime.now(timezone.utc)
    
    # Carregar prompts específicos para esta característica
    try:
        system_prompt, human_prompt = load_prompts_for_characteristic(caracteristica)
    except Exception as e:
        logger.critical(f"Não foi possível carregar prompts para '{caracteristica}': {e}")
        return
    
    # Criar diretório de saída para esta característica
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = MODEL_NAME_TO_USE.replace(':', '_').replace('/', '_').replace('.', 'p')
    temp_safe = str(TEMPERATURE_TO_USE).replace('.', 'p')
    
    output_dir = OUTPUT_BASE_DIR / caracteristica.lower() / f"{model_safe}_temp_{temp_safe}_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{caracteristica.lower()}_results.json"
    logger.info(f"Resultados serão salvos em: {results_file}")
    
    # Criar tasks para processamento paralelo
    tasks = []
    property_metadata = []
    
    for idx, row in df_dataset.iterrows():
        property_id_output = idx + 1
        property_folder_search = str(row.get('id_anuncio'))
        property_type_search = str(row.get('tipo_imovel'))
        
        property_dir = find_property_image_dir(IMAGES_BASE_DIR, property_folder_search, property_type_search)
        
        metadata = {
            "property_id": property_id_output,
            "dataset_idx": idx,
            "search_term_folder": property_folder_search,
            "type_folder": property_type_search,
            "property_directory": str(property_dir) if property_dir else None
        }
        property_metadata.append(metadata)
        
        if property_dir:
            task = process_property_for_characteristic_async(
                property_dir,
                caracteristica,
                system_prompt,
                human_prompt,
                MODEL_NAME_TO_USE,
                TEMPERATURE_TO_USE
            )
            tasks.append(task)
        else:
            # Criar resultado de erro para propriedades sem diretório
            tasks.append(asyncio.sleep(0))  # Placeholder assíncrono
    
    # Executar todas as tasks em paralelo com barra de progresso
    logger.info(f"Processando {len(tasks)} propriedades em paralelo...")
    processing_results = []
    
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{char_display}"):
        result = await coro
        processing_results.append(result)
    
    # Combinar metadados com resultados
    lista_resultados = []
    for i, (metadata, processing_result) in enumerate(zip(property_metadata, processing_results)):
        resultado_item = {
            "property_id": metadata["property_id"],
            "dataset_idx": metadata["dataset_idx"],
            "caracteristica": char_display,
            "modelo": MODEL_NAME_TO_USE,
            "temperatura": TEMPERATURE_TO_USE,
            "search_term_folder": metadata["search_term_folder"],
            "type_folder": metadata["type_folder"],
            "property_directory": metadata["property_directory"]
        }
        
        if not metadata["property_directory"]:
            resultado_item["status"] = "erro_diretorio_nao_encontrado"
            resultado_item["error_message"] = "Diretório não encontrado"
        elif isinstance(processing_result, dict) and "function_total_time_s" in processing_result:
            resultado_item.update({
                "num_imagens_encontradas": processing_result.get("num_images_found_in_dir", 0),
                "num_imagens_enviadas": processing_result.get("num_images_sent_to_api", 0),
                "api_usage_metadata": processing_result.get("api_usage_metadata"),
                "latencia_api_s": processing_result.get("api_response_time_s"),
                "latencia_total_s": processing_result.get("function_total_time_s"),
                "resposta_raw": processing_result.get("raw_model_output"),
                "resposta_parsed": processing_result.get("parsed_json_object"),
                "status_api": "sucesso" if processing_result.get("success_api_call") else "falha",
                "status_parsing": "sucesso" if processing_result.get("success_json_parsing") else "falha",
                "erro_parsing": processing_result.get("json_parsing_error"),
                "erro_processamento": processing_result.get("error_message_processing"),
                "imagens_processadas": processing_result.get("image_filenames_processed", [])
            })
            
            if not processing_result.get("success_api_call"):
                resultado_item["status"] = "erro_api"
            elif not processing_result.get("success_json_parsing"):
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
            "max_workers": MAX_WORKERS,
            "total_propriedades": len(df_dataset),
            "total_processadas": len(lista_resultados),
            "dataset_usado": str(DATASET_PATH.name),
            "prompts_dir": str(PROMPTS_BASE_DIR / caracteristica.lower())
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
    
    duration = (generation_end_time - generation_start_time).total_seconds()
    logger.info(f"Finalizado processamento de '{char_display}' - Duração: {duration:.2f}s ({duration/60:.2f} min)")

async def run_all_characteristics_async() -> None:
    """Executa a geração para TODAS as características."""
    overall_start = datetime.now(timezone.utc)
    
    logger.info("="*80)
    logger.info("INICIANDO PROCESSAMENTO ASSÍNCRONO DE TODAS AS CARACTERÍSTICAS")
    logger.info(f"Modelo: {MODEL_NAME_TO_USE}")
    logger.info(f"Temperatura: {TEMPERATURE_TO_USE}")
    logger.info(f"Workers simultâneos: {MAX_WORKERS}")
    logger.info(f"Total de características: {len(CARACTERISTICAS)}")
    logger.info("="*80)
    
    # Carregar dataset uma vez
    if not DATASET_PATH.is_file():
        logger.critical(f"Dataset não encontrado: {DATASET_PATH}")
        return
    
    try:
        df_dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset carregado: {len(df_dataset)} propriedades")
    except Exception as e:
        logger.critical(f"Erro ao carregar dataset: {e}")
        return
    
    # Processar cada característica sequencialmente (mas cada uma usa workers)
    for i, caracteristica in enumerate(CARACTERISTICAS, 1):
        logger.info(f"\n>>> Característica {i}/{len(CARACTERISTICAS)}: {CARACTERISTICAS_DISPLAY[caracteristica]}")
        await run_generation_for_single_characteristic_async(caracteristica, df_dataset)
    
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
        
        logger.info(f"Iniciando script ASSÍNCRONO de GERAÇÃO POR CARACTERÍSTICA")
        logger.info(f"Modelo: {MODEL_NAME_TO_USE}, Temperatura: {TEMPERATURE_TO_USE}")
        logger.info(f"Workers: {MAX_WORKERS}")
        
        if "--test" in sys.argv or "-t" in sys.argv:
            logger.info("Modo de teste: processando apenas 'Estrutura' com 2 propriedades")
            original_dataset = DATASET_PATH
            DATASET_PATH = Path("./data/CSVs/test_2_properties.csv")
            if not DATASET_PATH.exists():
                df_test = pd.read_csv(original_dataset, nrows=2)
                df_test.to_csv(DATASET_PATH, index=False)
            
            df = pd.read_csv(DATASET_PATH)
            asyncio.run(run_generation_for_single_characteristic_async("Estrutura", df))
            logger.info("Teste concluído!")
        else:
            asyncio.run(run_all_characteristics_async())
        
        logger.info("Script finalizado com sucesso!")
        
    except KeyboardInterrupt:
        logger.warning("Interrompido pelo usuário (KeyboardInterrupt)")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erro fatal: {e}")
        sys.exit(1)