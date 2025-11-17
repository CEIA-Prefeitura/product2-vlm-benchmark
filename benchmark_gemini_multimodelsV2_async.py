"""Script ASSÍNCRONO para GERAÇÃO de respostas JSON de modelos MULTIMODAIS GEMINI.

Este script processa múltiplas propriedades em PARALELO para diferentes configurações
de modelos e temperaturas, acelerando significativamente o processamento.

Características:
- Processamento assíncrono com múltiplos workers
- Suporte a múltiplos modelos e temperaturas
- Barra de progresso em tempo real
- Tratamento robusto de erros
"""

import os
import json
import time
import re
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import sys
from dotenv import load_dotenv

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from loguru import logger

# Tentativa de importar módulos utilitários e o conector Gemini
try:
    from utils.helpers import get_images_from_directory
    from gemini_connector_async import (
        gemini_generate_async,
        list_available_gemini_models,
        validate_gemini_model
    )
except ImportError as e:
    logger.critical(f"Erro de importação: {e}. Verifique gemini_connector_async.py e utils/helpers.py.")
    raise SystemExit(1)

load_dotenv()

# --- Configuração ---
MODEL_NAMES_TO_TEST = os.getenv("GEMINI_BENCHMARK_MODELS", "gemini-2.5-flash,gemini-2.5-pro").split(',')
TEMPERATURES_TO_TEST = [float(t) for t in os.getenv("BENCHMARK_TEMPERATURES", "0.0,0.5,1.0").split(',')]
MAX_WORKERS = int(os.getenv("GEMINI_MAX_WORKERS", "8"))

DATASET_PATH = Path(os.getenv("BENCHMARK_DATASET_PATH", "./data/CSVs/91_casas_terreas_anotadas.csv"))
IMAGES_BASE_DIR = Path(os.getenv("BENCHMARK_IMAGES_DIR", "./data/imagens/images"))
OUTPUT_BASE_DIR = Path(os.getenv("BENCHMARK_OUTPUT_DIR", "./resultados_geracao_gemini"))
PROMPTS_DIR = Path(os.getenv("BENCHMARK_PROMPTS_DIR", "./prompts/prompts_especialistas/casa_terrea"))
TIPO_PROMPT_LABEL = "analise_casa_terrea_gemini_v5"

SYSTEM_PROMPT_FILE = PROMPTS_DIR / "gemini" / "system_prompt_gemini_casa_terrea_V4.txt"
HUMAN_PROMPT_FILE = PROMPTS_DIR / "gemini" / "human_prompt_gemini_casa_terrea_V4.txt"

LOG_FILE_PATH = "logs/generation_gemini_async.log"

# --- Configurar Logging, Carregar Prompts ---
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(LOG_FILE_PATH, rotation="10 MB", retention="7 days", level="DEBUG", encoding="utf-8")

SYSTEM_PROMPT_CONTENT = ""
HUMAN_PROMPT_CONTENT = ""
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT_CONTENT = f.read()
    with open(HUMAN_PROMPT_FILE, 'r', encoding='utf-8') as f:
        HUMAN_PROMPT_CONTENT = f.read()
    logger.info(f"System prompt carregado de: {SYSTEM_PROMPT_FILE}")
    logger.info(f"Human prompt carregado de: {HUMAN_PROMPT_FILE}")
    if not SYSTEM_PROMPT_CONTENT or not HUMAN_PROMPT_CONTENT:
        raise ValueError("Prompts vazios.")
except Exception as e:
    logger.critical(f"Erro ao ler prompts: {e}")
    raise SystemExit(f"Erro crítico ao ler prompts: {e}")

# --- Funções Auxiliares ---
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
        return None, f"Falha parse JSON (limpa): {e}. Str: '{cleaned_json_str[:200]}...'"

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

# --- Função de Processamento Assíncrona ---
async def process_property_with_gemini_async(
    property_dir_path: Path,
    current_model_name: str,
    current_temperature: float
) -> Dict[str, Any]:
    """
    Versão ASSÍNCRONA do processamento de uma propriedade com modelo Gemini.
    
    Args:
        property_dir_path: Path para o diretório das imagens da propriedade.
        current_model_name: Nome do modelo Gemini a ser usado.
        current_temperature: Temperatura para a geração.
    
    Returns:
        Um dicionário contendo detalhes do processamento.
    """
    process_start_time = time.time()
    base_result = {
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
        # 1. Listar imagens do diretório (I/O bound - usar thread)
        image_paths_in_dir = await asyncio.to_thread(get_images_from_directory, str(property_dir_path))
        base_result["num_images_found_in_dir"] = len(image_paths_in_dir)
        base_result["image_paths_processed_full"] = image_paths_in_dir
        base_result["image_filenames_processed"] = [Path(p).name for p in image_paths_in_dir]

        if not image_paths_in_dir:
            base_result["error_message_processing"] = "Nenhuma imagem encontrada no diretório para processar."
            logger.warning(f"{base_result['error_message_processing']} para {property_dir_path}")
            base_result["function_total_time_s"] = time.time() - process_start_time
            return base_result
        
        base_result["num_images_sent_to_api"] = len(image_paths_in_dir)

        # 2. Construir `prompt_parts` para gemini_generate
        prompt_parts_for_gemini: List[Union[str, Dict[str, Any]]] = []
        prompt_parts_for_gemini.append(HUMAN_PROMPT_CONTENT)
        for img_path in image_paths_in_dir:
            prompt_parts_for_gemini.append({"type": "image_path", "image_path": img_path})

        logger.debug(f"Enviando {len(image_paths_in_dir)} imagens de {property_dir_path} para o modelo {current_model_name} com temp: {current_temperature}")
        
        # 3. Chamar gemini_generate_async
        api_response_data = await gemini_generate_async(
            prompt_parts=prompt_parts_for_gemini,
            system_prompt=SYSTEM_PROMPT_CONTENT,
            model_name=current_model_name,
            temperature=current_temperature
        )

        base_result["api_response_time_s"] = api_response_data.get("response_time_s")
        base_result["api_usage_metadata"] = api_response_data.get("usage_metadata")

        if api_response_data.get("error"):
            base_result["error_message_processing"] = f"API Gemini: {api_response_data['error']}"
            base_result["raw_model_output"] = api_response_data.get("response")
            logger.error(f"API Gemini retornou erro para {property_dir_path} (modelo {current_model_name}, temp {current_temperature}): {api_response_data['error']}")
        else:
            base_result["success_api_call"] = True
            base_result["raw_model_output"] = api_response_data.get("response")

            if base_result["raw_model_output"]:
                parsed_json, parsing_error = extract_json_from_response(base_result["raw_model_output"])
                base_result["parsed_json_object"] = parsed_json
                base_result["json_parsing_error"] = parsing_error
                if parsed_json and not parsing_error:
                    base_result["success_json_parsing"] = True
                    logger.info(f"JSON parseado com sucesso para {property_dir_path} (modelo {current_model_name}, temp {current_temperature}).")
                else:
                    logger.warning(f"Falha no parsing do JSON para {property_dir_path} (modelo {current_model_name}, temp {current_temperature}).")
            else:
                base_result["error_message_processing"] = "API Gemini sucesso, mas 'response' está vazio."
                base_result["json_parsing_error"] = base_result["error_message_processing"]
                logger.warning(f"{base_result['error_message_processing']} (Dir: {property_dir_path}, Modelo: {current_model_name}, Temp: {current_temperature})")
    
    except ValueError as ve:
        base_result["error_message_processing"] = f"Erro ao preparar dados: {ve}"
        logger.error(f"{base_result['error_message_processing']} para {property_dir_path}")
    except Exception as e:
        base_result["error_message_processing"] = f"Exceção em process_property_with_gemini_async: {type(e).__name__} - {e}"
        logger.exception(f"Exceção em process_property_with_gemini_async para {property_dir_path} (modelo {current_model_name}, temp {current_temperature})")

    base_result["function_total_time_s"] = time.time() - process_start_time
    return base_result

# --- Função Principal de Geração Assíncrona ---
async def run_multiconfig_gemini_generation_async() -> None:
    """Executa a geração para múltiplas configurações de forma assíncrona."""
    overall_start_time_iso = datetime.now(timezone.utc)

    logger.info(f"Iniciando execução ASSÍNCRONA multi-configuração GEMINI em: {overall_start_time_iso.isoformat()}")
    logger.info(f"Modelos Gemini a serem testados: {MODEL_NAMES_TO_TEST}")
    logger.info(f"Temperaturas a serem testadas: {TEMPERATURES_TO_TEST}")
    logger.info(f"Workers simultâneos: {MAX_WORKERS}")

    if not DATASET_PATH.is_file():
        logger.critical(f"Dataset não encontrado: {DATASET_PATH}")
        raise SystemExit(1)
    if not IMAGES_BASE_DIR.is_dir():
        logger.critical(f"Dir imagens não encontrado: {IMAGES_BASE_DIR}")
        raise SystemExit(1)

    try:
        df_dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset carregado de '{DATASET_PATH}' com {len(df_dataset)} propriedades.")
    except Exception as e:
        logger.critical(f"Erro ao carregar dataset: {e}")
        raise SystemExit(1)

    for model_name_current in MODEL_NAMES_TO_TEST:
        model_name_current = model_name_current.strip()
        if not model_name_current:
            continue

        logger.info(f"--- Iniciando testes para o MODELO GEMINI: {model_name_current} ---")

        for temp_current in TEMPERATURES_TO_TEST:
            logger.info(f"--- Iniciando geração ASSÍNCRONA para MODELO: {model_name_current}, TEMPERATURA: {temp_current} ---")

            generation_start_time_iso = datetime.now(timezone.utc)
            
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_safe = model_name_current.replace(':', '_').replace('/', '_').replace('.', 'p')
            temp_safe = str(temp_current).replace('.', 'p')

            current_run_output_dir = OUTPUT_BASE_DIR / f"output_{model_name_safe}_temp_{temp_safe}_{run_timestamp}"
            current_run_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Resultados para {model_name_current} @ temp {temp_current} serão salvos em: {current_run_output_dir}")

            generation_results_path = current_run_output_dir / "generation_results.json"
            
            # Criar tasks para processamento paralelo
            tasks = []
            property_metadata = []

            for idx, row_dataset in df_dataset.iterrows():
                current_property_id_output = idx + 1
                property_folder_search_term = str(row_dataset.get('id_anuncio'))
                property_folder_type_search_term = str(row_dataset.get('tipo_imovel'))
                property_dir_path = find_property_image_dir(IMAGES_BASE_DIR, property_folder_search_term, property_folder_type_search_term)
                
                metadata = {
                    "property_id": current_property_id_output,
                    "dataset_idx": idx,
                    "search_term_folder": property_folder_search_term,
                    "type_folder": property_folder_type_search_term,
                    "property_directory": str(property_dir_path) if property_dir_path else None
                }
                property_metadata.append(metadata)

                if property_dir_path:
                    task = process_property_with_gemini_async(
                        property_dir_path,
                        current_model_name=model_name_current,
                        current_temperature=temp_current
                    )
                    tasks.append(task)
                else:
                    # Placeholder para propriedades sem diretório
                    tasks.append(asyncio.sleep(0))

            # Executar todas as tasks em paralelo com barra de progresso
            logger.info(f"Processando {len(tasks)} propriedades em paralelo para {model_name_current} @ {temp_current}...")
            processing_results = []
            
            for coro in async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"{model_name_current} @ {temp_current}"
            ):
                result = await coro
                processing_results.append(result)

            # Combinar metadados com resultados
            lista_de_resultados_formatados_run = []
            for metadata, processing_result in zip(property_metadata, processing_results):
                resultado_item: Dict[str, Any] = {
                    "property_id": metadata["property_id"],
                    "dataset_idx": metadata["dataset_idx"],
                    "modelo": model_name_current,
                    "temperatura_usada": temp_current,
                    "search_term_folder": metadata["search_term_folder"],
                    "type_folder": metadata["type_folder"],
                    "property_directory_processed": metadata["property_directory"]
                }

                if not metadata["property_directory"]:
                    resultado_item["status_geracao"] = "erro_diretorio_nao_encontrado"
                    resultado_item["error_message"] = f"Diretório para '{metadata['search_term_folder']}' não encontrado."
                elif isinstance(processing_result, dict) and "function_total_time_s" in processing_result:
                    resultado_item.update({
                        "num_imagens_encontradas": processing_result.get("num_images_found_in_dir", 0),
                        "num_imagens_enviadas_api": processing_result.get("num_images_sent_to_api", 0),
                        "tipo_prompt": TIPO_PROMPT_LABEL,
                        "api_usage_metadata": processing_result.get("api_usage_metadata"),
                        "latencia_api_s": processing_result.get("api_response_time_s"),
                        "latencia_funcao_processamento_s": processing_result.get("function_total_time_s"),
                        "resposta_modelo_raw_str": processing_result.get("raw_model_output"),
                        "resposta_modelo_parsed_dict": processing_result.get("parsed_json_object"),
                        "status_api_call": "sucesso" if processing_result.get("success_api_call") else "falha",
                        "status_json_parsing": "sucesso" if processing_result.get("success_json_parsing") else "falha",
                        "json_parsing_error_msg": processing_result.get("json_parsing_error"),
                        "processing_error_msg": processing_result.get("error_message_processing"),
                        "imagens_processadas_nomes": processing_result.get("image_filenames_processed", []),
                    })
                    if not processing_result.get("success_api_call"):
                        resultado_item["status_geracao"] = "erro_api"
                    elif not processing_result.get("success_json_parsing"):
                        resultado_item["status_geracao"] = "erro_parsing_json"
                    else:
                        resultado_item["status_geracao"] = "processado_com_sucesso"
                
                lista_de_resultados_formatados_run.append(resultado_item)

            # Montar JSON final
            generation_end_time = datetime.now(timezone.utc)
            output_final_geracao_run = {
                "metadata_geracao_run": {
                    "timestamp_geracao_inicial_run": generation_start_time_iso.isoformat(),
                    "timestamp_geracao_final_run": generation_end_time.isoformat(),
                    "timestamp_total_run": (generation_end_time - generation_start_time_iso).total_seconds(),
                    "modelo_usado": model_name_current,
                    "temperatura_usada": temp_current,
                    "max_workers": MAX_WORKERS,
                    "total_propriedades_dataset": len(df_dataset),
                    "total_itens_na_saida_run": len(lista_de_resultados_formatados_run),
                    "dataset_usado": str(DATASET_PATH.name),
                    "tipo_prompt_usado": TIPO_PROMPT_LABEL,
                    "system_prompt_file": str(SYSTEM_PROMPT_FILE.name),
                    "human_prompt_file": str(HUMAN_PROMPT_FILE.name),
                },
                "resultados_gerados_run": lista_de_resultados_formatados_run
            }

            # Salvar JSON
            try:
                with open(generation_results_path, 'w', encoding='utf-8') as f_out:
                    json.dump(output_final_geracao_run, f_out, indent=2, ensure_ascii=False,
                              default=lambda o: int(o) if isinstance(o, np.integer) else
                                                float(o) if isinstance(o, np.floating) else
                                                o.tolist() if isinstance(o, np.ndarray) else str(o))
                logger.info(f"Resultados para {model_name_current} @ temp {temp_current} salvos em: {generation_results_path}")
            except Exception as e_json_save:
                logger.error(f"Erro ao salvar JSON para {model_name_current} @ temp {temp_current}: {e_json_save}")
            
            duration = (generation_end_time - generation_start_time_iso).total_seconds()
            logger.info(f"--- Finalizada geração ASSÍNCRONA para MODELO GEMINI: {model_name_current}, TEMPERATURA: {temp_current} ---")
            logger.info(f"    Duração: {duration:.2f}s ({duration/60:.2f} min)")
        
        logger.info(f"--- Finalizados todos os testes de temperatura para o MODELO GEMINI: {model_name_current} ---")
    
    overall_end = datetime.now(timezone.utc)
    total_duration = (overall_end - overall_start_time_iso).total_seconds()
    logger.info(f"Execução ASSÍNCRONA multi-configuração GEMINI finalizada em: {overall_end.isoformat()}")
    logger.info(f"Duração total: {total_duration:.2f}s ({total_duration/60:.2f} min)")

# --- Bloco de Testes de Diagnóstico (Assíncrono) ---
async def run_diagnostic_tests_async(model_to_test: str, temp_to_test: float, num_properties_to_test: int = 1):
    """Testes de diagnóstico assíncronos."""
    logger.info("="*20 + f" INICIANDO TESTES DE DIAGNÓSTICO ASSÍNCRONOS (MODELO GEMINI: {model_to_test}, TEMP: {temp_to_test}) " + "="*20)
    
    logger.info(f"--- Teste 1: Validação (Heurística) do Modelo Gemini {model_to_test} ---")
    if not validate_gemini_model(model_to_test):
        logger.warning(f"Modelo de teste '{model_to_test}' não passou na validação heurística.")
    else:
        logger.info(f"Modelo de teste '{model_to_test}' parece válido.")

    logger.info(f"--- Teste 2: Dataset e Pastas (Primeiras {num_properties_to_test} props) ---")
    if not DATASET_PATH.is_file():
        logger.error(f"Dataset não encontrado {DATASET_PATH}")
        return
    
    try:
        df_test = pd.read_csv(DATASET_PATH, nrows=num_properties_to_test)
        for idx, row_test in df_test.iterrows():
            prop_id_search = str(row_test.get('id_anuncio'))
            prop_type_search = str(row_test.get('tipo_imovel'))
            test_dir = find_property_image_dir(IMAGES_BASE_DIR, prop_id_search, prop_type_search)
            if test_dir:
                logger.info(f"Prop (busca {prop_id_search}, idx {idx}): Dir {test_dir}")
                try:
                    imgs_in_dir = await asyncio.to_thread(get_images_from_directory, str(test_dir))
                    logger.info(f"  Encontradas {len(imgs_in_dir)} imagens. Primeira: {imgs_in_dir[0] if imgs_in_dir else 'N/A'}")
                except Exception as e_get_imgs:
                    logger.error(f"  Erro ao listar imagens em {test_dir}: {e_get_imgs}")
            else:
                logger.warning(f"Prop (busca {prop_id_search}, idx {idx}): Dir NÃO encontrado.")
    except Exception as e:
        logger.error(f"Erro teste dataset/pastas: {e}")

    logger.info(f"--- Teste 3: Processamento Assíncrono de Uma Propriedade com {model_to_test} @ Temp {temp_to_test} ---")
    first_valid_dir = None
    try:
        df_s_test = pd.read_csv(DATASET_PATH, nrows=min(5, len(pd.read_csv(DATASET_PATH))))
        for idx_s, row_s in df_s_test.iterrows():
            prop_id_s = str(row_s.get('id_anuncio'))
            prop_type_s = str(row_s.get('tipo_imovel'))
            dir_p = find_property_image_dir(IMAGES_BASE_DIR, prop_id_s, prop_type_s)
            if dir_p:
                try:
                    imgs = await asyncio.to_thread(get_images_from_directory, str(dir_p))
                    if imgs:
                        first_valid_dir = dir_p
                        break
                except:
                    pass
        
        if first_valid_dir:
            logger.info(f"Testando process_property_with_gemini_async com: {first_valid_dir}, Modelo: {model_to_test}, Temp: {temp_to_test}")
            res_s = await process_property_with_gemini_async(first_valid_dir, model_to_test, temp_to_test)
            logger.info(f"Resultado (API: {res_s.get('success_api_call')}, JSON Parse: {res_s.get('success_json_parsing')}):")
            logger.info(f"  JSON Obj (preview): {str(res_s.get('parsed_json_object'))[:200] if res_s.get('parsed_json_object') else 'N/A'}")
            logger.info(f"  API Usage: {res_s.get('api_usage_metadata')}")
            if res_s.get('json_parsing_error'):
                logger.warning(f"  Erro Parsing JSON: {res_s.get('json_parsing_error')}")
            if res_s.get('error_message_processing'):
                logger.error(f"  Erro Processamento: {res_s.get('error_message_processing')}")
        else:
            logger.error("Nenhum dir válido com imgs para teste de processamento Gemini.")
    except Exception as e:
        logger.error(f"Erro teste processamento único Gemini: {e}")
    
    logger.info("="*20 + " TESTES DE DIAGNÓSTICO GEMINI ASSÍNCRONOS CONCLUÍDOS " + "="*20)

# --- Função principal do script ---
if __name__ == "__main__":
    try:
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)

        logger.info(f"Iniciando script ASSÍNCRONO de GERAÇÃO MULTI-CONFIGURAÇÃO para Modelos GEMINI.")
        logger.info(f"Modelos configurados: {MODEL_NAMES_TO_TEST}")
        logger.info(f"Temperaturas configuradas: {TEMPERATURES_TO_TEST}")
        logger.info(f"Workers: {MAX_WORKERS}")
        
        if "--test" in sys.argv or "-t" in sys.argv:
            test_model = MODEL_NAMES_TO_TEST[0].strip() if MODEL_NAMES_TO_TEST else "gemini-2.5-flash"
            test_temp = TEMPERATURES_TO_TEST[0] if TEMPERATURES_TO_TEST else 0.1
            
            logger.info(f"Executando testes de diagnóstico ASSÍNCRONOS para: Modelo Gemini={test_model}, Temperatura={test_temp}")
            asyncio.run(run_diagnostic_tests_async(model_to_test=test_model, temp_to_test=test_temp, num_properties_to_test=1))
            
            user_response = input("Testes de diagnóstico concluídos. Executar geração multi-configuração completa com Gemini? (S/n): ").strip().lower()
            if not (user_response == "" or user_response.startswith("s")):
                logger.info("Geração multi-configuração Gemini cancelada pelo usuário.")
                sys.exit(0)
        
        logger.info("Iniciando geração ASSÍNCRONA multi-configuração completa com Gemini...")
        asyncio.run(run_multiconfig_gemini_generation_async())
        logger.info("Geração ASSÍNCRONA multi-configuração Gemini finalizada com sucesso!")
        
    except SystemExit:
        pass
    except KeyboardInterrupt:
        logger.warning("Interrompido (KeyboardInterrupt).")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Erro fatal na geração multi-configuração Gemini: {e}"); sys.exit(1)