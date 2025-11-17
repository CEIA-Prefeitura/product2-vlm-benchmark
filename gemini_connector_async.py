'''
Script ASSÍNCRONO para a conexão na Gemini API com suporte a múltiplos workers
'''

import os
import base64
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from loguru import logger

# Langchain e Google specific imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:
    logger.critical("Langchain Google GenAI não instalado. Execute: pip install langchain-google-genai")
    raise

# --- Configuração Inicial e Chave API ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.critical("A chave da API do Google (GOOGLE_API_KEY) não está definida nas variáveis de ambiente ou no arquivo .env.")
    raise ValueError("GOOGLE_API_KEY não definida.")

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
MAX_CONCURRENT_REQUESTS = int(os.getenv("GEMINI_MAX_WORKERS", "5"))  # Número de workers simultâneos

# Configuração do logger para este módulo
logger.add("logs/gemini_connector_async.log", rotation="1 MB", retention="7 days", level="DEBUG", encoding="utf-8")

# --- Semáforo global para controlar concorrência ---
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# --- Funções Auxiliares Internas ---
def _encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """Codifica bytes de uma imagem em uma string base64."""
    return base64.b64encode(image_bytes).decode("utf-8")

def _read_image_bytes_from_path(image_path: str) -> Optional[bytes]:
    """Lê uma imagem de um caminho e retorna seus bytes."""
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError:
        logger.error(f"Arquivo de imagem não encontrado: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Erro ao ler arquivo de imagem {image_path}: {e}")
        return None

def _truncate_for_logging(data: Any, max_len: int = 100) -> str:
    """Trunca dados para logging, útil para strings longas como base64."""
    s = str(data)
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s

# --- Funções Principais do Conector (Assíncronas) ---

def list_available_gemini_models(llm_client: Optional[ChatGoogleGenerativeAI] = None) -> List[str]:
    """Lista modelos Gemini conhecidos."""
    known_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite"
    ]
    logger.info(f"Retornando lista estática de modelos Gemini conhecidos: {known_models}")
    return known_models

def validate_gemini_model(model_name: str, llm_client: Optional[ChatGoogleGenerativeAI] = None) -> bool:
    """Valida se um nome de modelo Gemini é conhecido."""
    if "gemini" in model_name.lower():
        logger.debug(f"Nome do modelo '{model_name}' parece ser um modelo Gemini válido (verificação heurística).")
        return True
    logger.warning(f"Nome do modelo '{model_name}' não parece ser um modelo Gemini conhecido.")
    return False

async def gemini_generate_async(
    prompt_parts: List[Union[str, Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Versão ASSÍNCRONA da função de geração Gemini.
    Usa semáforo para controlar o número de requisições simultâneas.
    """
    async with _semaphore:  # Controla concorrência
        return await _gemini_generate_internal(
            prompt_parts=prompt_parts,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs
        )

async def _gemini_generate_internal(
    prompt_parts: List[Union[str, Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Implementação interna assíncrona da chamada à API Gemini."""
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs
    )

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
        logger.debug(f"System Prompt (primeiros 100 chars): {_truncate_for_logging(system_prompt)}")

    # Construir o HumanMessage com todas as partes (texto e imagens)
    human_content = []
    for part in prompt_parts:
        if isinstance(part, str):
            human_content.append({"type": "text", "text": part})
            logger.debug(f"Human Prompt Text Part (primeiros 100 chars): {_truncate_for_logging(part)}")
        elif isinstance(part, bytes):
            base64_image = _encode_image_bytes_to_base64(part)
            human_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            logger.debug(f"Human Prompt Image Part (bytes): base64 (primeiros 30): {base64_image[:30]}...")
        elif isinstance(part, dict) and part.get("type") == "image_path":
            # Ler imagem de forma assíncrona
            image_bytes = await asyncio.to_thread(_read_image_bytes_from_path, part["image_path"])
            if image_bytes:
                base64_image = _encode_image_bytes_to_base64(image_bytes)
                mime_type = "image/jpeg"
                ext = os.path.splitext(part["image_path"])[1].lower()
                if ext == ".png": mime_type = "image/png"
                elif ext == ".gif": mime_type = "image/gif"
                elif ext == ".webp": mime_type = "image/webp"

                human_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })
                logger.debug(f"Human Prompt Image Part (path: {part['image_path']}): base64 (primeiros 30): {base64_image[:30]}...")
            else:
                logger.error(f"Não foi possível ler a imagem do caminho: {part['image_path']}")
        elif isinstance(part, dict) and part.get("type") == "image_url":
            human_content.append(part)
            logger.debug(f"Human Prompt Image Part (url): {_truncate_for_logging(part['image_url']['url'])}")
        else:
            logger.warning(f"Tipo de parte de prompt desconhecido ou malformado: {part}")

    if not human_content:
        logger.error("Nenhum conteúdo válido (texto ou imagem) fornecido para o HumanMessage.")
        return {"error": "Conteúdo do prompt humano vazio.", "model_name": model_name}
        
    messages.append(HumanMessage(content=human_content))

    start_time = time.time()
    try:
        logger.info(f"Enviando requisição para Gemini model: {model_name}")
        
        # Executar a chamada síncrona do LLM em uma thread separada
        ai_message: AIMessage = await asyncio.to_thread(llm.invoke, messages)
        response_content = ai_message.content
        
        usage_metadata = ai_message.response_metadata.get("usage_metadata") or \
                         ai_message.response_metadata.get("token_usage")

        response_time_s = time.time() - start_time
        logger.success(f"Resposta recebida de Gemini ({model_name}) em {response_time_s:.2f}s.")
        logger.debug(f"Resposta crua do Gemini: {_truncate_for_logging(response_content, 500)}")

        return {
            "response": response_content,
            "model_name": model_name,
            "usage_metadata": usage_metadata,
            "response_time_s": response_time_s,
            "error": None
        }
    except Exception as e:
        response_time_s = time.time() - start_time
        logger.exception(f"Erro durante a chamada à API Gemini ({model_name}) após {response_time_s:.2f}s: {e}")
        return {
            "error": str(e),
            "model_name": model_name,
            "response_time_s": response_time_s,
            "response": None,
            "usage_metadata": None
        }

# Manter versão síncrona para compatibilidade
def gemini_generate(
    prompt_parts: List[Union[str, Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Versão síncrona (wrapper) para compatibilidade com código existente.
    Internamente usa a versão assíncrona.
    """
    return asyncio.run(gemini_generate_async(
        prompt_parts=prompt_parts,
        system_prompt=system_prompt,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs
    ))

# --- Bloco de Teste ---
if __name__ == "__main__":
    logger.info("Executando testes do gemini_connector_async...")

    async def run_tests():
        # Teste 1: Listar modelos
        print("\n--- Teste 1: Modelos Gemini Conhecidos ---")
        print(list_available_gemini_models())

        # Teste 2: Validar um modelo
        test_model = "gemini-2.5-pro"
        print(f"\n--- Teste 2: Validar Modelo ({test_model}) ---")
        is_valid = validate_gemini_model(test_model)
        print(f"O modelo '{test_model}' passou na validação heurística? {is_valid}")

        # Teste 3: Gerar texto simples (múltiplas requisições paralelas)
        print("\n--- Teste 3: Geração de Texto Simples (3 requisições paralelas) ---")
        tasks = []
        for i in range(3):
            prompt = [f"Conte-me um fato interessante sobre o número {i+1}."]
            tasks.append(gemini_generate_async(prompt_parts=prompt, model_name=test_model, temperature=0.7))
        
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            if result.get("error"):
                print(f"  Requisição {i+1} - Erro: {result['error']}")
            else:
                print(f"  Requisição {i+1} - Resposta: {result.get('response')[:100]}...")
                print(f"  Tempo: {result.get('response_time_s'):.2f}s")

    asyncio.run(run_tests())
    logger.info("Testes do gemini_connector_async concluídos.")