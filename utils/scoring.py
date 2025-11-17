# utils/scoring.py

from typing import Dict, Any, Union
import pandas as pd
from loguru import logger
import unicodedata
import re

def normalize_text_for_scoring(text: Any) -> str:
    """
    Normaliza o texto para correspondência com as chaves de pontuação:
    - Converte para string
    - Converte para minúsculas
    - Remove acentos
    - Remove espaços extras no início/fim
    - Substitui múltiplos espaços por um único espaço
    """
    if pd.isna(text) or text is None:
        return "" # Retorna string vazia para valores nulos/NaN
    
    # Converte para string e minúsculas
    s = str(text).lower().strip()
    
    # Remove acentos
    nfkd_form = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    
    # Substitui múltiplos espaços por um único
    s = re.sub(r'\s+', ' ', s)
    
    return s

# --- Tabela BIC de Pontuações ---
# Estrutura: { 'nome_da_coluna_no_csv': { 'opcao_normalizada': pontos, ... }, ... }
# Todos os nomes de colunas e opções são normalizados (minúsculos, sem acentos).

PONTUACAO_BIC = {
    "estrutura": {
        "alvenaria": 3,
        "concreto": 5,
        "mista": 5,
        "madeira tratada": 3,
        "metalica": 5,
        "adobe/taipa/rudimentar": 1,
        "adobe": 1, "taipa": 1, "rudimentar": 1, # Alias
        "na": 0 # Default para "não aplicável" ou não encontrado
    },
    "esquadrias": {
        "ferro": 2,
        "aluminio": 4,
        "madeira": 3,
        "rustica": 1,
        "especial": 5,
        "sem": 0,
        "na": 0
    },
    "piso": {
        "ceramica": 4,
        "cimento": 3,
        "taco": 2,
        "tijolo": 1,
        "terra": 0,
        "especial/porcelanato": 5,
        "especial": 5, "porcelanato": 5, # Alias
        "na": 0
    },
    "forro": {
        "laje": 4,
        "madeira": 3,
        "gesso simples/pvc": 2,
        "gesso simples": 2, "pvc": 2, # Alias
        "especial": 5,
        "sem": 0,
        "na": 0
    },
    "instalacao_eletrica": {
        "embutida": 5,
        "semi embutida": 3,
        "externa": 1,
        "sem": 0,
        "na": 0
    },
    "instalacao_sanitaria": {
        "interna": 3,
        "completa": 4,
        "mais de uma": 5,
        "externa": 2,
        "sem": 0,
        "na": 0
    },
    "revestimento_interno": {
        "reboco": 2,
        "massa": 3,
        "material ceramico": 4,
        "especial": 5,
        "sem": 0,
        "na": 0
    },
    "acabamento_interno": {
        "pintura lavavel": 3,
        "pintura simples": 2,
        "caiacao": 1,
        "especial": 5,
        "sem": 0,
        "na": 0
    },
    "revestimento_externo": {
        "reboco": 1,
        "massa": 2,
        "material ceramico": 2,
        "especial": 4,
        "sem": 0,
        "na": 0
    },
    "acabamento_externo": {
        "pintura lavavel": 2,
        "pintura simples": 1,
        "caiacao": 1,
        "especial": 5,
        "sem": 0,
        "na": 0
    },
    "cobertura": {
        "telha de barro": 4,
        "fibrocimento": 3,
        "aluminio": 4,
        "zinco": 4,
        "laje": 4,
        "palha": 1,
        "especial": 5,
        "sem": 0,
        "na": 0
    },
    # Pontuações para Benfeitorias. As chaves devem corresponder aos nomes das colunas no CSV.
    "benfeitorias": {
        "piscina": 1,
        "sauna": 1,
        "home cinema (area comum)": 1,
        "churrasqueira coletiva": 1,
        "churrasqueira privativa": 2,
        "quadra poliesportiva": 1,
        "quadra de tenis": 2,
        "playground / brinquedoteca": 1,
        "elevador": 1,
        "energia solar": 1,
        "academia de ginastica": 1,
        "salao de festas": 1,
        "espaco gourmet": 2,
        "gerador": 1,
        "heliponto": 3,
        "escaninhos": 1,
        "mais de dois box de garagem": 1,
        "laje tecnica": 1,
        "sala reuniao / coworking": 1,
        "isolamento acustico": 1,
        "rede frigorigena": 1,
        "mais de uma suite": 1,
        "lavabo": 1
    }
}

# Dicionário para as faixas de pontuação por tipo de imóvel
PADRAO_POR_PONTUACAO = {
    "condominios verticais": [
        (40, 'E'), (45, 'D'), (51, 'C'), (59, 'B'), (float('inf'), 'A')
    ],
    "condominios horizontais": [
        (36, 'E'), (41, 'D'), (45, 'C'), (50, 'B'), (float('inf'), 'A')
    ],
    "demais construcoes": [
        (30, 'E'), (38, 'D'), (42, 'C'), (46, 'B'), (float('inf'), 'A')
    ]
}

# Lista de colunas que representam as características principais (não benfeitorias)
CARACTERISTICAS_PRINCIPAIS_COLS = [
    "estrutura", "esquadrias", "piso", "forro", "instalacao_eletrica", 
    "instalacao_sanitaria", "revestimento_interno", "acabamento_interno", 
    "revestimento_externo", "acabamento_externo", "cobertura"
]

# Lista de colunas que representam benfeitorias
# Nomes devem ser normalizados para corresponder às chaves em PONTUACAO_BIC["benfeitorias"]
BENFEITORIAS_COLS = [normalize_text_for_scoring(col) for col in PONTUACAO_BIC["benfeitorias"].keys()]


def calcular_pontuacao_bic(
    imovel_data: Union[pd.Series, Dict[str, Any]]
) -> Dict[str, Union[int, float]]:
    """
    Calcula a pontuação BIC total para um imóvel com base em suas classificações.

    Args:
        imovel_data: Uma linha de um DataFrame do Pandas (pd.Series) ou um dicionário
                     contendo as classificações do imóvel. Os nomes das chaves/colunas
                     devem corresponder aos fornecidos na documentação do projeto
                     (ex: 'estrutura', 'piso', 'Piscina', 'Sauna', etc.).

    Returns:
        Um dicionário contendo:
        - 'pontuacao_caracteristicas': A soma dos pontos das características principais.
        - 'pontuacao_benfeitorias': A soma dos pontos das benfeitorias.
        - 'pontuacao_total': A soma total das pontuações.
    """
    pontuacao_caracteristicas = 0
    pontuacao_benfeitorias = 0

    # Normalizar as chaves (nomes das colunas) dos dados do imóvel uma vez para correspondência case-insensitive e sem acentos
    imovel_data_norm_keys = {normalize_text_for_scoring(k): v for k, v in imovel_data.items()}

    # Calcular pontuação das características principais
    for col_name in CARACTERISTICAS_PRINCIPAIS_COLS:
        classificacao_raw = imovel_data_norm_keys.get(col_name)
        classificacao_norm = normalize_text_for_scoring(classificacao_raw)
        pontos = PONTUACAO_BIC.get(col_name, {}).get(classificacao_norm, 0)
        pontuacao_caracteristicas += pontos
        logger.trace(f"Característica '{col_name}': Classificação '{classificacao_norm}' -> Pontos: {pontos}")

    # --- BLOCO MODIFICADO PARA BENFEITORIAS ---
    # Calcular pontuação das benfeitorias, verificando o valor 'sim'
    for col_name_original in PONTUACAO_BIC["benfeitorias"].keys():
        # A chave em PONTUACAO_BIC["benfeitorias"] já é normalizada, ex: "piscina"
        # Precisamos encontrar a coluna correspondente nos dados do imóvel, que pode não estar normalizada.
        # Por isso usamos imovel_data_norm_keys
        
        # A chave já está normalizada, pois BENFEITORIAS_COLS foi criada com normalização
        # Vamos iterar sobre as chaves normalizadas para garantir a correspondência
        col_name_norm = normalize_text_for_scoring(col_name_original)
        
        valor_benfeitoria_raw = imovel_data_norm_keys.get(col_name_norm)
        
        # Normaliza o valor da célula (ex: "Sim", "sim", " SIM ") -> "sim"
        valor_norm = normalize_text_for_scoring(valor_benfeitoria_raw)
        
        # Verifica se o valor normalizado é exatamente "sim"
        if valor_norm == 'sim':
            # Se for "sim", busca a pontuação correspondente
            pontos = PONTUACAO_BIC.get("benfeitorias", {}).get(col_name_norm, 0)
            pontuacao_benfeitorias += pontos
            logger.trace(f"Benfeitoria '{col_name_norm}': Presente ('sim') -> Pontos: {pontos}")
    # --- FIM DO BLOCO MODIFICADO ---

    # Pontuação total
    pontuacao_total = pontuacao_caracteristicas + pontuacao_benfeitorias

    return {
        "pontuacao_caracteristicas": pontuacao_caracteristicas,
        "pontuacao_benfeitorias": pontuacao_benfeitorias,
        "pontuacao_total": pontuacao_total
    }
    
def determinar_padrao(pontuacao_total: int, tipo_imovel: str) -> str:
    """
    Determina o padrão (A, B, C, D, E) com base na pontuação total e no tipo de imóvel.

    Args:
        pontuacao_total: A pontuação BIC total calculada para o imóvel.
        tipo_imovel: A string do tipo de imóvel (ex: 'Condominios Verticais').

    Returns:
        A letra do padrão correspondente (ex: 'A', 'B', 'C', 'D', 'E') ou 'Indefinido'.
    """
    if tipo_imovel is None or pd.isna(tipo_imovel):
        logger.warning("Tipo de imóvel não fornecido, não foi possível determinar o padrão.")
        return "Indefinido"

    tipo_imovel_norm = normalize_text_for_scoring(tipo_imovel)
    
    # Encontrar a chave correta no dicionário, permitindo variações
    faixas_pontuacao = None
    for key in PADRAO_POR_PONTUACAO:
        if key in tipo_imovel_norm:
            faixas_pontuacao = PADRAO_POR_PONTUACAO[key]
            break
            
    if faixas_pontuacao is None:
        logger.warning(f"Tipo de imóvel normalizado '{tipo_imovel_norm}' não corresponde a nenhuma chave em PADRAO_POR_PONTUACAO. Chaves: {list(PADRAO_POR_PONTUACAO.keys())}")
        return "Tipo Imovel Invalido"

    # Itera sobre as faixas (limite_superior, padrao)
    # A lógica é: se pontuacao <= limite_superior, então é aquele padrão.
    if pontuacao_total >= 1: # A pontuação deve ser pelo menos 1
        for limite_superior, padrao in faixas_pontuacao:
            if pontuacao_total <= limite_superior:
                return padrao
    
    # Se a pontuação for 0 ou negativa (não deveria acontecer), ou outra condição não coberta
    logger.warning(f"Pontuação total ({pontuacao_total}) para o tipo '{tipo_imovel_norm}' não se encaixou em nenhuma faixa. Verifique os dados.")
    return "Fora da Faixa"

def determinar_padrao_caracteristicas(pontuacao_caracteristicas: int, tipo_imovel: str) -> str:
    """
    Determina o padrão (A, B, C, D, E) com base na pontuação total e no tipo de imóvel.

    Args:
        pontuacao_total: A pontuação BIC total calculada para o imóvel.
        tipo_imovel: A string do tipo de imóvel (ex: 'Condominios Verticais').

    Returns:
        A letra do padrão correspondente (ex: 'A', 'B', 'C', 'D', 'E') ou 'Indefinido'.
    """
    if tipo_imovel is None or pd.isna(tipo_imovel):
        logger.warning("Tipo de imóvel não fornecido, não foi possível determinar o padrão.")
        return "Indefinido"

    tipo_imovel_norm = normalize_text_for_scoring(tipo_imovel)
    
    # Encontrar a chave correta no dicionário, permitindo variações
    faixas_pontuacao = None
    for key in PADRAO_POR_PONTUACAO:
        if key in tipo_imovel_norm:
            faixas_pontuacao = PADRAO_POR_PONTUACAO[key]
            break
            
    if faixas_pontuacao is None:
        logger.warning(f"Tipo de imóvel normalizado '{tipo_imovel_norm}' não corresponde a nenhuma chave em PADRAO_POR_PONTUACAO. Chaves: {list(PADRAO_POR_PONTUACAO.keys())}")
        return "Tipo Imovel Invalido"

    # Itera sobre as faixas (limite_superior, padrao)
    # A lógica é: se pontuacao <= limite_superior, então é aquele padrão.
    if pontuacao_caracteristicas >= 1: # A pontuação deve ser pelo menos 1
        for limite_superior, padrao_caracteristicas in faixas_pontuacao:
            if pontuacao_caracteristicas <= limite_superior:
                return padrao_caracteristicas
    
    # Se a pontuação for 0 ou negativa (não deveria acontecer), ou outra condição não coberta
    logger.warning(f"Pontuação carcateristicas ({pontuacao_caracteristicas}) para o tipo '{tipo_imovel_norm}' não se encaixou em nenhuma faixa. Verifique os dados.")
    return "Fora da Faixa"

def calcular_pontuacao_e_padrao(
    imovel_data: Union[pd.Series, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Função wrapper que calcula a pontuação BIC e determina o padrão do imóvel.

    Args:
        imovel_data: Uma linha de um DataFrame do Pandas (pd.Series) ou um dicionário
                     contendo as classificações do imóvel. Deve incluir a coluna 'tipo_imovel'.

    Returns:
        Um dicionário contendo as pontuações e o padrão final:
        - 'pontuacao_caracteristicas'
        - 'pontuacao_benfeitorias'
        - 'pontuacao_total'
        - 'padrao'
    """
    # 1. Calcular a pontuação
    pontuacoes = calcular_pontuacao_bic(imovel_data)
    pontuacao_total = pontuacoes.get("pontuacao_total", 0)
    pontuacao_caracteristicas = pontuacoes.get("pontuacao_caracteristicas", 0)

    # 2. Obter o tipo de imóvel dos dados
    # A chave/nome da coluna deve ser 'tipo_imovel'
    tipo_imovel = imovel_data.get("tipo_padrao", imovel_data.get("tipo_de_imovel"))
    if tipo_imovel is None:
        logger.warning(f"Coluna 'tipo_padrao' não encontrada nos dados. Dados disponíveis: {list(imovel_data.keys())}")

    # 3. Determinar o padrão
    padrao = determinar_padrao(pontuacao_total, tipo_imovel)
    padrao_caracteristicas = determinar_padrao_caracteristicas(pontuacao_caracteristicas, tipo_imovel)
    
    # 4. Combinar os resultados
    resultado_final = {
        **pontuacoes, # Inclui pontuacao_caracteristicas, pontuacao_benfeitorias, pontuacao_total
        "padrao_caracteristicas": padrao_caracteristicas,
        "padrao": padrao
    }
    
    return resultado_final