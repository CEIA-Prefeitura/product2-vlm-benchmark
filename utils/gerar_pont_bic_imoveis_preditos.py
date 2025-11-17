import json
import sys
import os
from typing import Dict, List, Any
from pathlib import Path

# Tabela de pontuação BIC
SCORING_TABLE = {
    "Estrutura": {
        "alvenaria": 3,
        "concreto": 5,
        "mista": 5,
        "madeira_tratada": 3,
        "metalica": 5,
        "adobe_taipa_rudimentar": 1
    },
    "Esquadrias": {
        "ferro": 2,
        "aluminio": 4,
        "madeira": 3,
        "rustica": 1,
        "especial": 5,
        "sem": 0
    },
    "Piso": {
        "ceramica": 4,
        "cimento": 3,
        "taco": 2,
        "tijolo": 1,
        "terra": 0,
        "especial_porcelanato": 5
    },
    "Forro": {
        "laje": 4,
        "madeira": 3,
        "gesso_simples_pvc": 2,
        "especial": 5,
        "sem": 0
    },
    "Instalação Elétrica": {
        "embutida": 5,
        "semi_embutida": 3,
        "externa": 1,
        "sem": 0
    },
    "Instalação Sanitária": {
        "interna": 3,
        "completa": 4,
        "mais_de_uma": 5,
        "externa": 2,
        "sem": 0
    },
    "Revestimento Interno": {
        "reboco": 2,
        "massa": 3,
        "material_ceramico": 4,
        "especial": 5,
        "sem": 0
    },
    "Acabamento Interno": {
        "pintura_lavavel": 3,
        "pintura_simples": 2,
        "caiacao": 1,
        "especial": 5,
        "sem": 0
    },
    "Revestimento Externo": {
        "reboco": 1,
        "massa": 2,
        "material_ceramico": 2,
        "especial": 4,
        "sem": 0
    },
    "Acabamento Externo": {
        "pintura_lavavel": 2,
        "pintura_simples": 1,
        "caiacao": 1,
        "especial": 5,
        "sem": 0
    },
    "Cobertura": {
        "telha_de_barro": 4,
        "fibrocimento": 3,
        "aluminio": 4,
        "zinco": 4,
        "laje": 4,
        "palha": 1,
        "especial": 5,
        "sem": 0
    },
    "Benfeitorias": {
        "piscina": 1,
        "sauna": 1,
        "home_cinema_area_comum": 1,
        "churrasqueira_coletiva": 1,
        "churrasqueira_privativa": 2,
        "quadra_poliesportiva": 1,
        "quadra_tenis": 2,
        "playground_brinquedoteca": 1,
        "elevador": 1,
        "energia_solar": 1,
        "academia_ginastica": 1,
        "salao_festas": 1,
        "espaco_gourmet": 2,
        "gerador": 1,
        "heliponto": 3,
        "escaninhos": 1,
        "mais_dois_box_garagem": 1,
        "laje_tecnica": 1,
        "sala_reuniao_coworking": 1,
        "isolamento_acustico": 1,
        "rede_frigorigena": 1,
        "mais_de_uma_suite": 1,
        "lavabo": 1
    }
}

def normalize_key(value: str) -> str:
    """Normaliza string para chave de busca na tabela"""
    return value.lower().strip().replace(" ", "_").replace("/", "_")

def get_score(characteristic: str, value: str) -> int:
    """Obtém a pontuação de uma característica"""
    if characteristic not in SCORING_TABLE:
        return 0
    
    normalized_value = normalize_key(value)
    return SCORING_TABLE[characteristic].get(normalized_value, 0)

def calculate_benfeitoria_score(benfeitorias: List[str]) -> int:
    """Calcula a pontuação total das benfeitorias"""
    total = 0
    for benfeitoria in benfeitorias:
        normalized = normalize_key(benfeitoria)
        total += SCORING_TABLE["Benfeitorias"].get(normalized, 0)
    return total

def find_key_case_insensitive(data: Dict[str, Any], key: str) -> str:
    """Encontra uma chave no dicionário ignorando maiúsculas/minúsculas"""
    key_lower = key.lower().replace(" ", "_")
    for k in data.keys():
        if k.lower().replace(" ", "_") == key_lower:
            return k
    return None

def calculate_characteristics_score(property_data: Dict[str, Any]) -> int:
    """Calcula a pontuação das características (sem benfeitorias)"""
    total = 0
    characteristics = [
        "Estrutura", "Esquadrias", "Piso", "Forro",
        "Instalação Elétrica", "Instalação Sanitária",
        "Revestimento Interno", "Acabamento Interno",
        "Revestimento Externo", "Acabamento Externo", "Cobertura"
    ]
    
    for char in characteristics:
        actual_key = find_key_case_insensitive(property_data, char)
        if actual_key and property_data[actual_key] is not None:
            total += get_score(char, property_data[actual_key])
    
    return total

def add_bic_scores(properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adiciona os scores BIC a cada imóvel"""
    for prop in properties:
        if prop is None:
            continue
            
        if "resposta_modelo_parsed_dict" in prop and prop["resposta_modelo_parsed_dict"] is not None:
            data = prop["resposta_modelo_parsed_dict"]
            
            # Calcula pontuações
            characteristics_score = calculate_characteristics_score(data)
            benfeitorias = data.get("Benfeitorias", []) or []
            benfeitorias_score = calculate_benfeitoria_score(benfeitorias)
            total_score = characteristics_score + benfeitorias_score
            
            # Adiciona campos de pontuação ao dicionário
            prop["pontuacao_bic"] = {
                "pontuacao_caracteristicas": characteristics_score,
                "pontuacao_benfeitorias": benfeitorias_score,
                "pontuacao_total": total_score,
                "detalhe_caracteristicas": get_characteristics_details(data),
                "detalhe_benfeitorias": get_benfeitorias_details(benfeitorias)
            }
        else:
            # Adiciona pontuação com scores zerados se não houver dados
            prop["pontuacao_bic"] = {
                "pontuacao_caracteristicas": 0,
                "pontuacao_benfeitorias": 0,
                "pontuacao_total": 0,
                "detalhe_caracteristicas": {},
                "detalhe_benfeitorias": {}
            }
    
    return properties

def get_characteristics_details(data: Dict[str, Any]) -> Dict[str, int]:
    """Retorna detalhamento de cada característica e sua pontuação"""
    characteristics = [
        "Estrutura", "Esquadrias", "Piso", "Forro",
        "Instalação Elétrica", "Instalação Sanitária",
        "Revestimento Interno", "Acabamento Interno",
        "Revestimento Externo", "Acabamento Externo", "Cobertura"
    ]
    
    details = {}
    for char in characteristics:
        if char in data:
            details[char] = get_score(char, data[char])
    
    return details

def get_benfeitorias_details(benfeitorias: List[str]) -> Dict[str, int]:
    """Retorna detalhamento de cada benfeitoria e sua pontuação"""
    details = {}
    for benfeitoria in benfeitorias:
        normalized = normalize_key(benfeitoria)
        details[benfeitoria] = SCORING_TABLE["Benfeitorias"].get(normalized, 0)
    
    return details

def process_json_file(input_file: str, output_file: str):
    """Processa arquivo JSON adicionando pontuações BIC"""
    try:
        # Lê arquivo de entrada
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Se é um arquivo com lista de resultados
        if isinstance(data, list):
            data = add_bic_scores(data)
        elif isinstance(data, dict) and "resultados_gerados_run" in data:
            if data["resultados_gerados_run"] is not None:
                data["resultados_gerados_run"] = add_bic_scores(data["resultados_gerados_run"])
        
        # Escreve arquivo de saída
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Arquivo processado com sucesso!")
        print(f"  Entrada: {input_file}")
        print(f"  Saída: {output_file}")
        
    except FileNotFoundError:
        print(f"✗ Erro: Arquivo '{input_file}' não encontrado")
    except json.JSONDecodeError:
        print(f"✗ Erro: Arquivo JSON inválido")
    except Exception as e:
        print(f"✗ Erro: {str(e)}")

# Exemplo de uso
if __name__ == "__main__":
    # Validar argumentos da linha de comando
    if len(sys.argv) < 2:
        print("Uso: python codigo.py <caminho_arquivo.json>")
        print("\nExemplo:")
        print("  python utils/codigo.py teste/arquivo.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Gerar nome do arquivo de saída
    path = Path(input_file)
    output_file = path.parent / f"{path.stem}_com_bic{path.suffix}"
    
    # Validar se arquivo existe
    if not os.path.exists(input_file):
        print(f"✗ Erro: Arquivo '{input_file}' não encontrado")
        sys.exit(1)
    
    # Processar arquivo
    process_json_file(input_file, str(output_file))