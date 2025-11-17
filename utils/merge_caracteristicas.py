"""
Script para dar merge em todos os 11 jsons de caracteríscticas gerados separadamente,
gerando apenas 1 json com todas as 11 carcateríticas classificadas por imóvel.

Para avaliar basta utilizar o avaliador, normalmente.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Mapeamento das características e seus nomes nas pastas
CARACTERISTICAS = {
    'estrutura': 'Estrutura',
    'esquadrias': 'Esquadrias',
    'piso': 'Piso',
    'forro': 'Forro',
    'instalacao_eletrica': 'Instalação Elétrica',
    'instalacao_sanitaria': 'Instalação Sanitária',
    'revestimento_interno': 'Revestimento Interno',
    'acabamento_interno': 'Acabamento Interno',
    'revestimento_externo': 'Revestimento Externo',
    'acabamento_externo': 'Acabamento Externo',
    'cobertura': 'Cobertura'
}

def carregar_json_caracteristica(base_dir: Path, caracteristica: str) -> Dict:
    """Carrega o JSON de uma característica específica."""
    json_path = base_dir / caracteristica / f"{caracteristica}_results.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extrair_metadata_global(jsons_caracteristicas: Dict[str, Dict]) -> Dict:
    """Extrai metadata global do primeiro JSON disponível."""
    primeira_carac = next(iter(jsons_caracteristicas.values()))
    metadata = primeira_carac.get('metadata', {})
    
    return {
        'modelo_usado': metadata.get('modelo_usado', 'N/A'),
        'temperatura_usada': metadata.get('temperatura_usada', 0.5),
        'total_propriedades_dataset': metadata.get('total_propriedades', 0),
        'dataset_usado': metadata.get('dataset_usado', 'N/A')
    }

def merge_propriedade(property_id: int, jsons_caracteristicas: Dict[str, Dict]) -> Dict:
    """Faz o merge de todas as características para uma propriedade específica."""
    # Buscar dados da propriedade em cada JSON de característica
    propriedade_data = {}
    
    for carac_key, carac_nome in CARACTERISTICAS.items():
        json_data = jsons_caracteristicas.get(carac_key, {})
        resultados = json_data.get('resultados', [])
        
        # Encontrar a propriedade específica
        for resultado in resultados:
            if resultado.get('property_id') == property_id:
                propriedade_data[carac_key] = resultado
                break
    
    if not propriedade_data:
        return None
    
    # Pegar dados base da primeira característica disponível
    primeira_carac = next(iter(propriedade_data.values()))
    
    # Construir resposta_modelo_parsed_dict
    resposta_parsed = {}
    
    for carac_key, carac_nome in CARACTERISTICAS.items():
        if carac_key in propriedade_data:
            resultado = propriedade_data[carac_key]
            parsed = resultado.get('resposta_parsed', {})
            valor = parsed.get('valor_classificado', 'NA')
            resposta_parsed[carac_nome] = valor
        else:
            resposta_parsed[carac_nome] = 'NA'
    
    # Adicionar Benfeitorias e Tipo de Imóvel (se existirem)
    # Normalmente viriam de uma característica específica ou do JSON geral
    resposta_parsed['Benfeitorias'] = []
    resposta_parsed['Tipo de Imóvel'] = 'residencial'  # Default
    
    # Construir objeto final da propriedade
    propriedade_merged = {
        'property_id': property_id,
        'dataset_idx': primeira_carac.get('dataset_idx', 0),
        'modelo': primeira_carac.get('modelo', 'N/A'),
        'temperatura_usada': primeira_carac.get('temperatura', 0.5),
        'search_term_folder': primeira_carac.get('search_term_folder', ''),
        'type_folder': primeira_carac.get('type_folder', ''),
        'property_directory_processed': primeira_carac.get('property_directory', ''),
        'num_imagens_encontradas': primeira_carac.get('num_imagens_encontradas', 0),
        'num_imagens_enviadas_api': primeira_carac.get('num_imagens_enviadas', 0),
        'api_usage_metadata': primeira_carac.get('api_usage_metadata'),
        'resposta_modelo_parsed_dict': resposta_parsed,
        'status_api_call': 'sucesso',  # Validar se todas características foram sucesso
        'status_json_parsing': 'sucesso'
    }
    
    # Validar status geral
    todos_sucesso = all(
        propriedade_data.get(carac, {}).get('status', '') == 'sucesso'
        for carac in CARACTERISTICAS.keys()
        if carac in propriedade_data
    )
    
    if not todos_sucesso:
        propriedade_merged['status_api_call'] = 'parcial'
    
    return propriedade_merged

def merge_todos_jsons(base_dir: str, output_file: str):
    """Função principal para fazer o merge de todos os JSONs."""
    base_path = Path(base_dir)
    timestamp_inicio = datetime.now().astimezone()
    
    print(f"Iniciando merge de JSONs em: {base_path}")
    
    # Carregar todos os JSONs de características
    jsons_caracteristicas = {}
    for carac_key in CARACTERISTICAS.keys():
        try:
            print(f"Carregando: {carac_key}...")
            jsons_caracteristicas[carac_key] = carregar_json_caracteristica(base_path, carac_key)
        except FileNotFoundError as e:
            print(f"AVISO: {e}")
            continue
    
    if not jsons_caracteristicas:
        raise ValueError("Nenhum JSON de característica foi carregado!")
    
    print(f"\nTotal de características carregadas: {len(jsons_caracteristicas)}")
    
    # Extrair metadata global
    metadata_global = extrair_metadata_global(jsons_caracteristicas)
    
    # Obter lista de property_ids
    primeira_carac_key = next(iter(jsons_caracteristicas.keys()))
    resultados = jsons_caracteristicas[primeira_carac_key].get('resultados', [])
    property_ids = [r.get('property_id') for r in resultados]
    
    print(f"Total de propriedades encontradas: {len(property_ids)}")
    
    # Fazer merge de cada propriedade
    resultados_merged = []
    for i, prop_id in enumerate(property_ids, 1):
        print(f"Processando propriedade {i}/{len(property_ids)} (ID: {prop_id})...", end='\r')
        propriedade = merge_propriedade(prop_id, jsons_caracteristicas)
        if propriedade:
            resultados_merged.append(propriedade)
    
    print(f"\nTotal de propriedades processadas com sucesso: {len(resultados_merged)}")
    
    # Calcular timestamps
    timestamp_fim = datetime.now().astimezone()
    duracao_total = (timestamp_fim - timestamp_inicio).total_seconds()
    
    # Construir JSON final
    output_data = {
        'metadata_geracao_run': {
            'timestamp_geracao_inicial_run': timestamp_inicio.isoformat(),
            'timestamp_geracao_final_run': timestamp_fim.isoformat(),
            'timestamp_total_run': duracao_total,
            'modelo_usado': metadata_global['modelo_usado'],
            'temperatura_usada': metadata_global['temperatura_usada'],
            'total_propriedades_dataset': metadata_global['total_propriedades_dataset'],
            'total_itens_na_saida_run': len(resultados_merged),
            'dataset_usado': metadata_global['dataset_usado'],
            'tipo_prompt_usado': 'analise_por_caracteristica_especializada',
            'caracteristicas_processadas': list(CARACTERISTICAS.values())
        },
        'resultados_gerados_run': resultados_merged
    }
    
    # Salvar JSON final
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Merge concluído! Arquivo salvo em: {output_path}")
    print(f"   - Características processadas: {len(jsons_caracteristicas)}/11")
    print(f"   - Propriedades processadas: {len(resultados_merged)}")
    print(f"   - Duração: {duracao_total:.2f}s")

if __name__ == "__main__":
    # Configurações
    BASE_DIR = "resultados_por_caracteristica"
    OUTPUT_FILE = "resultados_merged/resultados_completos_merged.json"
    
    try:
        merge_todos_jsons(BASE_DIR, OUTPUT_FILE)
    except Exception as e:
        print(f"\n❌ Erro durante o merge: {e}")
        raise