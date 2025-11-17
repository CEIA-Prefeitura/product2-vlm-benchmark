# analyze_scores.py

import pandas as pd
from pathlib import Path
from loguru import logger

# Importar a nova função wrapper
from scoring import calcular_pontuacao_e_padrao

# Configuração
DATASET_CSV_PATH = "./data/CSVs/91_casas_terreas_anotadas_bic.csv" # Seu CSV com as classificações
OUTPUT_CSV_PATH = "./resultados_sumarizados/91_casas_terreas_anotadas_com_bic.csv" # Novo nome de arquivo

def main():
    logger.info("Iniciando cálculo de pontuações BIC e determinação de Padrão...")

    try:
        df = pd.read_csv(DATASET_CSV_PATH)
        logger.info(f"Dataset carregado com {len(df)} registros de '{DATASET_CSV_PATH}'.")
    except FileNotFoundError:
        logger.error(f"Arquivo de dataset não encontrado em: {DATASET_CSV_PATH}")
        return
    except Exception as e:
        logger.error(f"Erro ao carregar o dataset: {e}")
        return
        
    # Verificar se a coluna 'tipo_padrao' existe
    if 'tipo_padrao' not in [col.lower().replace(' ', '_') for col in df.columns]:
         logger.warning("Coluna 'tipo_padrao' parece não existir no CSV. A determinação do padrão pode falhar.")
         # Pode ser necessário normalizar os nomes das colunas do DataFrame aqui se eles não forem consistentes.
         # df.columns = [normalize_text_for_scoring(col) for col in df.columns]

    logger.info("Calculando pontuações e padrão para cada imóvel no dataset...")
    # Chamar a nova função wrapper que retorna tudo
    results_series = df.apply(calcular_pontuacao_e_padrao, axis=1)

    # Converter a Series de dicionários em um DataFrame
    df_results = pd.json_normalize(results_series)

    # Juntar o DataFrame original com o novo DataFrame de resultados
    df_final = pd.concat([df, df_results], axis=1)

    # Salvar o novo DataFrame em um novo arquivo CSV
    try:
        output_path = Path(OUTPUT_CSV_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        logger.info(f"Arquivo CSV com pontuações e padrão salvo com sucesso em: {output_path}")
        
        # Mostrar as primeiras linhas do resultado final, incluindo a nova coluna 'padrao'
        print("\n--- Amostra do CSV Final com Pontuações e Padrão ---")
        print(df_final[['url', 'tipo_padrao', 'pontuacao_total', 'padrao']].head())
        print("\n-----------------------------------------------------")

    except Exception as e:
        logger.error(f"Erro ao salvar o CSV com as pontuações e padrão: {e}")

if __name__ == "__main__":
    main()