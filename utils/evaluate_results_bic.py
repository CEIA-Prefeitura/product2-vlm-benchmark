import json
import sys
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_ground_truth(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Carrega dados de ground truth do CSV"""
    ground_truth = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                imovel_id = row['id_anuncio']  # Mantém como string
                ground_truth[imovel_id] = {
                    'pontuacao_caracteristicas': int(row['pontuacao_caracteristicas']),
                    'pontuacao_benfeitorias': int(row['pontuacao_benfeitorias']) if row.get('pontuacao_benfeitorias') else None,
                    'pontuacao_total': int(row['pontuacao_total']) if row.get('pontuacao_total') else None,
                    'padrao_caracteristicas': row['padrao_caracteristicas'],
                    'padrao': row['padrao'],
                    'tipo_imovel': row.get('tipo_imovel', 'HOME')
                }
        print(f"✓ Ground truth carregado: {len(ground_truth)} registros")
        print(f"  Primeiros 5 IDs: {sorted(list(ground_truth.keys()))[:5]}")
        return ground_truth
    except Exception as e:
        print(f"✗ Erro ao carregar ground truth: {str(e)}")
        return {}

def load_predicted_data(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Carrega dados preditos do JSON processado"""
    predicted = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get("resultados_gerados_run", []) if isinstance(data, dict) else data
        
        for item in results:
            if item is None:
                continue
            
            prop_id = str(item.get('search_term_folder'))  # Converte para string
            if prop_id and 'pontuacao_bic' in item:
                # Extrair tipo de imóvel do JSON
                imovel_parsed = item.get('resposta_modelo_parsed_dict', {})
                imovel_type = str(item.get('type_folder'))
                
                predicted[prop_id] = {
                    'pontuacao_caracteristicas': item['pontuacao_bic']['pontuacao_caracteristicas'],
                    'pontuacao_benfeitorias': item['pontuacao_bic'].get('pontuacao_benfeitorias'),
                    'pontuacao_total': item['pontuacao_bic'].get('pontuacao_total'),
                    'tipo_imovel': imovel_type
                }
        
        print(f"✓ Dados preditos carregados: {len(predicted)} registros")
        print(f"  Primeiros 5 IDs: {sorted(list(predicted.keys()))[:5]}")
        return predicted
    except Exception as e:
        print(f"✗ Erro ao carregar dados preditos: {str(e)}")
        return {}

def score_to_grade(score: int, imovel_type: str) -> str:
    """Converte pontuação para classificação (A, B, C, D, E) baseado no tipo de imóvel"""
    
    # Tabela de limites por tipo de imóvel
    limites = {
        "APARTMENT": {
            "E": (1, 40),
            "D": (41, 45),
            "C": (46, 51),
            "B": (52, 59),
            "A": (60, float('inf'))
        },
        "CONDOMINIUM": {
            "E": (1, 36),
            "D": (37, 41),
            "C": (42, 45),
            "B": (46, 50),
            "A": (51, float('inf'))
        },
        "HOME": {
            "E": (1, 30),
            "D": (31, 38),
            "C": (39, 42),
            "B": (43, 46),
            "A": (47, float('inf'))
        },
        "BUSINESS": {
            "E": (1, 30),
            "D": (31, 38),
            "C": (39, 42),
            "B": (43, 46),
            "A": (47, float('inf'))
        }
    }
    
    # Normalizar tipo de imóvel
    imovel_type = imovel_type.upper().replace(" ", "_").replace("-", "_")
    
    # Usar a tabela correspondente, ou padrão se não encontrado
    if imovel_type not in limites:
        imovel_type = "HOME"
    
    tabela = limites[imovel_type]
    
    # Encontrar o padrão correspondente
    for grade in ['A', 'B', 'C', 'D', 'E']:
        min_score, max_score = tabela[grade]
        if min_score <= score <= max_score:
            return grade
    
    # Fallback (não deveria chegar aqui)
    return 'E'

def calculate_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calcula métricas de classificação (Acurácia, F1, Precisão, Recall)"""
    # Calcula taxa de erro
    correct_predictions = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    taxa_erro = ((len(y_true) - correct_predictions) / len(y_true)) * 100 if len(y_true) > 0 else 0

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Calcular F1-score manualmente usando a fórmula correta
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    metrics = {
        'acuracia': accuracy_score(y_true, y_pred),
        'taxa_erro_padrao_percentual': taxa_erro,
        'quantidade_erros_padrao': len(y_true) - correct_predictions,
        'f1_score': f1_score,
        'precisao': precision,
        'recall': recall
    }
    return metrics

def calculate_score_metrics(y_true: List[int], y_pred: List[int], property_ids: List[str]) -> Dict[str, Any]:
    """Calcula métricas para valores brutos de pontuação"""
    errors = np.array(y_pred) - np.array(y_true)
    absolute_errors = np.abs(errors)
    
    # Contar erros: quando predição diferente do real
    num_errors = np.sum(absolute_errors > 0)
    taxa_erro = (num_errors / len(y_true)) * 100 if len(y_true) > 0 else 0
    
    # Encontrar índices de desvios máximos
    max_pos_idx = np.argmax(errors) if len(errors) > 0 else 0
    max_neg_idx = np.argmin(errors) if len(errors) > 0 else 0
    
    metrics = {
        'desvio_padrao': np.std(absolute_errors),
        'media_erros': np.mean(absolute_errors),
        'taxa_erro_percentual': taxa_erro,
        'quantidade_erros': int(num_errors),
        'tendencia_erro': 'para_cima' if np.mean(errors) > 0 else 'para_baixo',
        'media_tendencia': float(np.mean(errors)),
        'desvio_maximo_positivo': float(np.max(errors)) if len(errors) > 0 else 0,
        'desvio_maximo_positivo_id': property_ids[max_pos_idx] if len(property_ids) > max_pos_idx else 'N/A',
        'desvio_maximo_negativo': float(np.min(errors)) if len(errors) > 0 else 0,
        'desvio_maximo_negativo_id': property_ids[max_neg_idx] if len(property_ids) > max_neg_idx else 'N/A'
    }
    return metrics

def generate_report(ground_truth: Dict, predicted: Dict) -> Dict[str, Any]:
    """Gera relatório completo de métricas"""
    report = {
        'resumo': {},
        'caracteristicas': {},
        'benfeitorias': {},
        'total': {},
        'por_tipo_imovel': {}
    }
    
    # Preparar dados de características
    gt_scores_char = []
    pred_scores_char = []
    gt_grades_char = []
    pred_grades_char = []
    property_ids_char = []
    
    # Preparar dados por tipo de imóvel
    dados_por_tipo = {}
    
    matched_count = 0
    
    for prop_id, gt_data in ground_truth.items():
        if prop_id not in predicted:
            continue
        
        matched_count += 1
        pred_data = predicted[prop_id]
        
        # Características
        if 'pontuacao_caracteristicas' in pred_data and gt_data.get('pontuacao_caracteristicas'):
            gt_score = gt_data['pontuacao_caracteristicas']
            pred_score = pred_data['pontuacao_caracteristicas']
            
            gt_scores_char.append(gt_score)
            pred_scores_char.append(pred_score)
            property_ids_char.append(prop_id)
            
            # Usar tipo de imóvel do JSON (dados preditos)
            imovel_type = str(pred_data.get('tipo_imovel'))
            
            gt_grade = gt_data.get('padrao_caracteristicas', 'E')
            pred_grade = score_to_grade(pred_score, imovel_type)
            
            gt_grades_char.append(gt_grade)
            pred_grades_char.append(pred_grade)
            
            # Agrupar por tipo de imóvel
            if imovel_type not in dados_por_tipo:
                dados_por_tipo[imovel_type] = {
                    'gt_scores': [],
                    'pred_scores': [],
                    'gt_grades': [],
                    'pred_grades': [],
                    'property_ids': []
                }
            
            dados_por_tipo[imovel_type]['gt_scores'].append(gt_score)
            dados_por_tipo[imovel_type]['pred_scores'].append(pred_score)
            dados_por_tipo[imovel_type]['gt_grades'].append(gt_grade)
            dados_por_tipo[imovel_type]['pred_grades'].append(pred_grade)
            dados_por_tipo[imovel_type]['property_ids'].append(prop_id)
    
    print(f"✓ {matched_count} registros pareados entre ground truth e predições")
    
    # Calcular métricas gerais de características
    if gt_scores_char:
        report['caracteristicas']['metricas_classificacao'] = calculate_classification_metrics(
            gt_grades_char, pred_grades_char
        )
        report['caracteristicas']['metricas_pontuacao'] = calculate_score_metrics(
            gt_scores_char, pred_scores_char, property_ids_char
        )
        
        # Matriz de confusão
        cm = confusion_matrix(gt_grades_char, pred_grades_char, labels=['A', 'B', 'C', 'D', 'E'])
        report['caracteristicas']['matriz_confusao'] = {
            'labels': ['A', 'B', 'C', 'D', 'E'],
            'matriz': cm.tolist()
        }
    
    # Calcular métricas por tipo de imóvel
    for imovel_type, dados in dados_por_tipo.items():
        print(f"  Calculando métricas para: {imovel_type} ({len(dados['gt_scores'])} imóveis)")
        
        report['por_tipo_imovel'][imovel_type] = {
            'total_imoveis': len(dados['gt_scores']),
            'metricas_classificacao': calculate_classification_metrics(
                dados['gt_grades'], dados['pred_grades']
            ),
            'metricas_pontuacao': calculate_score_metrics(
                dados['gt_scores'], dados['pred_scores'], dados['property_ids']
            ),
            'matriz_confusao': {
                'labels': ['A', 'B', 'C', 'D', 'E'],
                'matriz': confusion_matrix(dados['gt_grades'], dados['pred_grades'], 
                                          labels=['A', 'B', 'C', 'D', 'E']).tolist()
            }
        }
    
    report['resumo'] = {
        'total_registros_ground_truth': len(ground_truth),
        'total_registros_preditos': len(predicted),
        'registros_pareados': matched_count,
        'tipos_imovel_encontrados': list(dados_por_tipo.keys())
    }
    
    return report

def save_report(report: Dict, output_path: str):
    """Salva relatório em arquivo JSON"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✓ Relatório JSON salvo em: {output_path}")
    except Exception as e:
        print(f"✗ Erro ao salvar relatório JSON: {str(e)}")

def save_report_csv(report: Dict, output_path: str):
    """Salva relatório em arquivo CSV"""
    try:
        rows = []
        
        # Linha de resumo geral
        resumo = report['resumo']
        rows.append({
            'Categoria': 'RESUMO GERAL',
            'Metrica': 'Total Ground Truth',
            'Valor': resumo['total_registros_ground_truth']
        })
        rows.append({
            'Categoria': 'RESUMO GERAL',
            'Metrica': 'Total Preditos',
            'Valor': resumo['total_registros_preditos']
        })
        rows.append({
            'Categoria': 'RESUMO GERAL',
            'Metrica': 'Registros Pareados',
            'Valor': resumo['registros_pareados']
        })
        rows.append({
            'Categoria': 'RESUMO GERAL',
            'Metrica': 'Tipos de Imóvel',
            'Valor': ', '.join(resumo.get('tipos_imovel_encontrados', []))
        })
        rows.append({})  # Linha em branco
        
        # Função para adicionar métricas de uma categoria
        def add_metrics_section(categoria: str, data: Dict):
            rows.append({'Categoria': categoria.upper()})
            
            if 'metricas_classificacao' in data:
                mc = data['metricas_classificacao']
                rows.append({
                    'Categoria': categoria,
                    'Metrica': 'Classificação (A, B, C, D, E)',
                    'Acuracia': f"{mc['acuracia']:.4f}",
                    'Acuracia %': f"{mc['acuracia']*100:.2f}%",
                    'Taxa Erro Padrao %': f"{mc['taxa_erro_padrao_percentual']:.2f}%",
                    'Quantidade Erros Padrao': mc['quantidade_erros_padrao'],
                    'F1-Score': f"{mc['f1_score']:.4f}",
                    'Precisao': f"{mc['precisao']:.4f}",
                    'Recall': f"{mc['recall']:.4f}"
                })
            
            if 'metricas_pontuacao' in data:
                mp = data['metricas_pontuacao']
                rows.append({
                    'Categoria': categoria,
                    'Metrica': 'Pontuação (Valor Bruto)',
                    'Desvio Padrao': f"{mp['desvio_padrao']:.4f}",
                    'Media de Erros': f"{mp['media_erros']:.4f}",
                    'Taxa Erro %': f"{mp['taxa_erro_percentual']:.2f}%",
                    'Quantidade Erros': mp['quantidade_erros'],
                    'Tendencia': mp['tendencia_erro'],
                    'Media Tendencia': f"{mp['media_tendencia']:+.4f}",
                    'Desvio Max Positivo': f"{mp['desvio_maximo_positivo']:+.0f}",
                    'ID Desvio Max Positivo': mp['desvio_maximo_positivo_id'],
                    'Desvio Max Negativo': f"{mp['desvio_maximo_negativo']:+.0f}",
                    'ID Desvio Max Negativo': mp['desvio_maximo_negativo_id']
                })
            
            rows.append({})  # Linha em branco
        
        # Métricas gerais de características
        if 'caracteristicas' in report and report['caracteristicas']:
            add_metrics_section('CARACTERÍSTICAS GERAL', report['caracteristicas'])
        
        # Métricas por tipo de imóvel
        if 'por_tipo_imovel' in report:
            for imovel_type, data in report['por_tipo_imovel'].items():
                rows.append({
                    'Categoria': f'TIPO: {imovel_type}',
                    'Metrica': 'Total de Imóveis',
                    'Valor': data['total_imoveis']
                })
                add_metrics_section(f'TIPO: {imovel_type}', data)
        
        # Escrever CSV
        if rows:
            all_keys = set()
            for row in rows:
                all_keys.update(row.keys())
            all_keys = sorted(list(all_keys))
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"✓ Relatório CSV salvo em: {output_path}")
    except Exception as e:
        print(f"✗ Erro ao salvar relatório CSV: {str(e)}")

def save_confusion_matrix_plot(cm_data: Dict, output_path: str):
    """Salva gráfico da matriz de confusão em PNG"""
    try:
        labels = cm_data['labels']
        matriz = np.array(cm_data['matriz'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Quantidade'})
        
        plt.title('Matriz de Confusão - Classificação de Padrão (A, B, C, D, E)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('Valor Real', fontsize=12)
        plt.xlabel('Valor Predito', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico da matriz de confusão salvo em: {output_path}")
    except Exception as e:
        print(f"✗ Erro ao salvar gráfico da matriz: {str(e)}")

def print_report_summary(report: Dict):
    """Imprime resumo do relatório no console"""
    print("\n" + "="*80)
    print("RELATÓRIO DE MÉTRICAS BIC")
    print("="*80)
    
    print(f"\nResumo Geral:")
    print(f"  Ground Truth: {report['resumo']['total_registros_ground_truth']} registros")
    print(f"  Preditos: {report['resumo']['total_registros_preditos']} registros")
    print(f"  Pareados: {report['resumo']['registros_pareados']} registros")
    print(f"  Tipos de Imóvel: {', '.join(report['resumo'].get('tipos_imovel_encontrados', []))}")
    
    # Função para imprimir métricas de uma categoria
    def print_metrics_section(titulo: str, data: Dict):
        print(f"\n{'─'*80}")
        print(titulo)
        print(f"{'─'*80}")
        
        if 'total_imoveis' in data:
            print(f"Total de Imóveis: {data['total_imoveis']}")
        
        if 'metricas_classificacao' in data:
            mc = data['metricas_classificacao']
            print(f"\nMétricas de Classificação (A, B, C, D, E):")
            print(f"  Acurácia:  {mc['acuracia']:.4f} ({mc['acuracia']*100:.2f}%)")
            print(f"  Taxa de Erro: {mc['taxa_erro_padrao_percentual']:.2f}% ({mc['quantidade_erros_padrao']} erros)")
            print(f"  F1-Score:  {mc['f1_score']:.4f}")
            print(f"  Precisão:  {mc['precisao']:.4f}")
            print(f"  Recall:    {mc['recall']:.4f}")
        
        if 'metricas_pontuacao' in data:
            mp = data['metricas_pontuacao']
            print(f"\nMétricas de Pontuação (Valor Bruto):")
            print(f"  Desvio Padrão: {mp['desvio_padrao']:.4f}")
            print(f"  Média de Erros: {mp['media_erros']:.4f}")
            print(f"  Taxa de Erro: {mp['taxa_erro_percentual']:.2f}% ({mp['quantidade_erros']} erros)")
            print(f"  Tendência: {mp['tendencia_erro']} (média: {mp['media_tendencia']:+.4f})")
            print(f"  Desvio Máximo Positivo: {mp['desvio_maximo_positivo']:+.0f} (ID: {mp['desvio_maximo_positivo_id']})")
            print(f"  Desvio Máximo Negativo: {mp['desvio_maximo_negativo']:+.0f} (ID: {mp['desvio_maximo_negativo_id']})")
        
        if 'matriz_confusao' in data:
            cm_data = data['matriz_confusao']
            print(f"\nMatriz de Confusão:")
            print(f"  Labels: {cm_data['labels']}")
            for i, label in enumerate(cm_data['labels']):
                print(f"  {label}: {cm_data['matriz'][i]}")
    
    # Métricas gerais
    if 'caracteristicas' in report and report['caracteristicas']:
        print_metrics_section('CARACTERÍSTICAS GERAL', report['caracteristicas'])
    
    # Métricas por tipo de imóvel
    if 'por_tipo_imovel' in report:
        for imovel_type, data in report['por_tipo_imovel'].items():
            print_metrics_section(f'TIPO DE IMÓVEL: {imovel_type.upper()}', data)
    
    print(f"\n{'='*80}\n")

def main():
    if len(sys.argv) < 3:
        print("Uso: python script.py <arquivo_json_com_bic> <arquivo_csv_ground_truth>")
        print("\nExemplo:")
        print("  python utils/metricas.py dados_com_bic.json ground_truth.csv")
        sys.exit(1)
    
    json_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    # Validar arquivos
    if not Path(json_file).exists():
        print(f"✗ Erro: Arquivo '{json_file}' não encontrado")
        sys.exit(1)
    
    if not Path(csv_file).exists():
        print(f"✗ Erro: Arquivo '{csv_file}' não encontrado")
        sys.exit(1)
    
    print(f"Processando métricas...")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
    
    # Carregar dados
    ground_truth = load_ground_truth(csv_file)
    predicted = load_predicted_data(json_file)
    
    if not ground_truth or not predicted:
        print("✗ Erro: Não foi possível carregar os dados necessários")
        sys.exit(1)
    
    # Gerar relatório
    report = generate_report(ground_truth, predicted)
    
    # Salvar relatório
    output_json = Path(json_file).parent / f"{Path(json_file).stem}_metricas.json"
    output_csv = Path(json_file).parent / f"{Path(json_file).stem}_metricas.csv"
    output_png = Path(json_file).parent / f"{Path(json_file).stem}_matriz_confusao.png"
    
    save_report(report, str(output_json))
    save_report_csv(report, str(output_csv))
    
    # Salvar gráfico da matriz de confusão geral
    if 'caracteristicas' in report and 'matriz_confusao' in report['caracteristicas']:
        save_confusion_matrix_plot(report['caracteristicas']['matriz_confusao'], str(output_png))
    
    # Salvar gráficos por tipo de imóvel
    if 'por_tipo_imovel' in report:
        for imovel_type, data in report['por_tipo_imovel'].items():
            if 'matriz_confusao' in data:
                output_png_tipo = Path(json_file).parent / f"{Path(json_file).stem}_matriz_confusao_{imovel_type}.png"
                save_confusion_matrix_plot(data['matriz_confusao'], str(output_png_tipo))
    
    # Imprimir resumo
    print_report_summary(report)

if __name__ == "__main__":
    main()