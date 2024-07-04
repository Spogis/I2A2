import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Diretórios do dataset
train_dir = 'datasets/training_set'
test_dir = 'datasets/test_set'


# Função para obter estatísticas das imagens
def obter_estatisticas_imagens(diretorio):
    tamanhos = []
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    tamanhos.append(img.shape)
    return tamanhos


# Função para gerar relatório de entendimento do negócio
def entendimento_negocio():
    objetivo = """
    Objetivo do Projeto:
    Desenvolver um modelo de classificação de imagens capaz de distinguir entre gatos e cachorros.
    Este modelo pode ser utilizado em um aplicativo móvel de reconhecimento de animais de estimação.
    """
    requisitos = """
    Requisitos do Negócio:
    - Alta precisão na classificação de imagens de gatos e cachorros.
    - Desempenho em tempo real para uso em aplicativos móveis.
    - Capacidade de lidar com diversas condições de iluminação e ângulos de imagem.
    """
    sucesso = """
    Sucesso do Projeto:
    O sucesso será medido pela precisão do modelo em um conjunto de dados de teste, com uma meta mínima de 90% de precisão.
    A rapidez do modelo em classificar uma imagem também será considerada, com um tempo de resposta ideal menor que 1 segundo.
    """
    return objetivo, requisitos, sucesso


# Função para gerar relatório de entendimento dos dados
def entendimento_dados(train_dir, test_dir):
    data_summary = {
        "Diretório de Treinamento": train_dir,
        "Diretório de Teste": test_dir,
        "Total de Imagens de Treinamento": len(os.listdir(train_dir + '/cats')) + len(os.listdir(train_dir + '/dogs')),
        "Total de Imagens de Teste": len(os.listdir(test_dir + '/cats')) + len(os.listdir(test_dir + '/dogs')),
        "Formato das Imagens": "JPEG",
        "Classes": ["cats", "dogs"],
        "Distribuição das Classes": {
            "Treinamento": {"cats": len(os.listdir(train_dir + '/cats')), "dogs": len(os.listdir(train_dir + '/dogs'))},
            "Teste": {"cats": len(os.listdir(test_dir + '/cats')), "dogs": len(os.listdir(test_dir + '/dogs'))}
        }
    }

    # Análise das imagens de treinamento
    tamanhos_treinamento = obter_estatisticas_imagens(train_dir)
    alturas_treinamento = [tamanho[0] for tamanho in tamanhos_treinamento]
    larguras_treinamento = [tamanho[1] for tamanho in tamanhos_treinamento]

    # Análise das imagens de teste
    tamanhos_teste = obter_estatisticas_imagens(test_dir)
    alturas_teste = [tamanho[0] for tamanho in tamanhos_teste]
    larguras_teste = [tamanho[1] for tamanho in tamanhos_teste]

    # Estatísticas das imagens
    stats = {
        "Treinamento": {
            "Altura Média": np.mean(alturas_treinamento),
            "Largura Média": np.mean(larguras_treinamento),
            "Altura Mínima": np.min(alturas_treinamento),
            "Altura Máxima": np.max(alturas_treinamento),
            "Largura Mínima": np.min(larguras_treinamento),
            "Largura Máxima": np.max(larguras_treinamento),
        },
        "Teste": {
            "Altura Média": np.mean(alturas_teste),
            "Largura Média": np.mean(larguras_teste),
            "Altura Mínima": np.min(alturas_teste),
            "Altura Máxima": np.max(alturas_teste),
            "Largura Mínima": np.min(larguras_teste),
            "Largura Máxima": np.max(larguras_teste),
        }
    }

    return data_summary, stats, alturas_treinamento, larguras_treinamento


# Função para gerar relatório de preparação dos dados
def preparacao_dados(train_dir):
    img_width, img_height = 150, 150
    batch_size = 32
    validation_split = 0.2

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    data_prep_summary = {
        "Dimensões das Imagens": (img_width, img_height),
        "Tamanho do Batch": batch_size,
        "Divisão de Validação": validation_split,
        "Aumento de Dados": ["shear_range=0.2", "zoom_range=0.2", "horizontal_flip=True"],
        "Número de Imagens de Treinamento": train_generator.samples,
        "Número de Imagens de Validação": validation_generator.samples,
    }
    return data_prep_summary


# Função para gerar gráficos e salvar como imagens
def gerar_graficos(alturas_treinamento, larguras_treinamento):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(alturas_treinamento, kde=True)
    plt.title('Distribuição das Alturas das Imagens de Treinamento')
    plt.xlabel('Altura (pixels)')
    plt.ylabel('Frequência')

    plt.subplot(1, 2, 2)
    sns.histplot(larguras_treinamento, kde=True)
    plt.title('Distribuição das Larguras das Imagens de Treinamento')
    plt.xlabel('Largura (pixels)')
    plt.ylabel('Frequência')

    plt.tight_layout()
    plt.savefig('distribuicao_tamanhos_treinamento.png')


# Função para gerar relatório CRISP-DM em PDF
def gerar_relatorio_pdf():
    objetivo, requisitos, sucesso = entendimento_negocio()
    data_summary, stats, alturas_treinamento, larguras_treinamento = entendimento_dados(train_dir, test_dir)
    data_prep_summary = preparacao_dados(train_dir)

    # Gerar gráficos
    gerar_graficos(alturas_treinamento, larguras_treinamento)

    # Criar PDF
    pdf_path = "relatorio_crisp_dm.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []

    # Estilos
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']
    style_heading = styles['Heading1']
    style_heading2 = styles['Heading2']

    # Adicionar título
    elements.append(Paragraph("Relatório CRISP-DM: Classificação de Imagens de Gatos e Cachorros", style_heading))
    elements.append(Spacer(1, 12))

    # Seção 1: Entendimento do Negócio
    elements.append(Paragraph("1. Entendimento do Negócio", style_heading2))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(objetivo, style_normal))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(requisitos, style_normal))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(sucesso, style_normal))
    elements.append(Spacer(1, 24))

    # Seção 2: Entendimento dos Dados
    elements.append(Paragraph("2. Entendimento dos Dados", style_heading2))
    elements.append(Spacer(1, 12))
    data_summary_str = f"""
    Diretório de Treinamento: {data_summary['Diretório de Treinamento']}<br/>
    Diretório de Teste: {data_summary['Diretório de Teste']}<br/>
    Total de Imagens de Treinamento: {data_summary['Total de Imagens de Treinamento']}<br/>
    Total de Imagens de Teste: {data_summary['Total de Imagens de Teste']}<br/>
    Formato das Imagens: {data_summary['Formato das Imagens']}<br/>
    Classes: {', '.join(data_summary['Classes'])}<br/>
    Distribuição das Classes (Treinamento): Gatos - {data_summary['Distribuição das Classes']['Treinamento']['cats']}, Cachorros - {data_summary['Distribuição das Classes']['Treinamento']['dogs']}<br/>
    Distribuição das Classes (Teste): Gatos - {data_summary['Distribuição das Classes']['Teste']['cats']}, Cachorros - {data_summary['Distribuição das Classes']['Teste']['dogs']}
    """
    elements.append(Paragraph(data_summary_str, style_normal))
    elements.append(Spacer(1, 12))

    stats_str = f"""
    Estatísticas das Imagens de Treinamento:<br/>
    Altura Média: {stats['Treinamento']['Altura Média']:.2f} pixels<br/>
    Largura Média: {stats['Treinamento']['Largura Média']:.2f} pixels<br/>
    Altura Mínima: {stats['Treinamento']['Altura Mínima']} pixels<br/>
    Altura Máxima: {stats['Treinamento']['Altura Máxima']} pixels<br/>
    Largura Mínima: {stats['Treinamento']['Largura Mínima']} pixels<br/>
    Largura Máxima: {stats['Treinamento']['Largura Máxima']} pixels<br/><br/>
    Estatísticas das Imagens de Teste:<br/>
    Altura Média: {stats['Teste']['Altura Média']:.2f} pixels<br/>
    Largura Média: {stats['Teste']['Largura Média']:.2f} pixels<br/>
    Altura Mínima: {stats['Teste']['Altura Mínima']} pixels<br/>
    Altura Máxima: {stats['Teste']['Altura Máxima']} pixels<br/>
    Largura Mínima: {stats['Teste']['Largura Mínima']} pixels<br/>
    Largura Máxima: {stats['Teste']['Largura Máxima']} pixels
    """
    elements.append(Paragraph(stats_str, style_normal))
    elements.append(Spacer(1, 24))

    # Seção 3: Preparação dos Dados
    elements.append(Paragraph("3. Preparação dos Dados", style_heading2))
    elements.append(Spacer(1, 12))
    data_prep_summary_str = f"""
    Dimensões das Imagens: {data_prep_summary['Dimensões das Imagens']}<br/>
    Tamanho do Batch: {data_prep_summary['Tamanho do Batch']}<br/>
    Divisão de Validação: {data_prep_summary['Divisão de Validação']}<br/>
    Aumento de Dados: {', '.join(data_prep_summary['Aumento de Dados'])}<br/>
    Número de Imagens de Treinamento: {data_prep_summary['Número de Imagens de Treinamento']}<br/>
    Número de Imagens de Validação: {data_prep_summary['Número de Imagens de Validação']}
    """
    elements.append(Paragraph(data_prep_summary_str, style_normal))
    elements.append(Spacer(1, 24))

    # Adicionar gráficos
    elements.append(Paragraph("Gráficos de Distribuição dos Tamanhos das Imagens de Treinamento", style_heading2))
    elements.append(Spacer(1, 12))
    elements.append(Image('distribuicao_tamanhos_treinamento.png', width=550, height=450))

    # Construir o PDF
    doc.build(elements)
    print(f"Relatório CRISP-DM gerado e salvo como '{pdf_path}'")


# Executar geração do relatório
gerar_relatorio_pdf()
