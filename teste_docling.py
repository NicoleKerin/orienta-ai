import os
from pathlib import Path
from docling.document_converter import DocumentConverter

# 1. MAPEAMENTO DE PASTAS (Usando Pathlib)
# Detecta a pasta atual onde o script está rodando (OrientAI)
current_dir = Path(__file__).parent

# Define onde buscar os PDFs e onde salvar os Markdowns
pasta_pdfs = current_dir / "documentos"
pasta_markdowns = current_dir / "markdowns"

# Cria a pasta "markdowns" automaticamente se ela não existir
pasta_markdowns.mkdir(exist_ok=True)


# 2. INICIALIZAÇÃO DO DOCLING
print("=" * 50)
print("Inicializando o motor do Docling...")
print("=" * 50)
converter = DocumentConverter()


# 3. BUSCA DOS ARQUIVOS
# Procura todos os arquivos .pdf dentro da pasta "documentos"
pdf_files = list(pasta_pdfs.glob("*.pdf"))

if not pdf_files:
    print(f"\n[AVISO] Nenhum arquivo PDF foi encontrado na pasta: {pasta_pdfs}")
    print("Por favor, coloque seus PDFs lá dentro e execute o script novamente.")
else:
    print(f"\nSucesso! Encontrado(s) {len(pdf_files)} arquivo(s) para conversão.\n")
    print("-" * 50)

    # 4. LOOP DE CONVERSÃO EM LOTE
    for i, pdf_path in enumerate(pdf_files, 1):
        
        # Define o destino final mudando a pasta e alterando a extensão para .md
        output_path = pasta_markdowns / f"{pdf_path.stem}.md"
        
        print(f"[{i}/{len(pdf_files)}] Processando: {pdf_path.name}")
        print(f" -> Destino: markdowns/{output_path.name}")
        
        try:
            # O Docling faz a mágica de ler o PDF estruturadamente
            result = converter.convert(str(pdf_path.resolve()))
            markdown_content = result.document.export_to_markdown()
            
            # Grava o arquivo .md final na pasta isolada de markdowns
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                
            print(f" -> [OK] Convertido com sucesso!\n")
            
        except Exception as e:
            print(f" -> [ERRO] Falha ao converter este arquivo. Detalhes: {e}\n")

    print("-" * 50)
    print("PROCESSO CONCLUÍDO!")
    print(f"Os PDFs originais continuam na pasta: /{pasta_pdfs.name}")
    print(f"Os novos arquivos para a sua IA estão prontos na pasta: /{pasta_markdowns.name}")
    print("=" * 50)