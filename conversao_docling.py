import sys
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from pypdf import PdfReader

current_dir = Path(__file__).parent
pasta_pdfs = current_dir / "documentos"
pasta_markdowns = current_dir / "markdowns"
pasta_markdowns.mkdir(exist_ok=True)


def extrair_texto_simples(pdf_path: Path, paginas: list[int]) -> str:
    """Extrai texto puro (sem IA) de páginas específicas usando pypdf.
    Serve como reforço para páginas que o Docling não conseguiu processar."""
    leitor = PdfReader(str(pdf_path))
    blocos = []
    for num_pagina in paginas:
        try:
            pagina = leitor.pages[num_pagina - 1]  # pypdf começa em 0
            texto = (pagina.extract_text() or "").strip()
            blocos.append(f"\n\n--- [Página {num_pagina} extraída em modo simples] ---\n\n{texto}")
        except Exception as e:
            blocos.append(f"\n\n--- [Página {num_pagina}: falhou também no modo simples: {e}] ---\n\n")
    return "\n".join(blocos)


def converter_arquivo(nome_arquivo: str):
    pdf_path = pasta_pdfs / nome_arquivo

    if not pdf_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {pdf_path}")
        return

    print(f"Processando: {pdf_path.name}")

    # Desliga processamento de imagem/OCR, que é a causa provável do travamento
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.do_picture_classification = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

    result = converter.convert(str(pdf_path.resolve()), raises_on_error=False)

    print(f"Status da conversão: {result.status}")
    print(f"Páginas processadas: {len(result.pages)}")

    markdown_content = result.document.export_to_markdown()

    if result.status == ConversionStatus.PARTIAL_SUCCESS and result.errors:
        paginas_com_erro = sorted({
            int(err.error_message.split("Page ")[1].split(":")[0])
            for err in result.errors
            if "Page" in err.error_message
        })
        print(f"[AVISO] Páginas com erro: {paginas_com_erro}. Tentando modo simples para elas...")

        texto_reforco = extrair_texto_simples(pdf_path, paginas_com_erro)
        markdown_content += "\n\n## Conteúdo recuperado (páginas com erro no processamento avançado)\n" + texto_reforco

    output_path = pasta_markdowns / f"{pdf_path.stem}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python teste.py nome_do_arquivo.pdf")
    else:
        converter_arquivo(sys.argv[1])