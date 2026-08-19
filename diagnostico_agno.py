"""
diagnostico_agno.py
---------------------
Descobre a assinatura real de Knowledge, add_content, ChromaDb e
ChunkingStrategy na versao do agno que voce tem instalada, ja que
a documentacao oficial nao bate com o pacote publicado.

Uso:
    python diagnostico_agno.py

Cola a saida completa de volta na conversa.
"""

import inspect


def show(label, obj):
    print(f"\n=== {label} ===")
    try:
        print(inspect.signature(obj))
    except (TypeError, ValueError) as e:
        print(f"(nao foi possivel inspecionar: {e})")


def find_class(candidates, class_name):
    for module_path in candidates:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            return getattr(mod, class_name), module_path
        except (ImportError, AttributeError):
            continue
    return None, None


Knowledge, k_path = find_class(["agno.knowledge.knowledge", "agno.knowledge"], "Knowledge")
ChunkingStrategy, cs_path = find_class(
    ["agno.knowledge.chunking.base", "agno.knowledge.chunking.strategy"], "ChunkingStrategy"
)

if Knowledge:
    print(f"Knowledge encontrado em: {k_path}")
    show("Knowledge.__init__", Knowledge.__init__)
    show("Knowledge.add_content", Knowledge.add_content)
    if hasattr(Knowledge, "add_content_async"):
        show("Knowledge.add_content_async", Knowledge.add_content_async)
else:
    print("Knowledge nao encontrado nos caminhos testados.")

if ChunkingStrategy:
    print(f"\nChunkingStrategy encontrado em: {cs_path}")
else:
    print("\nChunkingStrategy nao encontrado nos caminhos testados.")

# Tenta achar tambem algum "Reader" generico, ja que em varias
# versoes do agno o chunking_strategy e passado pro reader, nao
# direto pro add_content.
Reader, r_path = find_class(
    ["agno.knowledge.reader.base", "agno.knowledge.reader"], "Reader"
)
if Reader:
    print(f"\nReader encontrado em: {r_path}")
    show("Reader.__init__", Reader.__init__)

# Lista todos os submodulos do agno, pra mapear a estrutura completa.
print("\n=== Submodulos do pacote agno ===")
import agno
import pkgutil

for m in pkgutil.walk_packages(agno.__path__, agno.__name__ + "."):
    if any(k in m.name for k in ("knowledge", "reader", "chunk", "document")):
        print(m.name)