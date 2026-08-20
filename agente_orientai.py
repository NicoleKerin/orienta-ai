

import json
import os
import sys
import time

CHUNKS_FILE = "chunks.json"
COLLECTION_NAME = "orientai"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # mesmo modelo multilingue de antes
GROQ_MODEL = "llama-3.1-8b-instant"


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Variavel de ambiente GROQ_API_KEY nao encontrada.")
        print("Crie uma chave gratis em https://console.groq.com e defina:")
        print('    export GROQ_API_KEY="sua-chave-aqui"')
        sys.exit(1)

    if not os.path.exists(CHUNKS_FILE):
        print(f"Arquivo '{CHUNKS_FILE}' nao encontrado.")
        print("Rode primeiro o chunk_markdown.py pra gerar os chunks.")
        sys.exit(1)

    def find_class(candidates, class_name):
        """Tenta importar `class_name` de cada caminho em `candidates`, na ordem."""
        for module_path in candidates:
            try:
                mod = __import__(module_path, fromlist=[class_name])
                return getattr(mod, class_name)
            except (ImportError, AttributeError):
                continue
        return None

    def diagnostico_e_sai():
        print("\nNao consegui montar os imports certos pra essa versao do agno.")
        print("Roda o comando abaixo e me manda o resultado completo, que eu ajusto:\n")
        print('    python -c "import agno, pkgutil; [print(m.name) for m in pkgutil.walk_packages(agno.__path__, agno.__name__+chr(46))]"\n')
        sys.exit(1)

    from agno.agent import Agent

    Knowledge = find_class(
        ["agno.knowledge.knowledge", "agno.knowledge"], "Knowledge"
    )
    ChromaDb = find_class(
        ["agno.vectordb.chroma", "agno.vectordb.chroma.chroma_db"], "ChromaDb"
    )
    SentenceTransformerEmbedder = find_class(
        [
            "agno.embedder.sentence_transformer",
            "agno.knowledge.embedder.sentence_transformer",
        ],
        "SentenceTransformerEmbedder",
    )
    Groq = find_class(
        ["agno.models.groq", "agno.models.groq.groq"], "Groq"
    )

    if not all([Knowledge, ChromaDb, SentenceTransformerEmbedder, Groq]):
        diagnostico_e_sai()

    # O caminho de import do ChunkingStrategy/Document muda entre
    # versoes do agno (a doc oficial e o codigo-fonte real as vezes
    # divergem). Tenta os caminhos conhecidos, na ordem mais provavel.
    ChunkingStrategy = find_class(
        ["agno.knowledge.chunking.base", "agno.knowledge.chunking.strategy"],
        "ChunkingStrategy",
    )
    Document = find_class(
        ["agno.knowledge.document", "agno.knowledge.document.base"],
        "Document",
    )
    TextReader = find_class(
        ["agno.knowledge.reader.text_reader"],
        "TextReader",
    )

    if ChunkingStrategy is None or Document is None or TextReader is None:
        diagnostico_e_sai()


    class ChunksProntosDoOrientAI(ChunkingStrategy):
        def chunk(self, document: Document) -> list:
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            return [
                Document(
                    content=c["text"],
                    meta_data={
                        "source_file": c["source_file"],
                        "header_path": " > ".join(c.get("header_path", [])),
                        "chunk_id": c["id"],
                    },
                )
                for c in chunks
            ]

    print("Configurando a base de conhecimento (embeddings + Chroma)...")
    knowledge = Knowledge(
        vector_db=ChromaDb(
            collection=COLLECTION_NAME,
            embedder=SentenceTransformerEmbedder(id=EMBEDDING_MODEL),
            path="./chroma_db",  # fica salvo localmente nessa pasta,
            persistent_client=True,
        ),
    )

    # Nessa versao do agno, a chunking_strategy vai no Reader, nao
    # direto no add_content(). O reader ainda "le" o chunks.json (por
    # isso o path=CHUNKS_FILE abaixo), mas quem decide como cortar o
    # conteudo em pedacos e o metodo chunk() da nossa estrategia -
    # que ignora o texto bruto lido e usa os chunks ja prontos.
    reader = TextReader(chunking_strategy=ChunksProntosDoOrientAI())

    print("Gerando embeddings e indexando no Chroma (pode demorar na primeira vez)...")

    INGESTED_FLAG = "./data/.ingested"
    if not os.path.exists(INGESTED_FLAG):
        knowledge.add_content(
            name="Documentos do IFSC",
            path=CHUNKS_FILE,
            reader=reader,
        )

    agent = Agent(
        model=Groq(id=GROQ_MODEL),
        knowledge=knowledge,
        search_knowledge=True,
        instructions=[
            "Você é um assistente que responde perguntas com base nos documentos institucionais do IFSC.",
            "Responda sempre em português, de forma clara e objetiva.",
            "Baseie suas respostas apenas no conteúdo recuperado da base de conhecimento.",
            "Se a informação não estiver nos documentos, diga isso claramente em vez de inventar.",
            "Quando fizer sentido, cite de qual documento ou capitulo veio a informação.",
            "Adequar a linguagem das respostas para um público entre 14 e 20 anos."
        ],
    )

    print("\nPronto! Base indexada. Digite sua pergunta (ou 'sair' pra encerrar).\n")

    MAX_TENTATIVAS = 3

    while True:
        pergunta = input("Voce: ").strip()
        if pergunta.lower() in ("sair", "exit", "quit"):
            break
        if not pergunta:
            continue

        for tentativa in range(1, MAX_TENTATIVAS + 1):
            try:
                agent.print_response(pergunta, stream=True)
                break  # deu certo, sai do loop de retry
            except Exception as e:
                # Erro conhecido/intermitente do Groq com function calling.
                # Tenta de novo automaticamente antes de desistir.
                if tentativa < MAX_TENTATIVAS:
                    print(f"\n(falha temporaria, tentando de novo... {tentativa}/{MAX_TENTATIVAS})")
                    time.sleep(1.5)
                else:
                    print(f"\nNao consegui responder apos {MAX_TENTATIVAS} tentativas.")
                    print(f"Erro: {e}")
                    print("Tenta reformular a pergunta ou perguntar de novo.")
        print()


if __name__ == "__main__":
    main()