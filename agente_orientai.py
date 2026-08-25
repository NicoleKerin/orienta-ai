import json
import os
import sys
import time

import streamlit as st

CHUNKS_FILE = "chunks.json"
COLLECTION_NAME = "orientai"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL = "qwen/qwen3.6-27b"
MAX_TENTATIVAS = 3


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
    st.error(
        "Não consegui montar os imports certos para essa versão do agno.\n\n"
        "Roda o diagnóstico no terminal e me manda o resultado."
    )
    st.stop()


@st.cache_resource(show_spinner="Configurando a base de conhecimento (embeddings + Chroma)...")
def montar_agente():
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

    knowledge = Knowledge(
        vector_db=ChromaDb(
            collection=COLLECTION_NAME,
            embedder=SentenceTransformerEmbedder(id=EMBEDDING_MODEL),
            path="./chroma_db",
            persistent_client=True,
        )
    )

    reader = TextReader(chunking_strategy=ChunksProntosDoOrientAI())

    # Garante que a pasta data existe antes de criar a flag
    os.makedirs("./data", exist_ok=True)
    INGESTED_FLAG = "./data/.ingested"
    
    if not os.path.exists(INGESTED_FLAG):
        knowledge.add_content(
            name="Documentos do IFSC",
            path=CHUNKS_FILE,
            reader=reader,
        )
        open(INGESTED_FLAG, 'w').close()

    agent = Agent(
        model=Groq(id=GROQ_MODEL),
        knowledge=knowledge,
        search_knowledge=True,
        instructions=[
            "Faça apenas UMA busca na base de conhecimento por vez.",
            "Você se chama OrientAI, um assistente virtual dos estudantes do IFSC.",
            "Não repita sua apresentação a cada resposta - apresente-se só uma vez, no início da conversa.",
            "Seu tom é amigável, acolhedor e direto - como um veterano ajudando um calouro, nunca formal ou burocrático demais.",
            "Responda sempre em português, de forma clara e objetiva.",
            "Para dúvidas sobre regras, direitos, deveres ou documentos do IFSC, baseie-se estritamente na base de conhecimento.",
            "Para perguntas gerais, saudações, matemática básica ou conversas comuns que não exigem documentos institucionais, você pode responder normalmente usando seu conhecimento geral.",
            "Se a pergunta for sobre o IFSC e a informação não estiver nos documentos recuperados, diga isso claramente sem inventar.",
            "Se a pergunta for sobre o IFSC, quando fizer sentido, cite de qual documento ou capítulo veio a informação.",
        ],
    )
    return agent


MENSAGEM_BOAS_VINDAS = (
    "Oi! Eu sou o **OrientAI** 🎓, o assistente virtual dos estudantes do IFSC.\n\n"
    "Manda sua pergunta aí embaixo sobre os documentos institucionais!"
)


def _parece_erro(texto):
    if not texto:
        return True
    marcadores = ("failed_generation", "tool_use_failed", '"error"', "invalid_request_error")
    return any(m in texto for m in marcadores)


def responder_com_retry(agent, pergunta):
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = agent.run(pergunta)
            texto = resposta.content
            if _parece_erro(texto):
                if tentativa < MAX_TENTATIVAS:
                    time.sleep(1.5)
                    continue
                else:
                    return "Tive uma falha temporária pra buscar essa informação. Tenta de novo em instantes."
            return texto
        except Exception:
            if tentativa < MAX_TENTATIVAS:
                time.sleep(1.5)
            else:
                return "Tive uma falha temporária pra buscar essa informação. Tenta de novo em instantes."


# ============================================================
# INTERFACE WEB (STREAMLIT)
# ============================================================

st.set_page_config(page_title="OrientAI", page_icon="🎓", layout="centered")

st.title("🎓 OrientAI")
st.caption("Assistente baseado nos documentos institucionais do IFSC")

if not os.environ.get("GROQ_API_KEY"):
    st.error("Variável de ambiente GROQ_API_KEY não encontrada. Defina ela antes de rodar.")
    st.stop()

if not os.path.exists(CHUNKS_FILE):
    st.error(f"Arquivo '{CHUNKS_FILE}' não encontrado. Rode primeiro o chunk_markdown.py.")
    st.stop()

agent = montar_agente()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = [{"role": "assistant", "content": MENSAGEM_BOAS_VINDAS}]

for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pergunta = st.chat_input("Digite sua pergunta sobre os documentos do IFSC...")

if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando nos documentos..."):
            resposta = responder_com_retry(agent, pergunta)
        st.markdown(resposta)

    st.session_state.mensagens.append({"role": "assistant", "content": resposta})