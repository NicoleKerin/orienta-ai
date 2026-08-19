"""
app_web.py
------------
Interface web (chat bonitinho) para o agente do OrientAI, usando
Streamlit. Reaproveita a mesma logica de montagem do agno que ja
estava no agente_orientai.py (Chroma + embeddings + Groq).

Instalacao (uma vez so, alem do que voce ja tem):
    pip install streamlit

Uso:
    streamlit run app_web.py

Isso abre automaticamente uma aba no seu navegador com o chat.
Pra fechar, Ctrl+C no terminal onde rodou o comando.
"""

import json
import os
import sys
import time

import streamlit as st

CHUNKS_FILE = "chunks.json"
COLLECTION_NAME = "orientai"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
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


@st.cache_resource(show_spinner="Configurando a base de conhecimento (so na primeira vez)...")
def montar_agente():
    """
    Monta o agente uma unica vez e guarda em cache - assim o Streamlit
    nao reindexa tudo de novo a cada pergunta (ele reroda o script
    inteiro por baixo dos panos a cada interacao, e o cache evita
    refazer o trabalho pesado).
    """
    from agno.agent import Agent

    Knowledge = find_class(["agno.knowledge.knowledge", "agno.knowledge"], "Knowledge")
    ChromaDb = find_class(["agno.vectordb.chroma", "agno.vectordb.chroma.chroma_db"], "ChromaDb")
    SentenceTransformerEmbedder = find_class(
        ["agno.embedder.sentence_transformer", "agno.knowledge.embedder.sentence_transformer"],
        "SentenceTransformerEmbedder",
    )
    Groq = find_class(["agno.models.groq", "agno.models.groq.groq"], "Groq")
    ChunkingStrategy = find_class(
        ["agno.knowledge.chunking.base", "agno.knowledge.chunking.strategy"], "ChunkingStrategy"
    )
    Document = find_class(["agno.knowledge.document", "agno.knowledge.document.base"], "Document")
    TextReader = find_class(["agno.knowledge.reader.text_reader"], "TextReader")

    if not all([Knowledge, ChromaDb, SentenceTransformerEmbedder, Groq, ChunkingStrategy, Document, TextReader]):
        st.error(
            "Nao consegui importar as classes do agno. "
            "Roda o diagnostico_agno.py no terminal e manda o resultado pro Claude ajustar."
        )
        st.stop()

    class ChunksProntosDoOrientAI(ChunkingStrategy):
        def chunk(self, document):
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
        ),
    )

    # skip_if_exists do add_content nao esta funcionando nessa versao do
    # agno (ele reindexa de qualquer jeito). Entao a gente mesmo checa:
    # se a pasta chroma_db ja existe com arquivos dentro, e sinal de que
    # a indexacao ja rodou antes - nao reprocessa/reembeda de novo.
    ja_indexado = os.path.isdir("./chroma_db") and len(os.listdir("./chroma_db")) > 0

    if not ja_indexado:
        reader = TextReader(chunking_strategy=ChunksProntosDoOrientAI())
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
            "Voce se chama OrientAI, um assistente virtual dos estudantes do IFSC.",
            "Seu tom e amigavel, acolhedor e direto - como um veterano ajudando um calouro, nunca formal ou burocratico demais.",
            "Responda sempre em portugues, de forma clara e objetiva.",
            "Baseie suas respostas apenas no conteudo recuperado da base de conhecimento.",
            "Se a informacao nao estiver nos documentos, diga isso claramente e de forma simpatica, em vez de inventar.",
            "Quando fizer sentido, cite de qual documento ou capitulo veio a informacao.",
            "Nao repita sua apresentacao a cada resposta - apresente-se so uma vez, no inicio da conversa.",
        ],
    )
    return agent


MENSAGEM_BOAS_VINDAS = (
    "Oi! Eu sou o **OrientAI** 🎓, o assistente virtual dos estudantes do IFSC.\n\n"
    "Posso te ajudar com dúvidas sobre o Código de Convivência Discente e outros documentos "
    "institucionais - direitos, deveres, o que pode e o que não pode, medidas disciplinares, "
    "bolsas e mais.\n\n"
    "Manda sua pergunta aí embaixo!"
)


def _parece_erro(texto):
    """
    Detecta quando o agno devolve o erro do Groq (function calling
    malformado) como se fosse uma resposta normal, em vez de lancar
    uma excecao de verdade. Nesses casos o texto vem parecido com
    um JSON cru: {"error": {"message": "Failed to call a function...
    """
    if not texto:
        return True
    marcadores = ("failed_generation", "tool_use_failed", '"error"', "invalid_request_error")
    return any(m in texto for m in marcadores)


def responder_com_retry(agent, pergunta):
    """Tenta responder ate MAX_TENTATIVAS vezes antes de desistir (erro intermitente do Groq)."""
    ultimo_erro = None

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = agent.run(pergunta)
            texto = resposta.content

            if _parece_erro(texto):
                ultimo_erro = texto
                if tentativa < MAX_TENTATIVAS:
                    time.sleep(1.5)
                    continue
                else:
                    return (
                        "Tive uma falha temporaria pra buscar essa informacao. "
                        "Tenta reformular a pergunta ou perguntar de novo em alguns segundos."
                    )

            return texto

        except Exception as e:
            ultimo_erro = str(e)
            if tentativa < MAX_TENTATIVAS:
                time.sleep(1.5)
            else:
                return (
                    "Tive uma falha temporaria pra buscar essa informacao. "
                    "Tenta reformular a pergunta ou perguntar de novo em alguns segundos."
                )


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(page_title="OrientAI", page_icon="🎓", layout="centered")

st.title("🎓 OrientAI")
st.caption("Assistente baseado nos documentos institucionais do IFSC")

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "Variavel de ambiente GROQ_API_KEY nao encontrada.\n\n"
        "Cria uma chave gratis em https://console.groq.com e define ela no terminal "
        "antes de rodar o Streamlit."
    )
    st.stop()

if not os.path.exists(CHUNKS_FILE):
    st.error(f"Arquivo '{CHUNKS_FILE}' nao encontrado nessa pasta. Roda o chunk_markdown.py primeiro.")
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