import hashlib
import json
import os
import shutil
import time

# Evita checar internet no Hugging Face se o modelo ja foi baixado antes
if os.path.exists(os.path.expanduser("~/.cache/huggingface/hub")):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st

CHUNKS_FILE = "chunks.json"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "orientai"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
GEMINI_MODEL = "gemini-3.5-flash"
MAX_TENTATIVAS = 3


def find_class(candidates, class_name):
    for module_path in candidates:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            return getattr(mod, class_name)
        except (ImportError, AttributeError):
            continue
    return None


def calcular_hash_arquivo(caminho):
    with open(caminho, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@st.cache_resource(show_spinner="Carregando base de conhecimento do OrientAI...")
def montar_agente():
    from agno.agent import Agent

    Knowledge = find_class(["agno.knowledge.knowledge", "agno.knowledge"], "Knowledge")
    ChromaDb = find_class(["agno.vectordb.chroma", "agno.vectordb.chroma.chroma_db"], "ChromaDb")
    SentenceTransformerEmbedder = find_class(
        ["agno.embedder.sentence_transformer", "agno.knowledge.embedder.sentence_transformer"],
        "SentenceTransformerEmbedder",
    )
    Gemini = find_class(["agno.models.google", "agno.models.google.gemini"], "Gemini")
    ChunkingStrategy = find_class(
        ["agno.knowledge.chunking.base", "agno.knowledge.chunking.strategy"], "ChunkingStrategy"
    )
    Document = find_class(["agno.knowledge.document", "agno.knowledge.document.base"], "Document")
    TextReader = find_class(["agno.knowledge.reader.text_reader"], "TextReader")

    if not all([Knowledge, ChromaDb, SentenceTransformerEmbedder, Gemini, ChunkingStrategy, Document, TextReader]):
        st.error("Nao consegui importar as classes do agno. Verifique a instalacao.")
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

    hash_file = os.path.join(DB_PATH, "chunks.sha256")
    hash_atual = calcular_hash_arquivo(CHUNKS_FILE)
    hash_salvo = None

    if os.path.exists(hash_file):
        with open(hash_file, "r", encoding="utf-8") as f:
            hash_salvo = f.read().strip()

    precisa_indexar = (hash_atual != hash_salvo)

    if precisa_indexar and os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    knowledge = Knowledge(
        vector_db=ChromaDb(
            collection=COLLECTION_NAME,
            embedder=SentenceTransformerEmbedder(id=EMBEDDING_MODEL),
            path=DB_PATH,
        ),
    )

    if precisa_indexar:
        reader = TextReader(chunking_strategy=ChunksProntosDoOrientAI())
        knowledge.add_content(
            name="Documentos do IFSC",
            path=CHUNKS_FILE,
            reader=reader,
        )
        os.makedirs(DB_PATH, exist_ok=True)
        with open(hash_file, "w", encoding="utf-8") as f:
            f.write(hash_atual)

    return Agent(
        model=Gemini(id=GEMINI_MODEL),
        knowledge=knowledge,
        search_knowledge=True,
        instructions=[
            "Voce e o OrientAI, assistente virtual dos estudantes do IFSC.",
            "Tom: amigavel, acolhedor e direto — como um veterano ajudando um calouro, sem ser formal demais.",
            "REGRA DE SAUDAÇÃO: Voce JA se apresentou na mensagem inicial do chat. NUNCA diga 'Oi, eu sou o OrientAI' ou se apresente novamente. Va direto ao ponto da resposta.",
            "FOCO E ESCOPO: Baseie suas respostas EXCLUSIVAMENTE nos documentos recuperados da base de conhecimento do IFSC. Se perguntarem algo pessoal (ex: 'o que voce gosta de fazer') ou fora dos documentos, responda em 1 frase curta e amigavel que seu foco e ajudar com as regras e documentos do IFSC, sem inventar funcoes que nao tem.",
            "Se a informacao nao estiver nos documentos, diga isso claramente de forma simpatica, sem inventar.",
            "Quando encontrar a resposta nos documentos, cite o documento, capitulo ou artigo correspondente.",
            "Use o historico da conversa fornecido para manter o contexto e responder perguntas sobre mensagens anteriores.",
        ],
    )


MENSAGEM_BOAS_VINDAS = (
    "Oi! Eu sou o **OrientAI** 🎓, o assistente virtual dos estudantes do IFSC.\n\n"
    "Posso te ajudar com dúvidas sobre o Código de Convivência Discente e outros documentos "
    "institucionais — direitos, deveres, o que pode e o que não pode, medidas disciplinares, "
    "bolsas e mais.\n\n"
    "Manda sua pergunta aí embaixo!"
)


def formatar_historico(mensagens, limite=10):
    """Monta um resumo das ultimas mensagens para o agente ter memoria real da conversa."""
    recentes = mensagens[-limite:]
    linhas = []
    for m in recentes:
        papel = "Estudante" if m["role"] == "user" else "OrientAI"
        linhas.append(f"{papel}: {m['content']}")
    return "\n".join(linhas)


def responder_com_retry(agent, historico_mensagens, pergunta_atual):
    contexto_conversa = formatar_historico(historico_mensagens[:-1], limite=10)
    prompt_completo = (
        f"Histórico recente desta conversa:\n{contexto_conversa}\n\n"
        f"Nova mensagem do estudante: {pergunta_atual}"
    )

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = agent.run(prompt_completo)
            if resposta and resposta.content:
                return resposta.content
        except Exception as e:
            print(f"[Tentativa {tentativa}] Erro: {e}")
            if tentativa < MAX_TENTATIVAS:
                time.sleep(1.5)
    return "Tive uma falha temporaria pra buscar essa informacao. Tenta perguntar de novo em alguns segundos."


st.set_page_config(page_title="OrientAI", page_icon="🎓", layout="centered")
st.title("🎓 OrientAI")
st.caption("Assistente baseado nos documentos institucionais do IFSC")

if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    st.error('Defina sua chave no terminal antes de rodar: `$env:GOOGLE_API_KEY="sua-chave-aqui"`')
    st.stop()

if not os.path.exists(CHUNKS_FILE):
    st.error(f"Arquivo '{CHUNKS_FILE}' nao encontrado nessa pasta.")
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
            resposta = responder_com_retry(agent, st.session_state.mensagens, pergunta)
        st.markdown(resposta)

    st.session_state.mensagens.append({"role": "assistant", "content": resposta})
