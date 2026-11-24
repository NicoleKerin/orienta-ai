from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
import shutil
import time

# Evita checar internet no Hugging Face se o modelo já foi baixado antes
if os.path.exists(os.path.expanduser("~/.cache/huggingface/hub")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import streamlit as st

CHUNKS_FILE = "chunks.json"
DB_PATH = "./chroma_db"
LOCK_FILE = os.path.join(DB_PATH, ".indexing.lock")
HASH_FILE = os.path.join(DB_PATH, "chunks.sha256")
COLLECTION_NAME = "orientai"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

MENSAGEM_BOAS_VINDAS = (
    "Oi! Eu sou o **OrientAI** 🎓, o assistente virtual dos estudantes do IFSC.\n\n"
    "Posso te ajudar com dúvidas sobre o Código de Convivência Discente e outros documentos "
    "institucionais — direitos, deveres, o que pode e o que não pode, medidas disciplinares, "
    "bolsas e mais.\n\n"
    "Manda sua pergunta aí embaixo!"
)


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


def carregar_chunks(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    if not isinstance(dados, list):
        raise ValueError(f"Formato inválido em '{caminho}': esperado uma lista de chunks.")
    return dados


@contextmanager
def arquivo_lock(lock_path, timeout=60):
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    inicio = time.time()
    while os.path.exists(lock_path):
        if time.time() - inicio > timeout:
            raise TimeoutError(f"Tempo esgotado aguardando lock de indexação: {lock_path}")
        time.sleep(0.25)

    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        yield
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def importar_componentes_agno():
    classes = {
        "Knowledge": find_class(["agno.knowledge.knowledge", "agno.knowledge"], "Knowledge"),
        "ChromaDb": find_class(["agno.vectordb.chroma", "agno.vectordb.chroma.chroma_db"], "ChromaDb"),
        "SentenceTransformerEmbedder": find_class(
            ["agno.embedder.sentence_transformer", "agno.knowledge.embedder.sentence_transformer"],
            "SentenceTransformerEmbedder",
        ),
        "ChunkingStrategy": find_class(
            ["agno.knowledge.chunking.base", "agno.knowledge.chunking.strategy"], "ChunkingStrategy"
        ),
        "Document": find_class(["agno.knowledge.document", "agno.knowledge.document.base"], "Document"),
        "TextReader": find_class(["agno.knowledge.reader.text_reader"], "TextReader"),
    }
    faltando = [nome for nome, cls in classes.items() if cls is None]
    if faltando:
        raise ImportError(f"Não foi possível importar do agno: {', '.join(faltando)}")
    return classes


@st.cache_resource(show_spinner="Carregando base de conhecimento do OrientAI...")
def montar_base_conhecimento(chunks_path=CHUNKS_FILE):
    agno_cls = importar_componentes_agno()
    ChunkingStrategy = agno_cls["ChunkingStrategy"]
    Document = agno_cls["Document"]
    Knowledge = agno_cls["Knowledge"]
    ChromaDb = agno_cls["ChromaDb"]
    SentenceTransformerEmbedder = agno_cls["SentenceTransformerEmbedder"]
    TextReader = agno_cls["TextReader"]

    chunks_dados = carregar_chunks(chunks_path)
    hash_atual = calcular_hash_arquivo(chunks_path)

    class ChunksProntosDoOrientAI(ChunkingStrategy):
        def __init__(self, chunks):
            super().__init__()
            self._chunks = chunks

        def chunk(self, document):
            return [
                Document(
                    content=c["text"],
                    meta_data={
                        "source_file": c.get("source_file", "Documento IFSC"),
                        "header_path": " > ".join(c.get("header_path", [])),
                        "chunk_id": c.get("id", idx),
                    },
                )
                for idx, c in enumerate(self._chunks)
            ]

    with arquivo_lock(LOCK_FILE):
        hash_salvo = None
        if os.path.exists(HASH_FILE):
            with open(HASH_FILE, "r", encoding="utf-8") as f:
                hash_salvo = f.read().strip()

        precisa_indexar = hash_atual != hash_salvo

        if precisa_indexar and os.path.exists(DB_PATH):
            for entry in os.listdir(DB_PATH):
                alvo = os.path.join(DB_PATH, entry)
                if alvo == LOCK_FILE:
                    continue
                if os.path.isdir(alvo):
                    shutil.rmtree(alvo)
                else:
                    os.remove(alvo)

        knowledge = Knowledge(
            vector_db=ChromaDb(
                collection=COLLECTION_NAME,
                embedder=SentenceTransformerEmbedder(id=EMBEDDING_MODEL),
                path=DB_PATH,
            ),
        )

        if precisa_indexar:
            reader = TextReader(chunking_strategy=ChunksProntosDoOrientAI(chunks_dados))
            knowledge.add_content(
                name="Documentos do IFSC",
                path=chunks_path,
                reader=reader,
            )
            os.makedirs(DB_PATH, exist_ok=True)
            with open(HASH_FILE, "w", encoding="utf-8") as f:
                f.write(hash_atual)

    return knowledge


def buscar_contexto(knowledge, pergunta, limite=5):
    try:
        if hasattr(knowledge, "search"):
            docs = knowledge.search(query=pergunta, max_results=limite)
        elif hasattr(knowledge, "vector_db") and hasattr(knowledge.vector_db, "search"):
            docs = knowledge.vector_db.search(query=pergunta, limit=limite)
        else:
            docs = []

        docs = docs or []
        print(f"[DEBUG BUSCA] Encontrados: {len(docs)} documentos para '{pergunta}'")

        if not docs:
            return ""

        trechos = []
        for i, d in enumerate(docs, 1):
            meta = getattr(d, "meta_data", None) or {}
            fonte = meta.get("source_file", "Documento IFSC")
            secao = meta.get("header_path", "")
            conteudo = getattr(d, "content", "")
            trechos.append(f"[Trecho {i} | Fonte: {fonte} | Seção: {secao}]\n{conteudo}")

        return "\n\n---\n\n".join(trechos)
    except Exception as e:
        print(f"[DEBUG BUSCA] Erro na busca: {e}")
        return ""


def criar_agente_com_modelo(model_id):
    from agno.agent import Agent

    Gemini = find_class(["agno.models.google", "agno.models.google.gemini"], "Gemini")
    if Gemini is None:
        raise ImportError("Não foi possível importar a classe Gemini do agno.")

    data_hoje = datetime.now().strftime("%d/%m/%Y")

    return Agent(
        model=Gemini(id=model_id),
        instructions=[
            f"Data atual: {data_hoje}. Use essa data como referência para calcular prazos, anos ou idade da instituição.",
            "Você é o OrientAI, assistente virtual dos estudantes do IFSC.",
            "Tom: amigável, acolhedor e direto — como um veterano ajudando um calouro, sem ser formal demais.",
            "REGRA DE SAUDAÇÃO: Você JÁ se apresentou na mensagem inicial do chat. NUNCA diga 'Oi, eu sou o OrientAI' ou se apresente novamente. Vá direto ao ponto da resposta.",
            "FOCO E ESCOPO: Baseie suas respostas EXCLUSIVAMENTE nos trechos de documentos fornecidos no prompt. Se perguntarem algo pessoal ou fora dos documentos, responda em 1 frase curta e amigável que seu foco é ajudar com as regras e documentos do IFSC, sem inventar funções que não tem.",
            "Se a informação não estiver nos trechos fornecidos, diga isso claramente de forma simpática, sem inventar.",
            "Quando encontrar a resposta nos trechos, cite o documento, capítulo ou artigo correspondente.",
            "Use o histórico da conversa fornecido para manter o contexto e responder perguntas sobre mensagens anteriores.",
        ],
    )


def formatar_historico(mensagens, limite=10):
    recentes = mensagens[-limite:]
    linhas = []
    for m in recentes:
        papel = "Estudante" if m["role"] == "user" else "OrientAI"
        linhas.append(f"{papel}: {m['content']}")
    return "\n".join(linhas)


def _parece_erro(texto):
    if not texto or not texto.strip():
        return True
    marcadores = (
        '"error":',
        "'error':",
        "RESOURCE_EXHAUSTED",
        "UNAVAILABLE",
        "invalid_request_error",
        "Error from Gemini API",
        "HTTP 429",
        "HTTP 503",
    )
    return any(m in texto for m in marcadores)


def responder_com_fallback(knowledge, historico_mensagens, pergunta_atual):
    contexto_conversa = formatar_historico(historico_mensagens[:-1], limite=6)
    trechos_docs = buscar_contexto(knowledge, pergunta_atual, limite=5)

    prompt_completo = (
        f"Trechos recuperados dos documentos oficiais do IFSC:\n"
        f"{trechos_docs if trechos_docs else 'Nenhum trecho encontrado na base.'}\n\n"
        f"Histórico recente desta conversa:\n{contexto_conversa}\n\n"
        f"Nova mensagem do estudante: {pergunta_atual}"
    )

    for model_id in GEMINI_MODELS:
        try:
            agent = criar_agente_com_modelo(model_id)
        except Exception as e:
            print(f"[Fallback] Falha ao instanciar agente para {model_id}: {e}")
            continue

        for tentativa in range(2):
            try:
                resposta = agent.run(prompt_completo)
                texto = resposta.content if resposta else ""

                if _parece_erro(texto):
                    print(f"[Fallback] {model_id} (tentativa {tentativa + 1}) retornou erro. Aguardando...")
                    time.sleep(2)
                    continue

                return texto
            except Exception as e:
                print(f"[Fallback] {model_id} (tentativa {tentativa + 1}) falhou: {e}")
                time.sleep(2)

    return (
        "Os servidores do Gemini estão com instabilidade temporária ou atingiram o limite gratuito agora 😅. "
        "Tenta novamente em alguns segundos!"
    )


def main():
    st.set_page_config(page_title="OrientAI", page_icon="🎓", layout="centered")
    st.title("🎓 OrientAI")
    st.caption("Assistente baseado nos documentos institucionais do IFSC")

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        st.error('Defina sua chave no terminal antes de rodar: `$env:GOOGLE_API_KEY="sua-chave-aqui"`')
        st.stop()

    if not os.path.exists(CHUNKS_FILE):
        st.error(f"Arquivo '{CHUNKS_FILE}' não encontrado nessa pasta.")
        st.stop()

    try:
        knowledge = montar_base_conhecimento(CHUNKS_FILE)
    except Exception as e:
        st.error(f"Erro ao preparar a base de conhecimento: {e}")
        st.stop()

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
                resposta = responder_com_fallback(knowledge, st.session_state.mensagens, pergunta)
            st.markdown(resposta)

        st.session_state.mensagens.append({"role": "assistant", "content": resposta})


if __name__ == "__main__":
    main()
