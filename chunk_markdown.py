
import os
import re
import json
import glob

INPUT_DIR = "markdowns"
OUTPUT_FILE = "chunks.json"

MAX_CHUNK_SIZE = 600     # tamanho maximo do chunk FINAL, ja incluindo o prefixo [Capitulo > Secao]
MIN_CHUNK_SIZE = 150
OVERLAP_SIZE = 80

PLACEHOLDER_RE = re.compile(r"__BLOCK_\d+__")


def protect_special_blocks(text):
    blocks = {}
    counter = [0]

    def make_placeholder(match):
        key = f"__BLOCK_{counter[0]}__"
        blocks[key] = match.group(0)
        counter[0] += 1
        return key

    text = re.sub(r"```[\s\S]*?```", make_placeholder, text)

    def protect_tables(text):
        lines = text.split("\n")
        result = []
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("|"):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1

                header_lines = table_lines[:2] if len(table_lines) >= 2 else []
                body_lines = table_lines[2:] if len(table_lines) >= 2 else table_lines
                full_table_text = "\n".join(table_lines)

                if len(full_table_text) <= MAX_CHUNK_SIZE or not body_lines:
                    key = f"__BLOCK_{counter[0]}__"
                    blocks[key] = full_table_text
                    counter[0] += 1
                    result.append(key)
                else:
                    header_text = "\n".join(header_lines)
                    current_rows = []
                    for row in body_lines:
                        candidate_rows = current_rows + [row]
                        candidate_text = header_text + "\n" + "\n".join(candidate_rows) if header_text else "\n".join(candidate_rows)
                        if len(candidate_text) > MAX_CHUNK_SIZE and current_rows:
                            sub_table_text = header_text + "\n" + "\n".join(current_rows) if header_text else "\n".join(current_rows)
                            key = f"__BLOCK_{counter[0]}__"
                            blocks[key] = sub_table_text
                            counter[0] += 1
                            result.append(key)
                            result.append("")
                            current_rows = [row]
                        else:
                            current_rows = candidate_rows
                    if current_rows:
                        sub_table_text = header_text + "\n" + "\n".join(current_rows) if header_text else "\n".join(current_rows)
                        key = f"__BLOCK_{counter[0]}__"
                        blocks[key] = sub_table_text
                        counter[0] += 1
                        result.append(key)
            else:
                result.append(lines[i])
                i += 1
        return "\n".join(result)

    text = protect_tables(text)
    return text, blocks


def restore_special_blocks(text, blocks):
    for key, original in blocks.items():
        text = text.replace(key, original)
    return text


# ============================================================
# DIVISAO POR HEADER
# ============================================================

def split_by_headers(markdown_text):
    lines = markdown_text.split("\n")
    sections = []
    current_header_stack = []
    current_content = []

    header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

    def flush_section():
        content = "\n".join(current_content).strip()
        if content:
            header_path = [title for _, title in current_header_stack]
            level = current_header_stack[-1][0] if current_header_stack else 0
            sections.append({
                "header_path": header_path.copy(),
                "level": level,
                "content": content
            })

    for line in lines:
        match = header_pattern.match(line)
        if match:
            flush_section()
            current_content = []
            level = len(match.group(1))
            title = match.group(2).strip()
            while current_header_stack and current_header_stack[-1][0] >= level:
                current_header_stack.pop()
            current_header_stack.append((level, title))
        else:
            current_content.append(line)

    flush_section()
    return sections


def real_size(text, blocks):
    size = len(text)
    for key, original in blocks.items():
        if key in text:
            size += (len(original) - len(key)) * text.count(key)
    return size


def safe_cut_point(text, target):
    """Garante que o corte nunca caia no meio de um placeholder __BLOCK_N__."""
    for m in PLACEHOLDER_RE.finditer(text):
        if m.start() < target < m.end():
            return m.start()
    return target


def snap_to_word_boundary(text, index):
    """
    A partir de `index`, anda para tras ate encontrar um espaco em
    branco, evitando cortar uma palavra ao meio (ex: "Federal" -> "ral").
    Se nao encontrar espaco nenhum antes (palavra unica gigante), mantem
    o index original.
    """
    if index <= 0 or index >= len(text):
        return index
    original = index
    while index > 0 and not text[index].isspace():
        index -= 1
    return index if index > 0 else original


def safe_boundary(text, target):
    """Combina as duas protecoes: nunca corta um placeholder nem uma palavra."""
    point = safe_cut_point(text, target)
    point = snap_to_word_boundary(text, point)
    return point


# ============================================================
# DIVISAO POR TAMANHO
# ============================================================

def split_text_by_size(text, max_size, overlap, blocks=None, prefix_len=0):
    """
    max_size ja deve levar em conta o espaco reservado para qualquer
    prefixo que sera adicionado depois (ex: "[Capitulo > Secao]\n"),
    entao subtraimos prefix_len do limite disponivel para o conteudo.
    """
    blocks = blocks or {}
    effective_max = max(max_size - prefix_len, 100)  # nunca deixa o limite efetivo ficar ridiculo

    if real_size(text, blocks) <= effective_max:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for para in paragraphs:
        is_protected_block = para.strip() in blocks
        candidate = (current + "\n\n" + para).strip() if current else para

        if real_size(candidate, blocks) <= effective_max:
            current = candidate
        else:
            if current:
                chunks.append(current)
                cut = safe_boundary(current, max(len(current) - overlap, 0))
                overlap_text = current[cut:] if len(current) > overlap else current
                current = (overlap_text + "\n\n" + para).strip()
            else:
                current = para

            if is_protected_block:
                chunks.append(current)
                current = ""
                continue

            while real_size(current, blocks) > effective_max:
                cut_point = safe_boundary(current, effective_max)
                while real_size(current[:cut_point], blocks) > effective_max and cut_point > 0:
                    cut_point = safe_boundary(current, cut_point - 50)
                if cut_point <= 0:
                    cut_point = effective_max
                chunks.append(current[:cut_point])
                next_start = safe_boundary(current, max(cut_point - overlap, 0))
                current = current[next_start:]

    if current:
        chunks.append(current)

    return chunks


def merge_small_sections(sections, min_size, blocks=None):
    """
    Funde secoes pequenas com a seguinte, MAS SO se ambas tiverem um
    pai comum e nao-vazio no header_path (evita fundir secoes de
    nivel 1 sem relacao nenhuma).
    """
    blocks = blocks or {}
    merged = []
    buffer = None

    for section in sections:
        if buffer is None:
            buffer = section.copy()
            continue

        parent_a = buffer["header_path"][:-1]
        parent_b = section["header_path"][:-1]
        same_branch = len(parent_a) > 0 and parent_a == parent_b

        if real_size(buffer["content"], blocks) < min_size and same_branch:
            buffer["content"] += "\n\n" + section["content"]
        else:
            merged.append(buffer)
            buffer = section.copy()

    if buffer is not None:
        merged.append(buffer)

    return merged


# ============================================================
# PROCESSAMENTO POR ARQUIVO
# ============================================================

def chunk_markdown_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    protected_text, blocks = protect_special_blocks(raw_text)
    sections = split_by_headers(protected_text)
    sections = merge_small_sections(sections, MIN_CHUNK_SIZE, blocks)

    filename = os.path.basename(filepath)
    chunks = []
    chunk_index = 0

    for section in sections:
        header_path = section["header_path"]
        header_path_str = " > ".join(header_path) if header_path else "(sem secao)"
        # tamanho do prefixo que sera somado a cada pedaco dessa secao
        prefix = f"[{header_path_str}]\n" if header_path else ""
        prefix_len = len(prefix)

        pieces = split_text_by_size(section["content"], MAX_CHUNK_SIZE, OVERLAP_SIZE, blocks, prefix_len)

        for piece in pieces:
            restored_text = restore_special_blocks(piece, blocks)
            full_text = f"{prefix}{restored_text.strip()}" if prefix else restored_text.strip()

            chunks.append({
                "id": f"{filename}::chunk_{chunk_index}",
                "source_file": filename,
                "header_path": header_path,
                "text": full_text,
                "char_count": len(full_text)
            })
            chunk_index += 1

    return chunks


def main():
    md_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.md"), recursive=True)

    if not md_files:
        print(f"Nenhum arquivo .md encontrado em '{INPUT_DIR}'.")
        print("Verifique o caminho da pasta em INPUT_DIR no topo do script.")
        return

    all_chunks = []
    for filepath in md_files:
        try:
            file_chunks = chunk_markdown_file(filepath)
        except (OSError, UnicodeDecodeError) as e:
            print(f"  ERRO ao ler {os.path.basename(filepath)}: {e}")
            continue

        all_chunks.extend(file_chunks)
        print(f"  {os.path.basename(filepath)}: {len(file_chunks)} chunks")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(all_chunks)} chunks de {len(md_files)} arquivo(s).")
    print(f"Salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()