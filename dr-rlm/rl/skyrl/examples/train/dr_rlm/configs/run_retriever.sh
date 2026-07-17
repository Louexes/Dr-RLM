set -x

# =============================================================================
# DR-RLM offline corpus RETRIEVER  (held constant across L0..L4)
# -----------------------------------------------------------------------------
# Both the flat (L0/L1) and recursive (L2-L4) policies retrieve from the SAME
# frozen corpus with the SAME retriever (proposal §6, held constant). The REPL
# search()/get_doc() tools live in corpus_search.py; this script documents the two
# supported ways to back them. Pick ONE and set generator.search_backend to match
# in the run_dr_rlm_L*.sh scripts.
#
# corpus.jsonl schema (DR Tulu / wii): one JSON object per line, {id, contents, url}.
# Build it with your data-convert step (same corpus for every arm).
# =============================================================================

: "${DATA_DIR:=$HOME/data/dr-rlm}"
: "${CORPUS_PATH:=$DATA_DIR/corpus.jsonl}"
: "${INDEX_PATH:=$DATA_DIR/bm25}"
: "${MCP_PORT:=8003}"
: "${MCP_HOST:=0.0.0.0}"
# dr_agent lives outside SkyRL; corpus_search.py imports it from this path in-process.
: "${DR_TULU_AGENT:=/gpfs/home5/lgehringer/Dr-RLM/dr-tulu/agent}"

MODE="${1:-info}"   # one of: info | mcp | build-bm25

# -----------------------------------------------------------------------------
# OPTION A — IN-PROCESS bm25 / faiss / local_jsonl  (NO server to launch)
# -----------------------------------------------------------------------------
# corpus_search.py imports the dr_agent BM25Searcher / FaissSearcher directly into
# each rollout worker via get_backend(); there is NOTHING to start here. Just set in
# the training scripts (export before launching, or pass on the CLI):
#
#   # BM25 (real wii retrieval; needs Java on PATH for pyserini's Lucene index):
#   export SEARCH_BACKEND=bm25
#   export SEARCH_CORPUS_PATH=$DATA_DIR/corpus.jsonl
#   export SEARCH_INDEX_PATH=$DATA_DIR/bm25
#   #   -> generator.search_backend=bm25 generator.search_index_path=data/bm25
#   #      generator.search_corpus_path=data/corpus.jsonl
#   # NOTE: BM25Searcher uses pyserini/Lucene -> a JVM must be reachable:
#   #   module load Java/17  (or: export JAVA_HOME=/path/to/jdk; ensure `java` on PATH)
#   # If the dr_agent import or Java is missing, get_backend() falls back to local_jsonl.
#
#   # FAISS (dense Qwen3-Embedding retrieval; needs the embed model + a prebuilt index):
#   export SEARCH_BACKEND=faiss
#   export SEARCH_INDEX_PATH=$DATA_DIR/qwen3-8b           # corpus.pkl* prebuilt index dir
#   #   -> generator.search_backend=faiss generator.search_embed_model=Qwen/Qwen3-Embedding-8B
#
#   # local_jsonl (dependency-free TF-IDF-lite over corpus.jsonl; smoke tests / no Java):
#   export SEARCH_BACKEND=local_jsonl
#   export SEARCH_CORPUS_PATH=$DATA_DIR/corpus.jsonl
#   #   -> generator.search_backend=local_jsonl generator.search_corpus_path=data/corpus.jsonl
#
# This script can also (re)build the pyserini BM25 index for Option A:
#   bash run_retriever.sh build-bm25
if [ "$MODE" = "build-bm25" ]; then
  # Lucene index over the corpus, consumed by dr_agent BM25Searcher (Option A / bm25).
  # Requires Java on PATH (module load Java/17) and pyserini installed in the env.
  export JAVA_HOME="${JAVA_HOME:-$JAVA_HOME}"
  uv run python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$(dirname "$CORPUS_PATH")" \
    --index "$INDEX_PATH" \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 --storePositions --storeDocvectors --storeRaw
  exit $?
fi

# -----------------------------------------------------------------------------
# OPTION B — dr_agent FastMCP server over HTTP  (generator.search_backend=mcp_http)
# -----------------------------------------------------------------------------
# Run a standalone dr_agent MCP server; corpus_search.py's _McpHttpBackend calls its
# `local_search` tool over StreamableHTTP. Command/flags copied verbatim from
# dr-tulu/rl/open-instruct/train_local/train_dr_tulu_mini_base_local_bm25.sh
# (--mcp_server_command). Set in the training scripts:
#   export SEARCH_BACKEND=mcp_http
#   export SEARCH_ENDPOINT=http://localhost:$MCP_PORT/mcp
#   #   -> generator.search_backend=mcp_http generator.search_endpoint=http://localhost:8003/mcp
if [ "$MODE" = "mcp" ]; then
  # Same env the dr-tulu reference exports for its local MCP server.
  export USE_LOCAL_SEARCH=true
  export MCP_MAX_CONCURRENT_CALLS="${MCP_MAX_CONCURRENT_CALLS:-512}"
  export MCP_TRANSPORT_PORT="$MCP_PORT"
  export MCP_CACHE_DIR="${MCP_CACHE_DIR:-.cache-mcp-$RANDOM}"
  # run from the dr_agent package root so `-m dr_agent.mcp_backend.main` resolves
  (cd "$DR_TULU_AGENT" && uv run python -m dr_agent.mcp_backend.main \
    --transport http \
    --port "$MCP_PORT" \
    --host "$MCP_HOST" \
    --path /mcp \
    --local-searcher-type bm25 \
    --index-path "$INDEX_PATH" \
    --dataset-name "$CORPUS_PATH")
  exit $?
fi

# -----------------------------------------------------------------------------
# default: print usage
# -----------------------------------------------------------------------------
cat <<EOF
DR-RLM retriever — pick ONE backend (held constant across L0..L4):

  Option A (in-process, NO server):  set generator.search_backend to one of
      bm25 | faiss | local_jsonl   and point search_corpus_path/search_index_path
      at the frozen corpus.  Build the bm25 index with:  bash run_retriever.sh build-bm25
      (bm25 needs Java on PATH for pyserini; faiss needs Qwen3-Embedding + a prebuilt index;
       local_jsonl needs nothing). Nothing to launch — corpus_search.get_backend() loads it
       in each rollout worker.

  Option B (server):  bash run_retriever.sh mcp
      launches the dr_agent FastMCP server on :$MCP_PORT/mcp; then set
      generator.search_backend=mcp_http generator.search_endpoint=http://localhost:$MCP_PORT/mcp

Current paths: CORPUS_PATH=$CORPUS_PATH  INDEX_PATH=$INDEX_PATH
EOF
