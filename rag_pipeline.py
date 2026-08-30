"""
Module 5: RAGPipeline

Chunks and indexes financial news articles (from FNSPID, via NewsIngestion)
into FAISS using a free sentence-embedding model. Retrieves top-k passages
for a user question and generates a grounded, source-cited answer using
Groq - never OpenAI/ChatOpenAI. See config.py for why the model is Qwen
rather than Llama (every Llama model on Groq's free tier turned out to be
either decommissioned or unavailable to this project's account).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.documents import Document

import config

if TYPE_CHECKING:
    # Only imported for type-checkers (Pylance/mypy); the real imports are
    # local to each method at runtime so the heavier langchain_community/groq
    # dependencies stay lazily-loaded for anything that doesn't touch
    # embedding/generation (e.g. the test suite, chunk_articles-only usage).
    from groq.types.chat import ChatCompletionMessageParam
    from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer the user's question using "
    "ONLY the information in the provided article excerpts. If the excerpts do "
    "not contain enough information to answer, say so plainly instead of "
    "guessing. Cite which article(s) you used by their [n] marker in your answer."
)

# `Document` re-exported above is langchain_core's real class (not a
# duck-typed stand-in) - FAISS.from_documents() does an isinstance-adjacent
# structural check via type annotations, so a lookalike class fails static
# type-checking even though it works at runtime. langchain_core is a light
# dependency (no heavy ML libs pulled in), so importing it directly here
# costs nothing but fixes that mismatch.

# Hand-labeled evaluation query set for RAGPipeline.evaluate_retrieval_precision().
# Each entry is a natural question a real user might ask, manually labeled with
# which ticker's articles it should retrieve. This is a proxy for true semantic
# relevance (grading whether every retrieved chunk actually answers the
# question would need a human judge per chunk, which doesn't scale) - it
# measures whether retrieval finds passages about the *right company* at all,
# which is a real, honest, and necessary (if not sufficient) condition for a
# useful answer. Documented as such wherever the resulting metric is reported.
EVAL_QUERIES: list[dict[str, str]] = [
    {"question": "What is the latest news on Apple's stock?", "expected_ticker": "AAPL"},
    {"question": "How is Apple performing this quarter?", "expected_ticker": "AAPL"},
    {"question": "Is Apple a good stock to buy right now?", "expected_ticker": "AAPL"},
    {"question": "What are analysts saying about Apple shares?", "expected_ticker": "AAPL"},
    {"question": "What is the recent news on Microsoft stock?", "expected_ticker": "MSFT"},
    {"question": "How has Microsoft's share price been moving?", "expected_ticker": "MSFT"},
    {"question": "What do investors think about Microsoft right now?", "expected_ticker": "MSFT"},
    {"question": "Any updates on Microsoft's business performance?", "expected_ticker": "MSFT"},
    {"question": "What is happening with Google's stock lately?", "expected_ticker": "GOOGL"},
    {"question": "How is Alphabet performing in the market?", "expected_ticker": "GOOGL"},
    {"question": "What are the latest headlines about Google shares?", "expected_ticker": "GOOGL"},
    {"question": "Is Google stock a buy according to recent news?", "expected_ticker": "GOOGL"},
    {"question": "What is the latest news about Amazon stock?", "expected_ticker": "AMZN"},
    {"question": "How is Amazon's stock price trending recently?", "expected_ticker": "AMZN"},
    {"question": "What are analysts forecasting for Amazon shares?", "expected_ticker": "AMZN"},
    {"question": "Any recent updates on Amazon's business outlook?", "expected_ticker": "AMZN"},
    {"question": "What is the recent news about Tesla stock?", "expected_ticker": "TSLA"},
    {"question": "How has Tesla's stock been performing lately?", "expected_ticker": "TSLA"},
    {"question": "What are analysts saying about Tesla shares?", "expected_ticker": "TSLA"},
    {"question": "Is Tesla stock expected to rise or fall soon?", "expected_ticker": "TSLA"},
]


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.RETRIEVAL_TOP_K,
    ):
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.vectorstore: FAISS | None = None
        self.indexed_passage_count = 0

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    def chunk_articles(self, articles: pd.DataFrame) -> list[Document]:
        """articles must have columns: headline, body, source, date, ticker.

        Splits each article body into overlapping character chunks using
        LangChain's RecursiveCharacterTextSplitter, tagging each chunk with
        the source article's metadata (needed for citation later).
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        docs: list[Document] = []
        # enumerate() gives a real int article id regardless of the
        # DataFrame's index dtype/labels (iterrows()'s index is typed
        # Hashable, not guaranteed int-convertible - e.g. after a filter or
        # concat the index could be non-sequential or non-numeric).
        for article_id, (_, row) in enumerate(articles.iterrows()):
            body = str(row.get("body") or "").strip()
            headline = str(row.get("headline") or "").strip()
            full_text = f"{headline}\n\n{body}".strip()
            if not full_text:
                continue
            chunks = splitter.split_text(full_text)
            for j, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "article_id": article_id,
                        "chunk_id": j,
                        "headline": headline,
                        "source": row.get("source"),
                        "date": row.get("date"),
                        "ticker": row.get("ticker"),
                    },
                ))
        return docs

    # ------------------------------------------------------------------
    # Embedding + indexing
    # ------------------------------------------------------------------
    def build_index(self, articles: pd.DataFrame, force_rebuild: bool = False):
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        index_path = config.FAISS_INDEX_DIR
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            # transformers' default "low_cpu_mem_usage" load path memory-maps
            # the safetensors weights via a meta-device trick that reliably
            # hits "OSError: paging file is too small" on Windows with a
            # default-sized pagefile, even for a tiny (~90MB) model. Disabling
            # it forces a normal (still fast, model is small) load instead.
            model_kwargs={"model_kwargs": {"low_cpu_mem_usage": False}},
        )

        if not force_rebuild and (index_path / "index.faiss").exists():
            self.vectorstore = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True,
            )
            self.indexed_passage_count = self.vectorstore.index.ntotal
            logger.info("Loaded existing FAISS index (%d passages)", self.indexed_passage_count)
            return self.vectorstore

        docs = self.chunk_articles(articles)
        if not docs:
            raise ValueError("No chunks produced from articles - nothing to index")

        self.vectorstore = FAISS.from_documents(docs, embeddings)
        self.vectorstore.save_local(str(index_path))
        self.indexed_passage_count = len(docs)
        logger.info("Built FAISS index with %d passages from %d articles", len(docs), len(articles))
        return self.vectorstore

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> list[Document]:
        if self.vectorstore is None:
            raise RuntimeError("Index not built yet - call build_index() first")
        k = k or self.top_k
        return self.vectorstore.similarity_search(question, k=k)

    # ------------------------------------------------------------------
    # Generation (Groq / Llama)
    # ------------------------------------------------------------------
    def _format_context(self, docs: list[Document]) -> str:
        parts = []
        for i, d in enumerate(docs, start=1):
            headline = d.metadata.get("headline") or "(no headline)"
            date = d.metadata.get("date") or "unknown date"
            ticker = d.metadata.get("ticker") or "?"
            parts.append(f"[{i}] ({ticker}, {date}) {headline}\n{d.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def format_citations(docs: list[Document]) -> list[dict]:
        return [
            {
                "marker": f"[{i}]",
                "headline": d.metadata.get("headline"),
                "source": d.metadata.get("source"),
                "date": d.metadata.get("date"),
                "ticker": d.metadata.get("ticker"),
                "snippet": d.page_content[:300],
            }
            for i, d in enumerate(docs, start=1)
        ]

    def generate_answer(self, question: str, docs: list[Document]) -> str | None:
        """Call Groq, grounded on the retrieved passages only."""
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                "and put it in your .env file."
            )
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
        context = self._format_context(docs)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Article excerpts:\n\n{context}\n\nQuestion: {question}"},
        ]
        # qwen/qwen3.6-27b (the default GROQ_MODEL) is a reasoning model that
        # otherwise prepends a <think>...</think> trace to its response.
        # Confirmed by testing: Groq rejects this parameter outright
        # ("reasoning_effort is not supported with this model") on
        # non-reasoning models, so it's only included for models known to
        # accept it - not passed unconditionally.
        if config.GROQ_MODEL.startswith("qwen/"):
            response = client.chat.completions.create(
                model=config.GROQ_MODEL, messages=messages, temperature=0.1,
                reasoning_effort=config.GROQ_REASONING_EFFORT,
            )
        else:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL, messages=messages, temperature=0.1,
            )
        return response.choices[0].message.content

    def answer(self, question: str, k: int | None = None) -> dict:
        docs = self.retrieve(question, k=k)
        if not docs:
            return {
                "question": question,
                "answer": "No relevant passages were found in the indexed corpus for this question.",
                "citations": [],
            }
        answer_text = self.generate_answer(question, docs)
        return {
            "question": question,
            "answer": answer_text,
            "citations": self.format_citations(docs),
        }

    # ------------------------------------------------------------------
    # Evaluation: retrieval precision@k
    # ------------------------------------------------------------------
    def evaluate_retrieval_precision(
        self, queries: list[dict[str, str]] | None = None, k: int | None = None,
    ) -> dict:
        """Ticker-match precision@k over a hand-labeled query set (see EVAL_QUERIES).

        For each query, retrieves the top-k passages and measures what fraction
        are tagged with the query's expected ticker. This is a proxy for true
        relevance (see the EVAL_QUERIES docstring above for why), reported as
        such - not "the RAG pipeline is X% accurate" in general, specifically
        "X% of retrieved passages were about the right company."
        """
        queries = queries if queries is not None else EVAL_QUERIES
        k = k or self.top_k

        per_query = []
        for item in queries:
            docs = self.retrieve(item["question"], k=k)
            if not docs:
                precision = 0.0
            else:
                matches = sum(1 for d in docs if d.metadata.get("ticker") == item["expected_ticker"])
                precision = matches / len(docs)
            per_query.append({
                "question": item["question"],
                "expected_ticker": item["expected_ticker"],
                "retrieved_tickers": [d.metadata.get("ticker") for d in docs],
                "precision_at_k": precision,
            })

        mean_precision = sum(q["precision_at_k"] for q in per_query) / len(per_query) if per_query else 0.0
        return {
            "k": k,
            "n_queries": len(per_query),
            "mean_precision_at_k": mean_precision,
            "per_query": per_query,
        }

    # ------------------------------------------------------------------
    # Evaluation: end-to-end latency
    # ------------------------------------------------------------------
    def benchmark_latency(
        self,
        questions: list[str] | None = None,
        k: int | None = None,
        delay_between_calls_sec: float = 2.0,
        max_retries_per_call: int = 2,
    ) -> dict:
        """Time full end-to-end answer() calls (retrieval + Groq generation) in ms.

        Real wall-clock latency, not a synthetic estimate: each question is a
        genuine retrieve() + generate_answer() call against the live FAISS
        index and Groq API, so results include real network latency to Groq
        and reflect what a dashboard user would actually experience.

        Resilient to Groq's free-tier rate limit by design, not by accident:
        running ~20 back-to-back full-context queries against the free tier's
        8000 tokens/minute cap reliably exhausts it mid-run (confirmed live -
        a burst benchmark crashed with groq.RateLimitError after ~12 queries).
        The Groq SDK's own built-in retry/backoff isn't enough at sustained
        burst volume, so this method adds its own bounded retry per call plus
        a fixed delay between calls (excluded from the measured latency - it's
        a benchmark-harness courtesy to the shared rate limit, not part of
        what a single real user would experience), and degrades a query to a
        recorded failure rather than crashing the whole benchmark run.
        """
        import statistics
        import time

        from groq import RateLimitError

        if questions is None:
            questions = [q["question"] for q in EVAL_QUERIES]

        latencies_ms: list[float] = []
        failures: list[dict] = []

        for i, question in enumerate(questions):
            if i > 0:
                time.sleep(delay_between_calls_sec)

            for attempt in range(max_retries_per_call + 1):
                t0 = time.perf_counter()
                try:
                    self.answer(question, k=k)
                    latencies_ms.append((time.perf_counter() - t0) * 1000)
                    break
                except RateLimitError as exc:
                    if attempt < max_retries_per_call:
                        backoff = 10.0 * (attempt + 1)
                        logger.warning(
                            "benchmark_latency: rate limited on %r (attempt %d/%d), waiting %.0fs",
                            question, attempt + 1, max_retries_per_call, backoff,
                        )
                        time.sleep(backoff)
                    else:
                        logger.warning("benchmark_latency: giving up on %r after rate limiting", question)
                        failures.append({"question": question, "error": str(exc)})

        if not latencies_ms:
            return {
                "n_queries": 0, "n_failed": len(failures), "failures": failures,
                "median_ms": None, "mean_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None,
                "latencies_ms": [],
            }

        return {
            "n_queries": len(latencies_ms),
            "n_failed": len(failures),
            "failures": failures,
            "median_ms": statistics.median(latencies_ms),
            "mean_ms": statistics.mean(latencies_ms),
            "p95_ms": statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms),
            "min_ms": min(latencies_ms),
            "max_ms": max(latencies_ms),
            "latencies_ms": latencies_ms,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from news_ingestion import NewsIngestion

    ingest = NewsIngestion()
    articles = ingest.load_fnspid_filtered()
    print(f"Loaded {len(articles)} articles (source={ingest.fnspid_source_used})")

    rag = RAGPipeline()
    rag.build_index(articles)
    print(f"Indexed passage count: {rag.indexed_passage_count}")

    result = rag.answer("What recent news is there about Apple's earnings?")
    print(result["answer"])
    for c in result["citations"]:
        print(c["marker"], c["headline"], c["date"])
