"""Part 2 — does the trust score HELP the RAG? (A/B poisoning test).

Manufactures conflicts the approved corpus lacks: for 20 real docs, build a
low-trust TWIN with a few facts changed, index good+twin into an ISOLATED test
collection, then answer ~200 targeted questions with trust ON vs OFF. If trust
makes the RAG follow the good (correct) doc, trust helped.

Everything is isolated — its own Qdrant collection + docstore + trust table — so
the production index is never touched.

Run order (from the dc-available directory):
    python -m evaluation.trust_score.rag_ab.select_docs
    python -m evaluation.trust_score.rag_ab.make_twins
    python -m evaluation.trust_score.rag_ab.build_index
    python -m evaluation.trust_score.rag_ab.run_ab
    python -m evaluation.trust_score.rag_ab.analyze_ab
"""
