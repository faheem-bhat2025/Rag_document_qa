"""
Streamlit frontend for the RAG Document Q&A system.

Runs as a separate process from the FastAPI backend and talks to it purely
over HTTP using the `requests` library. Provides:
  1. A chat-style interface to ask questions and view answers with sources.
  2. An analytics dashboard showing usage stats from the /analytics endpoint.

Run (with the FastAPI backend already running on port 8000):
    streamlit run streamlit_app.py
"""

import requests
import pandas as pd
import streamlit as st

# Base URL of the FastAPI backend. Override in the sidebar if needed.
DEFAULT_API_URL = "http://localhost:8000"

st.set_page_config(page_title="AWS Agreement Q&A", page_icon="📄", layout="wide")



# Helpers: thin wrappers around the FastAPI endpoints                          
def ingest_document(api_url: str) -> dict:
    """Trigger POST /ingest to build the vector index."""
    resp = requests.post(f"{api_url}/ingest", timeout=300)
    resp.raise_for_status()
    return resp.json()


def ask_question(api_url: str, query: str, k: int = 4) -> dict:
    """Send a query to POST /ask and return the parsed response."""
    resp = requests.post(f"{api_url}/ask", json={"query": query, "k": k}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_analytics(api_url: str) -> dict:
    """Fetch usage analytics from GET /analytics."""
    resp = requests.get(f"{api_url}/analytics", timeout=30)
    resp.raise_for_status()
    return resp.json()


# Sidebar: configuration and document ingestion                               
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("Backend API URL", value=DEFAULT_API_URL).rstrip("/")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=4)

    st.divider()
    st.caption("Run this once before asking questions:")
    if st.button("📥 Ingest document", use_container_width=True):
        with st.spinner("Parsing, chunking, and embedding the PDF..."):
            try:
                result = ingest_document(api_url)
                st.success(
                    f"Ingested — {result.get('chunks_indexed', '?')} chunks indexed."
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend. Is the FastAPI server running?")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")


st.title("📄 AWS Customer Agreement — Q&A")
st.caption("Ask questions about the AWS Customer Agreement, grounded in the document.")

tab_chat, tab_analytics = st.tabs(["💬 Chat", "📊 Analytics"])


# Tab 1: Chat interface                                                     

with tab_chat:
    # Persist the conversation across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay the existing conversation.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # For assistant messages, show the sources that backed the answer.
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📑 Sources"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**Source {i}** (score: {src['score']:.3f})\n\n"
                            f"> {src['text']}"
                        )

    # New user input.
    if prompt := st.chat_input("Ask a question about the agreement..."):
        # Show and store the user's message.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the backend and show the answer.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    data = ask_question(api_url, prompt, k=top_k)
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    latency = data.get("latency_ms", 0)

                    st.markdown(answer)
                    if data.get("no_answer"):
                        st.info("No answer was found in the document for this query.")
                    if sources:
                        with st.expander("📑 Sources"):
                            for i, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**Source {i}** (score: {src['score']:.3f})\n\n"
                                    f"> {src['text']}"
                                )
                    st.caption(f"⏱️ {latency:.0f} ms")

                    # Store the assistant turn (with sources) in history.
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend. Is the FastAPI server running?")
                except requests.exceptions.HTTPError as exc:
                    # Surface meaningful backend errors (e.g. 409 = not ingested).
                    detail = ""
                    try:
                        detail = exc.response.json().get("detail", "")
                    except Exception:
                        detail = str(exc)
                    st.error(f"Request failed: {detail}")
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")



# Tab 2: Analytics dashboard                                                   
with tab_analytics:
    st.subheader("Usage Analytics")
    st.caption("Computed from the SQL logs of every /ask call.")

    if st.button("🔄 Refresh analytics", use_container_width=False):
        st.session_state.pop("analytics", None)  # force a refetch

    # Fetch (and cache in session) the analytics data.
    if "analytics" not in st.session_state:
        try:
            st.session_state.analytics = fetch_analytics(api_url)
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend. Is the FastAPI server running?")
            st.session_state.analytics = None
        except Exception as exc:
            st.error(f"Could not load analytics: {exc}")
            st.session_state.analytics = None

    analytics = st.session_state.get("analytics")

    if analytics:
        # Top-line metrics.
        col1, col2, col3 = st.columns(3)
        col1.metric("Total queries", analytics.get("total_queries", 0))
        col2.metric("Avg latency (ms)", f"{analytics.get('average_latency_ms', 0):.1f}")
        col3.metric("No-answer queries", len(analytics.get("no_answer_queries", [])))

        st.divider()

        # Most frequently asked questions — table + bar chart.
        st.markdown("#### Most frequently asked questions")
        most_frequent = analytics.get("most_frequent_questions", [])
        if most_frequent:
            df_freq = pd.DataFrame(most_frequent)
            st.dataframe(df_freq, use_container_width=True, hide_index=True)
            st.bar_chart(df_freq.set_index("query")["times_asked"])
        else:
            st.write("No queries logged yet.")

        st.divider()

        # Queries where no answer was found.
        st.markdown("#### Queries with no answer found")
        no_answer = analytics.get("no_answer_queries", [])
        if no_answer:
            st.dataframe(
                pd.DataFrame(no_answer), use_container_width=True, hide_index=True
            )
        else:
            st.write("No unanswered queries logged yet.")