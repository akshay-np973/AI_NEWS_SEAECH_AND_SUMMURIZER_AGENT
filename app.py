
import os
import json
import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


st.set_page_config(
    page_title="AI News Summarizer Agent",
    page_icon="📰",
    layout="wide",
)

if "history" not in st.session_state:
    # each item: {query, timestamp, summary, sources, style, model}
    st.session_state.history = []

if "selected_entry" not in st.session_state:
    st.session_state.selected_entry = None

RECENCY_OPTIONS = {
    "Any time": None,
    "Past 24 hours": 1,
    "Past week": 7,
    "Past month": 30,
}

STYLE_INSTRUCTIONS = {
    "Bullet points": (
        "Summarize the following news into clear, concise bullet points. "
        "Group related stories together and give each bullet a short bold headline."
    ),
    "One paragraph": (
        "Summarize the following news into a single well-structured paragraph "
        "covering only the most important developments."
    ),
    "Detailed with headings": (
        "Summarize the following news into a structured summary using markdown "
        "'##' headings for each sub-topic, with 2-4 bullet points under each heading."
    ),
}

MODEL_OPTIONS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

with st.sidebar:
    st.title("⚙️ Settings")

    with st.expander("🔑 API keys", expanded=True):
        tavily_key = st.text_input(
            "Tavily API key",
            value=os.getenv("TAVILY_API_KEY", ""),
            type="password",
            help="Get a free key at tavily.com",
        )
        groq_key = st.text_input(
            "Groq API key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Get a free key at console.groq.com",
        )

    model_name = st.selectbox("Groq model", MODEL_OPTIONS, index=0)
    temperature = st.slider("Model creativity (temperature)", 0.0, 1.0, 0.3, 0.1)

    st.markdown("#### Search options")
    max_results = st.slider("Number of sources to fetch", 3, 10, 5)
    search_depth = st.radio("Search depth", ["basic", "advanced"], horizontal=True)
    recency_label = st.selectbox("Recency filter", list(RECENCY_OPTIONS.keys()))
    include_images = st.checkbox("Include images in raw results", value=False)

    st.markdown("#### Summary options")
    summary_style = st.selectbox("Summary style", list(STYLE_INSTRUCTIONS.keys()))
    add_tags = st.checkbox("Add key-topic tags", value=True)
    add_sentiment = st.checkbox("Add overall sentiment", value=True)

    st.markdown("---")
    st.markdown("#### 📜 History")
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.selected_entry = None
        st.rerun()

    if not st.session_state.history:
        st.caption("No searches yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            label = f"{item['query'][:28]}{'…' if len(item['query']) > 28 else ''}"
            if st.button(f"🕒 {label}", key=f"hist_{i}", use_container_width=True):
                st.session_state.selected_entry = item



def build_agent():
    """Instantiate the search tool and the summarization chain."""
    if not tavily_key or not groq_key:
        st.error("⚠️ Please provide both a Tavily API key and a Groq API key in the sidebar.")
        st.stop()

    os.environ["TAVILY_API_KEY"] = tavily_key
    os.environ["GROQ_API_KEY"] = groq_key

    tool_kwargs = {
        "max_results": max_results,
        "search_depth": search_depth,
        "include_images": include_images,
    }
    days = RECENCY_OPTIONS[recency_label]
    if days is not None:
        # Some versions of TavilySearchResults accept "days" directly;
        # fall back gracefully if not supported.
        tool_kwargs["days"] = days

    try:
        search_tool = TavilySearchResults(**tool_kwargs)
    except TypeError:
        # Retry without the optional/unsupported kwargs
        tool_kwargs.pop("days", None)
        tool_kwargs.pop("include_images", None)
        search_tool = TavilySearchResults(**tool_kwargs)

    llm = ChatGroq(model=model_name, temperature=temperature)

    extra_instructions = []
    if add_tags:
        extra_instructions.append(
            '- End with a line "🏷️ **Key Topics:** " followed by 3-5 short comma-separated tags.'
        )
    if add_sentiment:
        extra_instructions.append(
            '- End with a line "📊 **Overall Sentiment:** " (Positive / Neutral / Negative / Mixed) '
            "plus a one-sentence reason."
        )

    prompt = ChatPromptTemplate.from_template(
        """You are a professional, unbiased news analyst assistant.

{style_instruction}

Rules:
- Only use information present in the NEWS CONTENT below — never invent facts.
- Keep it skimmable and well formatted in markdown.
{extra_instructions}

NEWS CONTENT:
{news}
"""
    )

    chain = prompt | llm | StrOutputParser()
    return search_tool, chain, "\n".join(extra_instructions)


def run_single_query(query_text: str, search_tool, chain, extra_instructions: str):
    with st.spinner(f"🔎 Searching the web for “{query_text}”…"):
        try:
            raw_results = search_tool.run(query_text)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return None

    with st.spinner("🧠 Summarizing with Groq…"):
        try:
            summary = chain.invoke(
                {
                    "news": raw_results,
                    "style_instruction": STYLE_INSTRUCTIONS[summary_style],
                    "extra_instructions": extra_instructions,
                }
            )
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            return None

    entry = {
        "query": query_text,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "sources": raw_results,
        "style": summary_style,
        "model": model_name,
    }
    st.session_state.history.append(entry)
    return entry


def render_entry(entry: dict):
    st.subheader(f"📝 {entry['query']}")
    st.caption(
        f"🕒 {entry['timestamp']}  ·  🤖 {entry['model']}  ·  🎨 {entry['style']}"
    )
    st.markdown(entry["summary"])

    word_count = len(entry["summary"].split())
    read_min = max(1, round(word_count / 200))
    st.caption(f"📝 ~{word_count} words · ⏱️ ~{read_min} min read")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download summary (.md)",
            data=entry["summary"],
            file_name=f"summary_{entry['timestamp'].replace(' ', '_').replace(':', '-')}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"dl_md_{entry['timestamp']}",
        )
    with col2:
        st.download_button(
            "⬇️ Download raw sources (.json)",
            data=json.dumps(entry["sources"], indent=2, default=str),
            file_name=f"sources_{entry['timestamp'].replace(' ', '_').replace(':', '-')}.json",
            mime="application/json",
            use_container_width=True,
            key=f"dl_json_{entry['timestamp']}",
        )

    with st.expander("🔗 View raw search results"):
        st.write(entry["sources"])



st.title("📰 AI News Summarizer Agent")
st.caption(
    "Live web search (Tavily) + LLM summarization (Groq via LangChain). "
    "Enter a topic, or run several at once in batch mode."
)

tab_single, tab_batch = st.tabs(["🔍 Single search", "📦 Batch search"])

with tab_single:
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "News topic",
            placeholder="e.g. Latest AI news 2026",
            label_visibility="collapsed",
        )
    with col2:
        run_btn = st.button("Search & Summarize", use_container_width=True, type="primary")

    if run_btn:
        if not query.strip():
            st.warning("Please enter a topic to search for.")
        else:
            search_tool, chain, extra_instructions = build_agent()
            entry = run_single_query(query.strip(), search_tool, chain, extra_instructions)
            if entry:
                st.session_state.selected_entry = entry

    if st.session_state.selected_entry:
        st.markdown("---")
        render_entry(st.session_state.selected_entry)

with tab_batch:
    st.caption("Enter one topic per line — each will be searched and summarized separately.")
    batch_input = st.text_area(
        "Topics",
        placeholder="Latest AI news 2026\nSpaceX launches\nGlobal chip shortage",
        height=120,
        label_visibility="collapsed",
    )
    batch_btn = st.button("Run batch search", type="primary")

    if batch_btn:
        topics = [t.strip() for t in batch_input.split("\n") if t.strip()]
        if not topics:
            st.warning("Please enter at least one topic.")
        else:
            search_tool, chain, extra_instructions = build_agent()
            progress = st.progress(0.0, text="Starting batch run…")
            for i, topic in enumerate(topics, start=1):
                progress.progress(i / len(topics), text=f"({i}/{len(topics)}) {topic}")
                entry = run_single_query(topic, search_tool, chain, extra_instructions)
                if entry:
                    render_entry(entry)
                    st.markdown("---")
            progress.empty()
            st.success(f"Done — summarized {len(topics)} topic(s). Check the sidebar for history.")