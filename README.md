# AI_NEWS_SEAECH_AND_SUMMURIZER_AGENT
AI-powered Code Generator and Explainer built with LangChain, Groq, Tavily, and Streamlit. The project generates code from natural-language prompts and explains the generated code in simple terms. It also integrates Tavily web search for up-to-date information, providing an interactive and user-friendly AI development experience.
A Streamlit agent that searches the live web with Tavily and summarizes the results with Groq LLMs via LangChain — grounded, up-to-date summaries instead of stale model knowledge.

Features
Live web search with configurable depth, source count, and recency window
Fast inference via Groq (multiple open models to choose from)
Bullet-point, paragraph, or detailed-headings summary styles
Auto-generated topic tags and sentiment
Batch mode for multiple topics at once
Session history, plus Markdown/JSON export
How it works

Tavily fetches current search results for your query → LangChain assembles a prompt around those results → Groq generates the summary → Streamlit renders it, with options to download or revisit past runs.

Getting started
bash
git clone https://github.com/<your-username>/ai-news-summarizer-agent.git
cd ai-news-summarizer-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

Create a .env file:

env
TAVILY_API_KEY=your_key
GROQ_API_KEY=your_key

Run it:

bash
streamlit run app.py
Project structure
├── app.py              # Streamlit UI + agent logic
├── requirements.txt
├── .env                # API keys (not committed)
└── README.md
Tech stack

Streamlit · LangChain · Tavily Search API · Groq · python-dotenv

Roadmap
Streaming output and inline citations
Persistent (SQLite) history
Scheduled daily digests
Dockerfile for one-command deploy
Contributing

PRs welcome — fork, branch, commit, and open a pull request. Please open an issue first for larger changes.

Acknowledgements

Built on top of Tavily, Groq, LangChain, and Streamlit — thanks to their teams for the tooling that makes this possible
