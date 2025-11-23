"""
Thin wrapper so `streamlit run main.py` loads the real Streamlit UI.
The CLI/notebook workflow still lives at `porter_agent/main.py`.
"""

from porter_agent.app import render_app

if __name__ == "__main__":
    render_app()
