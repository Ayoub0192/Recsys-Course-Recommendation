
import requests
import streamlit as st
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="LearnWise AI", layout="wide")

st.title("LearnWise AI – Adaptive Course Recommendation")

with st.sidebar:
    st.header("User Selection")
    user_id = st.text_input("User ID", value="u0001")
    k = st.slider("Number of recommendations (k)", min_value=3, max_value=20, value=5)
    run_btn = st.button("Get Recommendations")

if run_btn:
    with st.spinner("Fetching recommendations from API..."):
        try:
            resp = requests.get(
                f"{API_BASE}/recommend_next_lesson",
                params={"user_id": user_id, "k": k},
                timeout=30,
            )
            if resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                recs = data.get("recommendations", [])
                if not recs:
                    st.warning("No recommendations returned.")
                else:
                    df = pd.DataFrame(recs)
                    st.subheader(f"Top-{k} recommended lessons for {user_id}")
                    st.table(df)
        except Exception as e:
            st.error(f"Failed to reach API: {e}")

st.markdown("---")
st.caption("LearnWise AI – RecSys Startup Sprint MVP")
