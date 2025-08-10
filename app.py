# app.py — Drew Is • AI Coach (Streamlit)
# - Left: Coach answer (single engine: GPT-4.1)
# - Right: Comparison cards (GPT-4.1 generates strengths/limits/ideal for many models, incl. GPT-5 family)
# - No user token/temp controls (temperature fixed to 1)
# - Brand styling (#FF9900 + dark)
# - Detects model availability from your OpenAI account to avoid stale claims

import os, json, base64, textwrap, requests
import streamlit as st

# ============ Page & Styles ============
st.set_page_config(page_title="AI Coach — Model Comparison", layout="wide")

st.markdown("""
<style>
  .stApp { background:#0d0d0d; color:#e6e6e6; font-family:'Segoe UI', system-ui, sans-serif; }
  h1,h2,h3,h4,h5 { color:#FF9900; }
  .section-title { margin: 0 0 .25rem 0; font-weight:700; color:#FF9900; }
  .subtle { color:#bdbdbd; font-size:0.9rem; }
  .box { background:#1a1a1a; border:1px solid #333; border-radius:10px; padding:1rem; }
  textarea, input[type="text"] { background:#1a1a1a !important; color:#f0f0f0 !important; }
  div.stButton > button { background:#FF9900; color:#000; font-weight:700; border:none; border-radius:8px; padding:.5rem 1rem; }
  div.stButton > button:hover { background:#e68a00; color:#fff; }
  .card { background:#141414; border:1px solid #2a2a2a; border-radius:10px; padding:1rem; margin-bottom:1rem; }
  .kpis { display:flex; flex-wrap:wrap; gap:.5rem; margin:.25rem 0 .5rem 0; }
  .pill { border:1px solid #333; border-radius:999px; padding:.25rem .6rem; font-size:.8rem; color:#cfcfcf; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  @media (max-width: 960px) { .grid2 { grid-template-columns:1fr; } }
  .footer-logo { display:flex; justify-content:center; margin:2rem 0 0; }
</style>
""", unsafe_allow_html=True)

st.title("AI Coach — Model Comparison")
st.caption("Compare strengths, tradeoffs, and ideal use-cases. Clean, no-nonsense guidance.")

# ============ Keys / Headers ============
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()

HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

# Hidden defaults
TEXT_MODEL = "gpt-4.1"
TEMPERATURE = 1
ANSWER_MAXTOK = 600
COMPARE_MAXTOK = 1200

# ============ Catalog to Compare (you can edit labels/order freely) ============
CATALOG = {
    "Text LLMs": [
        {"id": "gpt-5",          "label": "GPT-5",          "provider": "OpenAI"},
        {"id": "gpt-5-mini",     "label": "GPT-5 mini",     "provider": "OpenAI"},
        {"id": "gpt-5-nano",     "label": "GPT-5 nano",     "provider": "OpenAI"},
        {"id": "gpt-4.1",        "label": "GPT-4.1",        "provider": "OpenAI"},
        {"id": "gpt-4.1-mini",   "label": "GPT-4.1 mini",   "provider": "OpenAI"},
        {"id": "claude-3.5",     "label": "Claude 3.5",     "provider": "Anthropic"},
        {"id": "gemini-1.5",     "label": "Gemini 1.5",     "provider": "Google"},
        {"id": "llama-3.1-70b",  "label": "Llama 3.1 70B",  "provider": "Meta"},
        {"id": "mistral-large-2","label": "Mistral Large 2","provider": "Mistral"},
        {"id": "cohere-command-a","label":"Cohere Command A","provider":"Cohere"},
    ],
    "Image Models": [
        {"id":"openai-image", "label":"OpenAI Image (latest)","provider":"OpenAI"},
        {"id":"sdxl",         "label":"Stable Diffusion XL",   "provider":"Stability"},
        {"id":"google-imagen","label":"Google Imagen",         "provider":"Google"},
        {"id":"midjourney",   "label":"Midjourney",            "provider":"Midjourney"},
    ],
    "Video Models": [
        {"id":"openai-video", "label":"OpenAI Video (latest)", "provider":"OpenAI"},
        {"id":"runway-gen3",  "label":"Runway Gen-3",          "provider":"Runway"},
        {"id":"pika",         "label":"Pika",                  "provider":"Pika"},
        {"id":"luma-dream",   "label":"Luma Dream Machine",    "provider":"Luma"},
    ],
    "Avatar Models": [
        {"id":"heygen",     "label":"HeyGen Avatars", "provider":"HeyGen"},
        {"id":"synthesia",  "label":"Synthesia",      "provider":"Synthesia"},
        {"id":"d-id",       "label":"D-ID",           "provider":"D-ID"},
    ],
}

# ============ Live availability from your key ============
def get_available_models() -> set:
    try:
        r = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, timeout=30)
        r.raise_for_status()
        return {m["id"] for m in r.json().get("data", [])}
    except Exception as e:
        print("Model list fetch failed:", e)
        return set()

AVAILABLE = get_available_models()  # e.g., {'gpt-4.1', 'gpt-5', ...}

# ============ OpenAI helpers (we only call GPT-4.1) ============
def openai_chat(messages, max_tokens=500, response_format=None) -> str:
    payload = {
        "model": TEXT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE
    }
    if response_format:
        payload["response_format"] = response_format
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error: {r.status_code} {r.text[:600]}")
    return r.json()["choices"][0]["message"]["content"]

def coach_answer(user_prompt: str) -> str:
    sys = (
        "You are 'AI Tools Coach' for professionals. Be concise, honest, tactical, and forward-looking. "
        "Use short sections and bullets when helpful. Avoid hype. If tradeoffs exist, state them bluntly."
    )
    return openai_chat(
        [{"role":"system","content":sys},
         {"role":"user","content":user_prompt}],
        max_tokens=ANSWER_MAXTOK
    )

def build_comparison(category: str, task_prompt: str) -> dict:
    """
    Ask GPT-4.1 to produce STRICT JSON comparison for the models in CATALOG[category].
    We pass an availability map (per your key) so the copy remains accurate (active vs not enabled).
    """
    models = CATALOG.get(category, [])
    names = [m["label"] for m in models]
    availability_map = {
        m["label"]: ("active" if m["id"] in AVAILABLE else "not-enabled")
        for m in models
    }

    schema = textwrap.dedent("""
    Return strict JSON ONLY (no code fences, no commentary):
    {
      "items": [
        {
          "name": "Exact model label from the list",
          "provider": "OpenAI|Anthropic|Google|Meta|Mistral|Cohere|Stability|Runway|HeyGen|Synthesia|D-ID|Midjourney",
          "type": "text|image|video|avatar",
          "availability": "active|not-enabled",
          "strengths": ["...", "..."],
          "limits": ["...", "..."],
          "ideal": ["...", "..."],
          "notes": "one or two-line note",
          "tiers": {
            "reasoning": "1|2|3|n/a",  // depth
            "latency": "1|2|3|n/a",    // lower is faster? use 1=low, 3=high overall capability; keep consistent
            "price": "1|2|3|n/a",
            "context": "small|medium|large|huge|n/a"
          }
        }
      ],
      "summary": "One-paragraph summary: who should pick what for this task."
    }
    """).strip()

    sys = (
        "You are an impartial analyst of AI model capabilities. Be current, specific, and avoid speculation. "
        "If a model is 'not-enabled', still describe typical capabilities and best-fit use-cases, "
        "but do not imply it is currently usable with this account."
    )
    user = textwrap.dedent(f"""
    Category: {category}
    Task prompt: {task_prompt}

    Models to include (exact labels, in this order):
    {", ".join(names)}

    Availability map (from this account):
    {json.dumps(availability_map, ensure_ascii=False)}

    Type mapping by category:
      - "Text LLMs" -> "text"
      - "Image Models" -> "image"
      - "Video Models" -> "video"
      - "Avatar Models" -> "avatar"

    {schema}
    """).strip()

    content = openai_chat(
        [{"role":"system","content":sys},{"role":"user","content":user}],
        max_tokens=COMPARE_MAXTOK,
        response_format={"type":"json_object"}
    )
    data = json.loads(content)
    # Guard: ensure each requested name appears, even if empty
    found = {it.get("name") for it in data.get("items", [])}
    for nm in names:
        if nm not in found:
            # add placeholder
            provider = next((m["provider"] for m in models if m["label"]==nm), "")
            _type = "text" if category=="Text LLMs" else "image" if category=="Image Models" else "video" if category=="Video Models" else "avatar"
            (data.setdefault("items", [])).append({
                "name": nm, "provider": provider, "type": _type,
                "availability": availability_map.get(nm,"not-enabled"),
                "strengths": [], "limits": [], "ideal": [],
                "notes": "", "tiers": {"reasoning":"n/a","latency":"n/a","price":"n/a","context":"n/a"}
            })
    return data

# ============ UI ============
left, right = st.columns([1,1], gap="large")

with left:
    st.markdown("<div class='section-title'>Ask AI Coach</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Get a straight answer for your situation.</div>", unsafe_allow_html=True)
    user_prompt = st.text_area(" ", placeholder="Type your question…", height=140, label_visibility="collapsed")
    if st.button("Get Coach Answer"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            try:
                st.markdown("<div class='box'>", unsafe_allow_html=True)
                st.markdown("#### Coach Answer")
                st.write(coach_answer(user_prompt))
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"OpenAI error: {e}")

with right:
    st.markdown("<div class='section-title'>Compare Models</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>See strengths, limits, and ideal use-cases.</div>", unsafe_allow_html=True)
    category = st.selectbox("Category", list(CATALOG.keys()))
    cmp_prompt = st.text_area("Task for comparison", placeholder="e.g., Summarize a 50-page contract and extract risk clauses", height=140)
    if st.button("Run Comparison"):
        if not cmp_prompt.strip():
            st.warning("Please enter a task prompt.")
        else:
            try:
                data = build_comparison(category, cmp_prompt)
                items = data.get("items", [])

                # Two-column card grid
                st.markdown("<div class='grid2'>", unsafe_allow_html=True)
                for item in items:
                    tiers = item.get("tiers", {})
                    strengths = item.get("strengths") or []
                    limits = item.get("limits") or []
                    ideal = item.get("ideal") or []
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"**{item.get('name','(unknown)')}**  \n*{item.get('provider','')}, {item.get('type','')}*")
                    st.caption(f"Availability: {item.get('availability','unknown')}")
                    if item.get("notes"):
                        st.write(item["notes"])
                    st.markdown("<div class='kpis'>", unsafe_allow_html=True)
                    st.markdown(f"<span class='pill'>reasoning {tiers.get('reasoning','n/a')}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span class='pill'>latency {tiers.get('latency','n/a')}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span class='pill'>price {tiers.get('price','n/a')}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span class='pill'>context {tiers.get('context','n/a')}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    if strengths:
                        st.markdown("**Strengths**")
                        st.write("• " + "\n• ".join(strengths))
                    if limits:
                        st.markdown("**Limits**")
                        st.write("• " + "\n• ".join(limits))
                    if ideal:
                        st.markdown("**Ideal for**")
                        st.write("• " + "\n• ".join(ideal))
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if data.get("summary"):
                    st.markdown("### Summary")
                    st.write(data["summary"])

            except Exception as e:
                st.error(f"OpenAI error: {e}")

# ============ Footer Logo ============
def render_footer_logo(path="Drew Is small logo 50.jpg", width=120):
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        st.markdown(
            f"""
            <div class='footer-logo'>
              <img src="data:image/jpeg;base64,{b64}" width="{width}" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.caption("")  # quiet fallback
        st.caption(f"(Logo not loaded: {e})")

render_footer_logo()
