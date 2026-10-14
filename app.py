import json

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

REPO = "shaikhshahud99/damagetriage-bright"
CLS = ["intact", "damaged", "destroyed"]
COL = {"HIGH": "#e53935", "REVIEW": "#fb8c00", "LOW": "#43a047"}

st.set_page_config(page_title="DamageTriage (BRIGHT)", layout="wide")


def _token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None


@st.cache_data(show_spinner="Data load ho raha hai...")
def load():
    tok = _token()

    def get(name):
        return hf_hub_download(REPO, "bundle/" + name, repo_type="model", token=tok)

    df = pd.read_csv(get("predictions.csv"))
    with open(get("triage_summary.json")) as f:
        summ = json.load(f)
    d = np.load(get("demo_crops.npz"))
    cams = {k: get(f"gradcam_{k}.png") for k in ("tp", "fn", "fp")}
    return df, summ, d["pre"], d["post"], d["row"], cams


df, summ, pre, post, rows, cams = load()
df["color"] = df.tier.map(COL)

st.title("DamageTriage: BRIGHT (Noto earthquake)")
st.caption(
    "Prototype: pre-event optical + post-event SAR se building damage triage, "
    "human review ke saath. Noto training me nahi tha, naya event hai."
)
t1, t2, t3, t4, t5 = st.tabs(["Dashboard", "Review queue", "Map", "Explain", "About"])

with t1:
    c = st.columns(4)
    c[0].metric("Buildings", f"{summ['n_buildings']:,}")
    for col_, t in zip(c[1:], ["HIGH", "REVIEW", "LOW"]):
        col_.metric(f"{t} tier", f"{summ['tiers'][t]['n']:,}")
    tb = pd.DataFrame(summ["tiers"]).T
    st.write(f"Har tier me asli damage rate (overall: {summ['true_damage_rate']:.1%}):")
    st.bar_chart(tb["true_damage_rate"])
    st.write("Damage score ka distribution:")
    h, _ = np.histogram(df.damage_score, bins=20, range=(0, 1))
    st.bar_chart(pd.Series(h, index=[f"{i / 20:.2f}" for i in range(20)]))

with t2:
    st.write(
        "Demo subset (800 buildings: 300 damaged/destroyed + 500 intact, yani damaged "
        "buildings asli data se zyada hain). REVIEW tier ki sabse uncertain buildings pehle. "
        "Pre aur SAR dekh ke apna label chuno."
    )
    sub = df.iloc[rows].copy()
    sub["k"] = np.arange(len(rows))
    sub["unc"] = (sub.damage_score - 0.5).abs()
    q = sub[sub.tier == "REVIEW"].sort_values("unc")
    st.write(f"REVIEW tier me {len(q)} buildings")
    if "fb" not in st.session_state:
        st.session_state.fb = {}
    if len(q):
        pos = 0
        if len(q) > 1:
            pos = int(st.number_input("Queue position", 0, len(q) - 1, 0, step=1))
        r = q.iloc[pos]
        k = int(r.k)
        a, b = st.columns(2)
        a.image(pre[k], caption="pre-event optical", width=170)
        b.image(post[k], caption="post-event SAR", width=170)
        st.write(
            f"score {r.damage_score:.2f} | intact {r.p_intact:.2f} | "
            f"damaged {r.p_damaged:.2f} | destroyed {r.p_destroyed:.2f}"
        )
        ch = st.radio(
            "Tera label",
            ["unsure", "intact", "damaged", "destroyed"],
            key=f"r{k}",
            horizontal=True,
        )
        st.session_state.fb[int(r.name)] = ch
        with st.expander("Ground truth (sirf evaluation ke liye)"):
            st.write(CLS[int(r.true_label)])
    done = {i: l for i, l in st.session_state.fb.items() if l != "unsure"}
    out = pd.DataFrame(
        {"df_index": list(done.keys()), "reviewer_label": list(done.values())}
    )
    st.write(f"Tune {len(done)} buildings label kiye")
    st.download_button("Corrections CSV download", out.to_csv(index=False), "review_labels.csv")

with t3:
    st.write("Lal = HIGH, narangi = REVIEW, hara = LOW")
    st.map(df, latitude="lat", longitude="lon", color="color", size=6)

with t4:
    st.write(
        "Grad-CAM: model ke 'damage' score ko kaunsa hissa badha raha hai "
        "(optical aur SAR branch alag)."
    )
    for k, title in [("tp", "Sahi pakde"), ("fn", "Miss hue"), ("fp", "False alarm")]:
        st.image(cams[k], caption=title)
    st.warning(
        "Limitation: SAR heatmap har case me building ke beech ek gol dhabba dikhata hai, "
        "aur 'miss hue' cases me heatmap lagbhag khaali hai. Isse damage ka saboot mat maano, "
        "ye sirf ek indication hai."
    )

with t5:
    st.markdown(
        """
**Setup:** Turkey, Libya, Hawaii, Marshall aur Bata events pe train, **Noto earthquake pe test** (naya event).
Model: do ResNet18 branch (pre-event optical + post-event SAR), 64x64 building crops.

**Noto pe result (ek run):** AUROC 0.635, PR-AUC 0.231 (random 0.154), top-20% flagged me 34% damaged pakde (random 20%).
Sirf Turkey pe train karne par AUROC 0.546 tha.

**Honest limitations:**
- Naye event pe signal kamzor hai. LOW tier me bhi damaged buildings hain, to ise auto-decision nahi, prioritization aid maano.
- Noto sirf ek region hai (~70 images), aur ye ek hi training run ke numbers hain.
- Optical branch ko pre-event image milti hai, to wo damage nahi, building ka type seekh sakti hai.
- Predictions precomputed hain, app me live inference nahi hai.

**Data:** BRIGHT dataset (Kullervo/BRIGHT, Hugging Face). Non-commercial use, dataset card ke license aur paper citation ke saath.
"""
    )
