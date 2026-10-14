import json

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "shaikhshahud99/damagetriage-bright"
CLASSES = ["Intact", "Damaged", "Destroyed"]
TIERS = ["HIGH", "REVIEW", "LOW"]
COL = {"HIGH": "#C0392B", "REVIEW": "#D68910", "LOW": "#2E7D5B"}
ACCENT = "#0B5C7A"
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SAR_MEAN, SAR_STD = 0.2255, 0.1447

st.set_page_config(page_title="DamageTriage", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1100px;}
    .legend {display: flex; gap: 20px; margin: 0.25rem 0 0.75rem 0; font-size: 0.92rem;}
    .legend i {display: inline-block; width: 10px; height: 10px; border-radius: 50%;
               margin-right: 7px; vertical-align: baseline;}
    .subtitle {color: #52606D; font-size: 1.02rem; margin-top: -0.6rem; max-width: 46rem;}
    .note {color: #52606D; font-size: 0.9rem;}
    div[data-testid="stMetricValue"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None


@st.cache_data(show_spinner="Loading data...")
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


@st.cache_resource(show_spinner="Loading model...")
def _load_session():
    import onnxruntime as ort

    path = hf_hub_download(REPO, "bundle/model_fusion.onnx", repo_type="model", token=_token())
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def load_model():
    try:
        return _load_session()
    except Exception:
        return None


def _to_64(img):
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if side == 64:
        return img
    method = Image.Resampling.BOX if side > 64 else Image.Resampling.BICUBIC
    return img.resize((64, 64), method)


def predict(sess, pre_img, post_img):
    o = np.asarray(_to_64(pre_img.convert("RGB")), dtype=np.float32) / 255.0
    o = ((o - IM_MEAN) / IM_STD).transpose(2, 0, 1)[None].astype(np.float32)
    s_ = np.asarray(_to_64(post_img.convert("L")), dtype=np.float32) / 255.0
    s_ = ((s_ - SAR_MEAN) / SAR_STD)[None, None].astype(np.float32)
    logits = sess.run(["logits"], {"optical": o, "sar": s_})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def tier_of(score):
    if score <= 0.3:
        return "LOW"
    return "REVIEW" if score <= 0.7 else "HIGH"


df, summ, pre, post, rows, cams = load()
df["color"] = df.tier.map(COL)
overall = summ["true_damage_rate"]

st.title("DamageTriage")
st.markdown(
    '<p class="subtitle">Prioritizes buildings for inspection after an earthquake, using '
    "pre-event optical and post-event SAR imagery. Shown on the Noto earthquake, an event "
    "the model never saw during training.</p>",
    unsafe_allow_html=True,
)

t_dash, t_try, t_rev, t_map, t_exp, t_about = st.tabs(
    ["Overview", "Try it", "Review queue", "Map", "Explainability", "About"]
)

# ---------------------------------------------------------------- Overview
with t_dash:
    total = summ["n_buildings"]
    c = st.columns(4)
    c[0].metric("Buildings assessed", f"{total:,}")
    for col_, t in zip(c[1:], TIERS):
        n = summ["tiers"][t]["n"]
        col_.metric(f"{t.capitalize()} priority", f"{n:,}", f"{n / total:.0%} of all")

    st.subheader("How well do the tiers separate damage?")
    st.markdown(
        f'<p class="note">Observed damage rate is the share of buildings in each tier that are '
        f"truly damaged or destroyed. Across all buildings it is {overall:.1%}.</p>",
        unsafe_allow_html=True,
    )
    tb = pd.DataFrame(
        {
            "Tier": TIERS,
            "Buildings": [summ["tiers"][t]["n"] for t in TIERS],
            "Observed damage rate": [summ["tiers"][t]["true_damage_rate"] for t in TIERS],
        }
    )
    tb["Share of buildings"] = tb["Buildings"] / total
    tb["Lift vs overall"] = tb["Observed damage rate"] / overall

    left, right = st.columns([3, 2])
    with left:
        show = pd.DataFrame(
            {
                "Tier": tb["Tier"],
                "Buildings": tb["Buildings"].map("{:,}".format),
                "Share of buildings": tb["Share of buildings"].map("{:.1%}".format),
                "Observed damage rate": tb["Observed damage rate"].map("{:.1%}".format),
                "Lift vs overall": tb["Lift vs overall"].map("{:.2f}x".format),
            }
        )
        st.dataframe(show, hide_index=True, width="stretch")
    with right:
        st.bar_chart(tb.set_index("Tier")["Observed damage rate"], color=ACCENT, height=220)

    st.subheader("Distribution of damage scores")
    st.markdown(
        '<p class="note">The damage score is the model\'s probability that a building is damaged '
        "or destroyed. Tiers are cut at 0.3 and 0.7.</p>",
        unsafe_allow_html=True,
    )
    h, _ = np.histogram(df.damage_score, bins=20, range=(0, 1))
    st.bar_chart(
        pd.Series(h, index=[f"{i / 20:.2f}" for i in range(20)], name="Buildings"),
        color=ACCENT,
        height=220,
    )

# ------------------------------------------------------------------ Try it
with t_try:
    st.markdown(
        '<p class="note">Check a single building. Provide a before image (optical) and an after '
        "image (SAR) of the same building, or pick a demo building. The model returns a damage "
        "score and a priority tier.</p>",
        unsafe_allow_html=True,
    )
    sess = load_model()
    if sess is None:
        st.info(
            "The live model is not available right now. The other tabs still work with the "
            "precomputed predictions."
        )
    else:
        source = st.radio("Input", ["Upload images", "Demo building"], horizontal=True)
        pre_in = post_in = truth = None
        if source == "Upload images":
            u1, u2 = st.columns(2)
            f1 = u1.file_uploader("Before image (optical, color)", type=["png", "jpg", "jpeg"], key="up_pre")
            f2 = u2.file_uploader("After image (SAR, grayscale)", type=["png", "jpg", "jpeg"], key="up_post")
            if f1 is not None and f2 is not None:
                pre_in = Image.open(f1)
                post_in = Image.open(f2)
        else:
            i_demo = st.slider("Demo building", 0, len(rows) - 1, 0)
            pre_in = Image.fromarray(pre[i_demo])
            post_in = Image.fromarray(post[i_demo])
            truth = CLASSES[int(df.iloc[int(rows[i_demo])].true_label)]

        if pre_in is None:
            st.info("Upload both images to see a result.")
        else:
            p = predict(sess, pre_in, post_in)
            score = float(1 - p[0])
            tier = tier_of(score)
            v1, v2, v3 = st.columns([1, 1, 2])
            v1.image(pre_in.convert("RGB"), caption="Before (optical)", width=190)
            v2.image(post_in.convert("L"), caption="After (SAR)", width=190)
            with v3:
                st.markdown(
                    f'Priority tier: <span style="color:{COL[tier]};font-weight:700">{tier}</span>'
                    f"  \nDamage score: **{score:.2f}**",
                    unsafe_allow_html=True,
                )
                for name, pk in zip(CLASSES, p):
                    st.progress(float(pk), text=f"{name}: {pk:.0%}")
            if truth is not None:
                with st.expander("Show ground truth (for evaluation only)"):
                    st.write(truth)
        st.warning(
            "The model was trained on 64 by 64 pixel crops, each centered on one building with "
            "about 50 percent margin, from imagery at roughly 0.3 m per pixel. Images that do "
            "not look like that give unreliable scores. Non-square images are center-cropped. "
            "The score ranks buildings for inspection; it does not certify a building as safe."
        )

# ------------------------------------------------------------ Review queue
with t_rev:
    st.markdown(
        '<p class="note">Buildings in the review tier, most uncertain first. Compare the two '
        "images, then record your own label. This demo uses a sample of 800 buildings that "
        "contains more damaged buildings than the real event.</p>",
        unsafe_allow_html=True,
    )
    sub = df.iloc[rows].copy()
    sub["k"] = np.arange(len(rows))
    sub["unc"] = (sub.damage_score - 0.5).abs()
    q = sub[sub.tier == "REVIEW"].sort_values("unc")
    n_q = len(q)

    if "fb" not in st.session_state:
        st.session_state.fb = {}
    if "pos" not in st.session_state:
        st.session_state.pos = 0

    def _step(delta, limit):
        st.session_state.pos = int(min(max(0, st.session_state.pos + delta), limit - 1))

    if n_q == 0:
        st.info("No buildings in the review tier for this sample.")
    else:
        pos = min(st.session_state.pos, n_q - 1)
        r = q.iloc[pos]
        k = int(r.k)
        reviewed = {i: lab for i, lab in st.session_state.fb.items() if lab != "Not sure"}

        nav = st.columns([1, 1, 4])
        nav[0].button("Previous", on_click=_step, args=(-1, n_q), disabled=pos == 0,
                      width="stretch")
        nav[1].button("Next", on_click=_step, args=(1, n_q), disabled=pos >= n_q - 1,
                      width="stretch")
        nav[2].markdown(
            f"Building {pos + 1} of {n_q}. Reviewed so far: {len(reviewed)}."
        )

        a, b, info = st.columns([1, 1, 2])
        a.image(pre[k], caption="Before (optical)", width=190)
        b.image(post[k], caption="After (SAR)", width=190)
        with info:
            st.markdown(
                f"Model tier: **{r.tier}**  \nDamage score: **{r.damage_score:.2f}**"
            )
            for name, p in zip(CLASSES, [r.p_intact, r.p_damaged, r.p_destroyed]):
                st.progress(float(p), text=f"{name}: {p:.0%}")

        choice = st.radio(
            "Your label",
            ["Not sure", "Intact", "Damaged", "Destroyed"],
            key=f"label_{k}",
            horizontal=True,
        )
        st.session_state.fb[int(r.name)] = choice
        with st.expander("Show ground truth (for evaluation only)"):
            st.write(CLASSES[int(r.true_label)])

        reviewed = {i: lab for i, lab in st.session_state.fb.items() if lab != "Not sure"}
        out = pd.DataFrame(
            {
                "building_id": list(reviewed.keys()),
                "reviewer_label": [v.lower() for v in reviewed.values()],
                "model_tier": [df.loc[i, "tier"] for i in reviewed.keys()],
                "damage_score": [round(float(df.loc[i, "damage_score"]), 4) for i in reviewed.keys()],
            }
        )
        st.download_button(
            "Download reviewed labels (CSV)",
            out.to_csv(index=False),
            "review_labels.csv",
            disabled=len(out) == 0,
        )

# --------------------------------------------------------------------- Map
with t_map:
    shown = st.multiselect("Tiers shown", TIERS, default=TIERS)
    st.markdown(
        '<div class="legend">'
        + "".join(f'<span><i style="background:{COL[t]}"></i>{t.capitalize()}</span>' for t in TIERS)
        + "</div>",
        unsafe_allow_html=True,
    )
    mdf = df[df.tier.isin(shown)]
    if len(mdf):
        st.map(mdf, latitude="lat", longitude="lon", color="color", size=6)
    else:
        st.info("Select at least one tier to show buildings on the map.")

# ---------------------------------------------------------- Explainability
with t_exp:
    st.markdown(
        '<p class="note">Grad-CAM highlights the image regions that pushed the damage score up. '
        "Each row shows the before image, the optical heatmap, the SAR image, and the SAR heatmap.</p>",
        unsafe_allow_html=True,
    )
    options = {
        "Correctly flagged": "tp",
        "Missed damage": "fn",
        "False alarm": "fp",
    }
    pick = st.radio("Group", list(options.keys()), horizontal=True)
    st.image(cams[options[pick]])
    st.warning(
        "Read these with care. The SAR heatmap is almost always a round blob at the center "
        "of the crop, whether or not the building is damaged, and the heatmaps for missed "
        "damage are nearly empty. They show where the model looked, not proof of damage."
    )

# ------------------------------------------------------------------- About
with t_about:
    st.subheader("Method")
    st.markdown(
        "A two-branch ResNet18 classifier. One branch reads the pre-event optical crop, the "
        "other reads the post-event SAR crop, and their features are combined to predict "
        "intact, damaged, or destroyed for each building. Crops are 64 by 64 pixels around "
        "each building footprint. The model was trained on the Turkey, Libya, Hawaii, Marshall "
        "and Bata events and tested on the Noto earthquake."
    )
    st.subheader("Results on the unseen Noto event")
    res = pd.DataFrame(
        {
            "Training data": ["Random ranking", "Turkey only", "Five events"],
            "AUROC": ["0.500", "0.546", "0.635"],
            "PR-AUC": ["0.154", "0.213", "0.231"],
            "Damage found in top 20% of scores": ["20%", "27%", "34%"],
        }
    )
    st.dataframe(res, hide_index=True, width="stretch")
    st.markdown(
        '<p class="note">Each figure comes from a single training run. PR-AUC for random '
        "ranking equals the damage rate (0.154).</p>",
        unsafe_allow_html=True,
    )
    st.subheader("Limitations")
    st.markdown(
        "- Signal on a new event is weak. Low-priority buildings still include damaged ones, "
        "so use this to prioritize inspections, not to decide that a building is safe.\n"
        "- Noto is a single region with about 70 image tiles, and the numbers come from one run.\n"
        "- The optical branch only sees pre-event imagery, so it may learn building type "
        "rather than damage.\n"
        "- The overview, review queue, map and explainability tabs show precomputed predictions on "
        "the Noto event. Only the Try it tab runs the model live."
    )
    st.subheader("Data")
    st.markdown(
        "BRIGHT dataset (Kullervo/BRIGHT on Hugging Face). Use is non-commercial; follow the "
        "dataset card's license and cite the BRIGHT paper."
    )
