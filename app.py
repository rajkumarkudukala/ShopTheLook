"""
app.py — Gradio demo for Shop-the-Look Visual Product Discovery.

Usage:
    python app.py
    Opens Gradio interface at http://localhost:7860
"""

import json
import gradio as gr
from src.pipeline import process_scene, process_image
from src.fetcher import convert_to_url, fetch_image

# --- Compatibility shim ---------------------------------------------------
# Gradio 5.x's API-info builder crashes when a component JSON schema contains
# a boolean (e.g. additionalProperties: true): gradio_client's get_type does
# `if "const" in schema` which raises "argument of type 'bool' is not iterable".
# This route is hit on every page load, so we guard get_type against non-dict
# schemas. Harmless if the underlying bug is already fixed in the installed
# gradio_client version.
try:
    import gradio_client.utils as _gcu

    _orig_get_type = _gcu.get_type

    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gcu.get_type = _safe_get_type

    # The recursive schema->type converter raises APIInfoParseError on a
    # boolean schema (e.g. additionalProperties: true). Short-circuit those.
    _orig_j2p = _gcu._json_schema_to_python_type

    def _safe_j2p(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_j2p(schema, defs)

    _gcu._json_schema_to_python_type = _safe_j2p
except Exception:
    pass
# -------------------------------------------------------------------------

# Load all unique validation scene signatures for the dropdown
with open("data/validation.jsonl") as f:
    ALL_SCENES = sorted(set(
        json.loads(line)["scene"] for line in f if line.strip()
    ))

CSS = """
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 {
    font-size: 2rem;
    margin-bottom: 0.25rem;
}
.main-header p {
    color: var(--body-text-color-subdued);
    font-size: 0.95rem;
    max-width: 640px;
    margin: 0 auto;
}
.stats-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 0.75rem 0;
}
.stat-card {
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    text-align: center;
    min-width: 120px;
}
.stat-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--body-text-color);
}
.stat-card .label {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.match-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: var(--block-background-fill);
}
.match-card h3 { margin-top: 0; }
.exact-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}
.exact-badge.yes { background: #dcfce7; color: #166534; }
.exact-badge.no  { background: #dbeafe; color: #1e40af; }
.scene-preview-col {
    display: flex;
    align-items: center;
    justify-content: center;
}
"""


def preview_scene(scene_signature: str):
    """Show preview image when a scene is selected from dropdown."""
    if not scene_signature:
        return None
    img = fetch_image(scene_signature)
    return img


def on_dropdown_select(scene_signature: str):
    """When dropdown changes, update the textbox and show preview."""
    img = fetch_image(scene_signature) if scene_signature else None
    return scene_signature, img


def on_textbox_change(scene_signature: str):
    """When textbox is typed into, show preview of that signature."""
    sig = scene_signature.strip()
    if not sig or len(sig) != 32:
        return None
    return fetch_image(sig)


def _render_result(result):
    """Turn a pipeline result dict into (scene_img, gallery, explanation, error)."""
    if result["error"]:
        return (
            None, [], gr.update(value=""),
            gr.update(value=result["error"], visible=True),
        )

    scene_img = result["scene_image"]

    gallery_items = []
    total_matches = 0
    exact_count = 0
    best_score = 0.0

    md_parts = []
    for i, item in enumerate(result["items"]):
        label = item["label"]
        conf = item["confidence"]
        person_tag = " (full outfit)" if item["is_full_person"] else ""

        gallery_items.append((item["crop"], f"Crop {i+1}: {label}"))

        md_parts.append(
            f'<div class="match-card">\n\n'
            f"### Detected: {label}{person_tag} — confidence {conf:.0%}\n\n"
        )

        for match in item["matches"]:
            total_matches += 1
            score = match["similarity_score"]
            best_score = max(best_score, score)
            is_exact = match["is_exact_match"]
            if is_exact:
                exact_count += 1
            pid = match["product_id"]
            badge_cls = "yes" if is_exact else "no"
            badge_text = "Exact Match" if is_exact else "Similar"

            md_parts.append(
                f'<span class="exact-badge {badge_cls}">{badge_text}</span> '
                f"**Score: {score:.3f}** | Product: `{pid[:16]}...`\n\n"
                f"{match['explanation']}\n\n"
                f"[View on Pinterest]({convert_to_url(pid)})\n\n"
            )

            catalog_img = fetch_image(pid)
            if catalog_img:
                gallery_items.append((catalog_img, f"#{match['rank']} ({score:.2f})"))

        md_parts.append("</div>\n\n")

    stats_html = (
        '<div class="stats-row">'
        f'<div class="stat-card"><div class="value">{len(result["items"])}</div>'
        f'<div class="label">Items Detected</div></div>'
        f'<div class="stat-card"><div class="value">{total_matches}</div>'
        f'<div class="label">Matches Found</div></div>'
        f'<div class="stat-card"><div class="value">{exact_count}</div>'
        f'<div class="label">Exact Matches</div></div>'
        f'<div class="stat-card"><div class="value">{best_score:.3f}</div>'
        f'<div class="label">Best Score</div></div>'
        "</div>"
    )

    explanation = stats_html + "\n\n" + "".join(md_parts)
    return (
        scene_img,
        gallery_items,
        gr.update(value=explanation),
        gr.update(visible=False),
    )


def run_demo(scene_signature: str, top_k: int):
    """Handler for the 'select scene / paste hash' path."""
    scene_signature = scene_signature.strip()
    if not scene_signature:
        return (
            None, [], gr.update(value=""),
            gr.update(value="Please enter or select a scene.", visible=True),
        )
    result = process_scene(scene_signature, top_k=int(top_k))
    return _render_result(result)


def run_upload(image, top_k: int):
    """Handler for the 'upload your own image' path."""
    if image is None:
        return (
            None, [], gr.update(value=""),
            gr.update(value="Please upload an image first.", visible=True),
        )
    result = process_image(image.convert("RGB"), top_k=int(top_k))
    return _render_result(result)


with gr.Blocks(title="Shop the Look") as demo:
    gr.HTML(
        '<div class="main-header">'
        "<h1>Shop the Look</h1>"
        "<p>Discover matching fashion products from the catalog. "
        "Select a scene from the dropdown or paste a signature hash, "
        "then click <b>Find Matching Products</b>.</p>"
        "</div>"
    )

    top_k_slider = gr.Slider(
        minimum=1, maximum=10, value=3, step=1,
        label="Results per detected item",
    )

    with gr.Tabs():
        # --- Tab 1: pick a known validation scene ---
        with gr.Tab("Select a scene"):
            with gr.Row():
                with gr.Column(scale=2):
                    scene_dropdown = gr.Dropdown(
                        choices=ALL_SCENES,
                        label="Select a Scene",
                        info=f"{len(ALL_SCENES)} validation scenes available",
                        filterable=True,
                        allow_custom_value=True,
                    )
                    sig_input = gr.Textbox(
                        label="Or paste a signature hash",
                        placeholder="e.g. cdab9160072dd1800038227960ff6467",
                        max_lines=1,
                    )
                    run_btn = gr.Button(
                        "Find Matching Products",
                        variant="primary",
                        size="lg",
                    )
                with gr.Column(scale=1):
                    scene_preview = gr.Image(
                        label="Scene Preview",
                        height=300,
                        interactive=False,
                    )

        # --- Tab 2: upload your own image ---
        with gr.Tab("Upload your own image"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.Image(
                        label="Upload a fashion photo",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=300,
                    )
                with gr.Column(scale=1):
                    upload_btn = gr.Button(
                        "Find Matching Products",
                        variant="primary",
                        size="lg",
                    )
                    gr.Markdown(
                        "Upload any photo of an outfit or garment. Note: matches "
                        "come from a 5k-product catalog, so results may be *similar* "
                        "rather than exact."
                    )

    error_box = gr.Textbox(label="Error", visible=False, interactive=False)
    gr.Markdown(
        "**Pipeline:** Scene image &rarr; YOLOS Fashionpedia detection "
        "&rarr; FashionCLIP embedding &rarr; FAISS cosine search "
        "&rarr; Attribute explanations"
    )

    # --- Results ---
    gr.Markdown("---")
    gallery = gr.Gallery(
        label="Detected Crops & Matched Products",
        columns=4,
        height=420,
        object_fit="contain",
    )
    explanation = gr.Markdown(label="Match Details")

    # --- Wiring ---

    # Dropdown selection updates textbox + preview
    scene_dropdown.change(
        fn=on_dropdown_select,
        inputs=[scene_dropdown],
        outputs=[sig_input, scene_preview],
    )

    # Typing in textbox updates preview (only if 32-char hash)
    sig_input.change(
        fn=on_textbox_change,
        inputs=[sig_input],
        outputs=[scene_preview],
    )

    # Run button
    run_btn.click(
        fn=run_demo,
        inputs=[sig_input, top_k_slider],
        outputs=[scene_preview, gallery, explanation, error_box],
    )

    # Enter key in textbox
    sig_input.submit(
        fn=run_demo,
        inputs=[sig_input, top_k_slider],
        outputs=[scene_preview, gallery, explanation, error_box],
    )

    # Upload button
    upload_btn.click(
        fn=run_upload,
        inputs=[upload_input, top_k_slider],
        outputs=[scene_preview, gallery, explanation, error_box],
    )


if __name__ == "__main__":
    demo.launch()
