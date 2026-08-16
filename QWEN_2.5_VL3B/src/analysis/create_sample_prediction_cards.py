"""
Create curated prediction cards for the Assamese Musical Instrument VLM project.

Purpose
-------
Generate publication-quality qualitative prediction cards from the categorized
test predictions CSV.

Outputs
-------
assets/sample_predictions/
    correct_01.png
    correct_02.png
    question_misunderstanding_01.png
    ...
    mixed_attribute_02.png
    sample_predictions.csv

Expected prediction CSV columns
--------------------------------
image
instrument
question_id
concept
question
ground_truth
prediction
failure_category
reason
predicted_attribute
attribute_confidence
attribute_reason

Project
-------
Assamese Musical Instrument VLM
Qwen2.5-VL-3B-Instruct
"""


from pathlib import Path
import re
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# CONFIGURATION
PROJECT_ROOT = Path(
    r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B"
)

PREDICTION_CSV = (
    PROJECT_ROOT
    / "inference"
    / "categorized_preds_with_attribute.csv"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "dataset_32images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "assets"
    / "sample_predictions"
)

N_SAMPLES_PER_CATEGORY = 2

# Card dimensions
CARD_WIDTH = 1800
CARD_HEIGHT = 1250

# Image dimensions
IMAGE_WIDTH = 650
IMAGE_HEIGHT = 420

# Rendering
BACKGROUND = "#F8FAFC"
WHITE = "#FFFFFF"
TEXT = "#172033"
MUTED = "#64748B"
BORDER = "#CBD5E1"
BLUE = "#2563EB"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
ORANGE = "#EA580C"
RED = "#DC2626"

# Category accent colors
CATEGORY_COLORS = {
    "Correct": "#16A34A",
    "Question Misunderstanding": "#7C3AED",
    "Hallucination": "#DC2626",
    "Partial Answer / Incomplete Answer": "#EA580C",
    "Truncation": "#D97706",
    "Repetition": "#2563EB",
    "Mixed Attribute": "#C026D3",
}


# CATEGORY ORDER
CATEGORY_ORDER = [
    "Correct",
    "Question Misunderstanding",
    "Hallucination",
    "Partial Answer / Incomplete Answer",
    "Truncation",
    "Repetition",
    "Mixed Attribute",
]


# FONT HANDLING
def get_font(size: int, bold: bool = False):
    """
    Load a professional Windows font.

    Falls back to DejaVu Sans if Arial is unavailable.
    """

    candidates = []

    if bold:
        candidates.extend([
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf",
        ])
    else:
        candidates.extend([
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ])

    candidates.append(
        r"C:\Windows\Fonts\DejaVuSans.ttf"
    )

    for font_path in candidates:
        path = Path(font_path)

        if path.exists():
            return ImageFont.truetype(str(path), size=size)

    return ImageFont.load_default()


FONT_TITLE = get_font(38, bold=True)
FONT_CATEGORY = get_font(25, bold=True)
FONT_SECTION = get_font(20, bold=True)
FONT_BODY = get_font(24)
FONT_BODY_BOLD = get_font(24, bold=True)
FONT_SMALL = get_font(19)
FONT_SMALL_BOLD = get_font(19, bold=True)
FONT_TINY = get_font(16)


# TEXT UTILITIES
def wrap_text(draw, text, font, max_width):
    """
    Wrap text based on rendered pixel width rather than character count.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip()

    if not text:
        return ""

    words = text.split()

    lines = []
    current = ""

    for word in words:

        candidate = word if not current else f"{current} {word}"

        bbox = draw.textbbox(
            (0, 0),
            candidate,
            font=font
        )

        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    return "\n".join(lines)


def truncate_text(text, max_chars=900):
    """
    Prevent extremely long model outputs from destroying the card layout.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "…"


# CATEGORY SLUG
def slugify_category(category):
    """
    Convert category name into a clean filename.
    """

    category = category.lower().strip()

    category = category.replace("/", "_")

    category = re.sub(
        r"[^a-z0-9]+",
        "_",
        category
    )

    return category.strip("_")


# IMAGE PATH RESOLUTION
def resolve_image_path(row):
    """
    Convert the Kaggle image path stored in the CSV into the local project path.

    Example:

    Kaggle:
    /kaggle/.../bahi/test/bahi_test1.jpg

    Local:
    dataset/dataset_32images/bahi/test/bahi_test1.jpg
    """

    filename = Path(str(row["image"])).name

    instrument = str(row["instrument"]).strip()

    # The CSV uses "test". Your dataset directory uses the same naming.
    split = None

    raw_path = str(row["image"]).lower()

    if "/test/" in raw_path:
        split = "test"

    elif "/train/" in raw_path:
        split = "train"

    elif "/validation/" in raw_path:
        split = "validation"

    elif "/val/" in raw_path:
        split = "validation"

    if split is None:
        raise ValueError(
            f"Could not determine dataset split for image: {row['image']}"
        )

    candidate = (
        DATASET_ROOT
        / instrument
        / split
        / filename
    )

    if candidate.exists():
        return candidate

    # Some datasets may use "val" instead of "validation".
    if split == "validation":

        candidate_val = (
            DATASET_ROOT
            / instrument
            / "val"
            / filename
        )

        if candidate_val.exists():
            return candidate_val

    raise FileNotFoundError(
        "\nImage not found.\n"
        f"CSV image path: {row['image']}\n"
        f"Expected local path: {candidate}\n"
    )


# DRAWING UTILITIES
def rounded_rectangle(
    draw,
    xy,
    radius,
    fill,
    outline=None,
    width=1
):
    draw.rounded_rectangle(
        xy,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width
    )


def draw_section(
    draw,
    x,
    y,
    width,
    title,
    content,
    title_font=FONT_SECTION,
    body_font=FONT_BODY,
    content_color=TEXT,
    max_lines=None,
):
    """
    Draw a labeled information section.
    """

    draw.text(
        (x, y),
        title.upper(),
        font=title_font,
        fill=MUTED
    )

    y += 32

    content = truncate_text(content)

    wrapped = wrap_text(
        draw,
        content,
        body_font,
        width
    )

    lines = wrapped.split("\n")

    if max_lines is not None:
        lines = lines[:max_lines]

    line_height = body_font.size + 9

    for line in lines:
        draw.text(
            (x, y),
            line,
            font=body_font,
            fill=content_color
        )

        y += line_height

    return y


def fit_image(image, max_width, max_height):
    """
    Resize image while preserving aspect ratio.
    """

    image = image.convert("RGB")

    ratio = min(
        max_width / image.width,
        max_height / image.height
    )

    new_size = (
        max(1, int(image.width * ratio)),
        max(1, int(image.height * ratio))
    )

    return image.resize(
        new_size,
        Image.Resampling.LANCZOS
    )


# CREATE CARD
def create_prediction_card(row, output_path, category_index):
    """
    Create one professional prediction card.
    """

    image_path = resolve_image_path(row)

    original = Image.open(image_path)

    image = fit_image(
        original,
        IMAGE_WIDTH,
        IMAGE_HEIGHT
    )

    canvas = Image.new(
        "RGB",
        (CARD_WIDTH, CARD_HEIGHT),
        BACKGROUND
    )

    draw = ImageDraw.Draw(canvas)

    category = str(row["failure_category"]).strip()

    accent = CATEGORY_COLORS.get(
        category,
        BLUE
    )

    # Header
    draw.text(
        (70, 50),
        "ASSAMESE MUSICAL INSTRUMENT VLM",
        font=FONT_SMALL_BOLD,
        fill=BLUE
    )

    draw.text(
        (70, 82),
        "Qualitative Prediction Analysis",
        font=FONT_TITLE,
        fill=TEXT
    )

    # Category badge
    badge_text = category.upper()

    badge_bbox = draw.textbbox(
        (0, 0),
        badge_text,
        font=FONT_CATEGORY
    )

    badge_width = (
        badge_bbox[2] -
        badge_bbox[0] +
        50
    )

    rounded_rectangle(
        draw,
        (
            CARD_WIDTH - badge_width - 70,
            60,
            CARD_WIDTH - 70,
            112
        ),
        radius=14,
        fill=accent
    )

    draw.text(
        (
            CARD_WIDTH - badge_width - 45,
            72
        ),
        badge_text,
        font=FONT_CATEGORY,
        fill=WHITE
    )

    # Divider
    draw.line(
        (70, 140, CARD_WIDTH - 70, 140),
        fill=BORDER,
        width=2
    )

    # Main image card
    image_x = 70
    image_y = 180

    image_card_w = 700
    image_card_h = 505

    rounded_rectangle(
        draw,
        (
            image_x,
            image_y,
            image_x + image_card_w,
            image_y + image_card_h
        ),
        radius=18,
        fill=WHITE,
        outline=BORDER,
        width=2
    )

    # Image background
    inner_x = image_x + 25
    inner_y = image_y + 25

    rounded_rectangle(
        draw,
        (
            inner_x,
            inner_y,
            inner_x + IMAGE_WIDTH,
            inner_y + IMAGE_HEIGHT
        ),
        radius=10,
        fill="#EEF2F7"
    )

    centered_x = (
        inner_x +
        (IMAGE_WIDTH - image.width) // 2
    )

    centered_y = (
        inner_y +
        (IMAGE_HEIGHT - image.height) // 2
    )

    canvas.paste(
        image,
        (centered_x, centered_y)
    )

    # Image metadata
    instrument = str(row["instrument"]).upper()
    question_id = str(row["question_id"])
    concept = str(row["concept"])

    image_label = (
        f"{instrument}  •  {question_id}  •  {concept}"
    )

    draw.text(
        (
            image_x + 25,
            image_y + 452
        ),
        image_label,
        font=FONT_SMALL_BOLD,
        fill=TEXT
    )

    # Prediction information panel
    panel_x = 815
    panel_y = 180
    panel_w = 915
    panel_h = 505

    rounded_rectangle(
        draw,
        (
            panel_x,
            panel_y,
            panel_x + panel_w,
            panel_y + panel_h
        ),
        radius=18,
        fill=WHITE,
        outline=BORDER,
        width=2
    )

    x = panel_x + 30
    y = panel_y + 28
    content_width = panel_w - 60

    y = draw_section(
        draw,
        x,
        y,
        content_width,
        "Question",
        row["question"],
        body_font=FONT_BODY,
        max_lines=3
    )

    y += 12

    # Ground truth
    y = draw_section(
        draw,
        x,
        y,
        content_width,
        "Ground Truth",
        row["ground_truth"],
        body_font=FONT_BODY,
        max_lines=3
    )

    y += 12

    # Prediction
    y = draw_section(
        draw,
        x,
        y,
        content_width,
        "Model Prediction",
        row["prediction"],
        body_font=FONT_BODY,
        max_lines=3
    )

    # Category-aware analysis panel
    analysis_y = 725

    rounded_rectangle(
        draw,
        (
            70,
            analysis_y,
            CARD_WIDTH - 70,
            1170
        ),
        radius=18,
        fill=WHITE,
        outline=BORDER,
        width=2
    )

    # Category
    draw.text(
        (100, analysis_y + 25),
        "FAILURE ANALYSIS",
        font=FONT_SECTION,
        fill=MUTED
    )

    draw.text(
        (100, analysis_y + 62),
        category,
        font=FONT_CATEGORY,
        fill=accent
    )

    # Reason
    draw.text(
        (100, analysis_y + 110),
        "Taxonomy Reason",
        font=FONT_SMALL_BOLD,
        fill=MUTED
    )

    reason = truncate_text(
        row["reason"],
        max_chars=500
    )

    reason_wrapped = wrap_text(
        draw,
        reason,
        FONT_SMALL,
        780
    )

    draw.multiline_text(
        (100, analysis_y + 140),
        reason_wrapped,
        font=FONT_SMALL,
        fill=TEXT,
        spacing=7
    )

    # Category-aware analysis
    attr_x = 940

    if category == "Correct":

        # Correct predictions do not require attribute analysis.
        # Instead, show a semantic evaluation summary using
        # information already present in the prediction CSV.

        draw.text(
            (attr_x, analysis_y + 25),
            "EVALUATION SUMMARY",
            font=FONT_SECTION,
            fill=MUTED
        )

        draw.text(
            (attr_x, analysis_y + 65),
            "✓ Correct Prediction",
            font=FONT_CATEGORY,
            fill=accent
        )

        draw.text(
            (attr_x, analysis_y + 115),
            "Concept",
            font=FONT_SMALL_BOLD,
            fill=MUTED
        )

        draw.text(
            (attr_x, analysis_y + 143),
            str(row["concept"]),
            font=FONT_BODY_BOLD,
            fill=TEXT
        )

        draw.text(
            (attr_x, analysis_y + 195),
            "Assessment",
            font=FONT_SMALL_BOLD,
            fill=MUTED
        )

        assessment = truncate_text(
            row["reason"],
            max_chars=500
        )

        assessment_wrapped = wrap_text(
            draw,
            assessment,
            FONT_SMALL,
            700
        )

        draw.multiline_text(
            (attr_x, analysis_y + 225),
            assessment_wrapped,
            font=FONT_SMALL,
            fill=TEXT,
            spacing=7
        )

    else:

        # Failure categories may contain attribute-level analysis.
        # Display it only when the CSV actually provides it.

        draw.text(
            (attr_x, analysis_y + 25),
            "ATTRIBUTE ANALYSIS",
            font=FONT_SECTION,
            fill=MUTED
        )

        predicted_attribute = row["predicted_attribute"]

        if pd.isna(predicted_attribute) or str(
            predicted_attribute
        ).strip() == "":
            predicted_attribute = "Not provided"

        confidence = row["attribute_confidence"]

        if pd.isna(confidence) or str(confidence).strip() == "":
            confidence_text = "Not provided"
        else:
            try:
                confidence_text = f"{float(confidence):.2f}"
            except (ValueError, TypeError):
                confidence_text = str(confidence)

        draw.text(
            (attr_x, analysis_y + 65),
            "Predicted Attribute",
            font=FONT_SMALL_BOLD,
            fill=MUTED
        )

        draw.text(
            (attr_x, analysis_y + 92),
            str(predicted_attribute),
            font=FONT_BODY_BOLD,
            fill=TEXT
        )

        draw.text(
            (attr_x + 300, analysis_y + 65),
            "Confidence",
            font=FONT_SMALL_BOLD,
            fill=MUTED
        )

        draw.text(
            (attr_x + 300, analysis_y + 92),
            confidence_text,
            font=FONT_BODY_BOLD,
            fill=accent
        )

        draw.text(
            (attr_x, analysis_y + 140),
            "Attribute Reason",
            font=FONT_SMALL_BOLD,
            fill=MUTED
        )

        attribute_reason = truncate_text(
            row["attribute_reason"],
            max_chars=420
        )

        if not attribute_reason:
            attribute_reason = (
                "No attribute-level analysis was recorded "
                "for this prediction."
            )

        attribute_wrapped = wrap_text(
            draw,
            attribute_reason,
            FONT_SMALL,
            700
        )

        draw.multiline_text(
            (attr_x, analysis_y + 170),
            attribute_wrapped,
            font=FONT_SMALL,
            fill=TEXT,
            spacing=7
        )

    # Footer
    footer = (
        "Qwen2.5-VL-3B-Instruct  •  Test-set qualitative example  •  "
        f"Category sample {category_index:02d}"
    )

    draw.text(
        (70, 1190),
        footer,
        font=FONT_TINY,
        fill=MUTED
    )

    # Save
    canvas.save(
        output_path,
        format="PNG",
        optimize=True
    )


# MAIN
def main():

    print("=" * 72)
    print("Sample Prediction Card Generator")
    print("=" * 72)

    # Validate paths
    if not PREDICTION_CSV.exists():

        raise FileNotFoundError(
            "\nPrediction CSV not found:\n"
            f"{PREDICTION_CSV}\n\n"
            "Check PREDICTION_CSV at the top of this script."
        )

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            "\nDataset directory not found:\n"
            f"{DATASET_ROOT}\n\n"
            "Check DATASET_ROOT at the top of this script."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load predictions
    df = pd.read_csv(
        PREDICTION_CSV
    )

    required_columns = [
        "image",
        "instrument",
        "question_id",
        "concept",
        "question",
        "ground_truth",
        "prediction",
        "failure_category",
        "reason",
        "predicted_attribute",
        "attribute_confidence",
        "attribute_reason",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required CSV columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(f"\nLoaded predictions: {len(df)}")

    # Show available categories
    print("\nCategory distribution:")

    counts = (
        df["failure_category"]
        .value_counts()
    )

    for category in CATEGORY_ORDER:

        print(
            f"  {category:<40} "
            f"{counts.get(category, 0)}"
        )

    # Select representative samples
    selected_rows = []

    print("\nSelecting samples...")

    for category in CATEGORY_ORDER:

        category_df = df[
            df["failure_category"].astype(str).str.strip()
            == category
        ].copy()

        if len(category_df) == 0:

            print(
                f"WARNING: No samples found for '{category}'"
            )

            continue

        # Deterministic selection.
        # This keeps generation reproducible.
        selected = category_df.head(
            N_SAMPLES_PER_CATEGORY
        )

        for _, row in selected.iterrows():
            selected_rows.append(row)

    # Generate cards
    metadata_rows = []

    category_counters = {}

    print("\nGenerating cards...")

    for row in selected_rows:

        category = str(
            row["failure_category"]
        ).strip()

        category_counters[category] = (
            category_counters.get(category, 0) + 1
        )

        number = category_counters[category]

        slug = slugify_category(
            category
        )

        filename = (
            f"{slug}_{number:02d}.png"
        )

        output_path = (
            OUTPUT_DIR
            / filename
        )

        try:

            create_prediction_card(
                row,
                output_path,
                number
            )

            print(
                f"  ✓ {filename}"
            )

            # Store complete metadata.
            metadata = row.to_dict()

            metadata["card_filename"] = filename

            metadata_rows.append(
                metadata
            )

        except Exception as error:

            print(
                f"  ✗ FAILED: {filename}"
            )

            print(
                f"    {error}"
            )

    # Save metadata CSV
    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_path = (
        OUTPUT_DIR
        / "sample_predictions.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig"
    )

    # Final report
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Cards created : {len(metadata_df)}"
    )

    print(
        f"Output folder : {OUTPUT_DIR}"
    )

    print(
        f"Metadata CSV  : {metadata_path}"
    )

    print("\nGenerated categories:")

    for category in CATEGORY_ORDER:

        count = len(
            metadata_df[
                metadata_df["failure_category"]
                == category
            ]
        )

        print(
            f"  {category:<40} {count}"
        )

    print("\n")


if __name__ == "__main__":
    main()