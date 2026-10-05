from pathlib import Path
import tempfile

import joblib
import streamlit as st

from src.demo_utils import (
    extract_features,
    extract_first_chorus,
    find_cached_chorus,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "lda_hit_predictor.joblib"

st.set_page_config(
    page_title="Hit Song Predictor",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Hit Song Prediction")

st.caption(
    "Upload a full song. The system identifies its chorus, "
    "extracts acoustic features, and predicts whether it is a HIT or NON-HIT."
)


if not MODEL_FILE.exists():
    st.error(
        "Model file not found. First run:\n\n"
        "python src/train_demo_model.py"
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


model = load_model()


uploaded = st.file_uploader(
    "Upload a full song",
    type=["mp3", "wav", "m4a", "flac"]
)


if uploaded is not None:

    st.audio(uploaded.getvalue())

    if st.button(
        "Analyze Song",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Analyzing song..."):

                # --------------------------------------------------
                # FIRST CHECK IF CHORUS IS ALREADY CACHED
                # --------------------------------------------------

                cached_chorus = find_cached_chorus(
                    uploaded.name,
                    PROJECT_ROOT
                )

                if cached_chorus is not None:

                    chorus_file = cached_chorus

                    source_message = (
                        "Cached chorus found — using previously "
                        "extracted chorus."
                    )

                    start = None
                    end = None

                # --------------------------------------------------
                # OTHERWISE RUN ALL-IN-ONE
                # --------------------------------------------------

                else:

                    source_message = (
                        "Song not found in chorus cache — "
                        "running automatic chorus detection."
                    )

                    with tempfile.TemporaryDirectory() as temp_dir:

                        temp_dir = Path(temp_dir)

                        suffix = (
                            Path(uploaded.name).suffix
                            or ".mp3"
                        )

                        input_file = (
                            temp_dir /
                            f"uploaded_song{suffix}"
                        )

                        input_file.write_bytes(
                            uploaded.getvalue()
                        )

                        chorus_file, start, end = (
                            extract_first_chorus(
                                input_file,
                                temp_dir,
                                PROJECT_ROOT
                            )
                        )

                        # Need bytes before temp directory disappears
                        chorus_bytes = chorus_file.read_bytes()

                        features = (
                            extract_features(chorus_file)
                            .reshape(1, -1)
                        )

                # --------------------------------------------------
                # IF CACHED CHORUS WAS USED
                # --------------------------------------------------

                if cached_chorus is not None:

                    chorus_bytes = (
                        chorus_file.read_bytes()
                    )

                    features = (
                        extract_features(chorus_file)
                        .reshape(1, -1)
                    )

                # --------------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------------

                prediction = int(
                    model.predict(features)[0]
                )

                probabilities = (
                    model.predict_proba(features)[0]
                )

                classes = list(model.classes_)

                hit_probability = float(
                    probabilities[
                        classes.index(1)
                    ]
                )


            # ======================================================
            # RESULTS
            # ======================================================

            st.success(source_message)

            if start is not None:
                st.write(
                    f"Detected chorus: "
                    f"{start:.1f}s – {end:.1f}s"
                )

            st.subheader("🎶 Chorus Used")

            st.audio(
                chorus_bytes,
                format="audio/wav"
            )

            st.divider()

            if prediction == 1:

                st.markdown(
                    "# 🔥 HIT"
                )

            else:

                st.markdown(
                    "# 🎧 NON-HIT"
                )

            st.metric(
                "Hit Probability",
                f"{hit_probability * 100:.1f}%"
            )

            st.caption(
                "Model: Linear Discriminant Analysis (LDA)"
            )


            with st.expander(
                "How was this prediction made?"
            ):

                st.write(
                    """
                    1. The uploaded full song is received.
                    2. The system checks whether its chorus has
                       already been extracted.
                    3. If available, the cached chorus is used.
                    4. Otherwise, All-In-One automatically detects
                       the first chorus.
                    5. Up to 15 seconds of the chorus are analyzed.
                    6. 518 acoustic features are extracted using Librosa.
                    7. PCA reduces the feature dimensions.
                    8. Linear Discriminant Analysis predicts
                       HIT or NON-HIT.
                    """
                )


        except Exception as exc:

            st.error(
                f"Could not analyze this song:\n\n{exc}"
            )