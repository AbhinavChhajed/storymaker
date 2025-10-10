import streamlit as st
from pathlib import Path 
from PIL import Image, ImageDraw
import time
from gtts import gTTS
from pydub import AudioSegment
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
list_of_chunks = ["Barnaby the Beagle, a bouncy ball of fur, loved to chase butterflies. Mittens, a sleek black cat with emerald eyes, preferred sunbeams and naps",
                  ". They were the unlikeliest of friends, living in the same cozy house, but usually keeping to themselves.",
                  "One sunny afternoon, Barnaby spotted a particularly magnificent monarch butterfly fluttering near the rose bushes",
". He barked excitedly, his tail a blur, and took off in hot pursuit",
". The butterfly led him on a merry chase, right into a thorny thicket! Barnaby yelped, tangled and stuck.",
"Hearing Barnaby's cries, Mittens, usually aloof, sprang into action",
". With surprising agility, she weaved through the branches, her sleek body navigating the thorns with ease",
". She reached Barnaby, gently nudging his head with her nose, and helped him untangle himself.",
"Free at last, Barnaby licked Mittens's face gratefully. Mittens, purring softly, rubbed against his head",
". They were the best of friends, a dog and a cat, proving that even opposites can find happiness together."]
    

def generate_story_and_images(chunks):
    story_chunks = chunks

    # Create folders
    Path("images").mkdir(exist_ok=True)
    Path("audio").mkdir(exist_ok=True)

    image_paths, audio_paths, durations = [], [], []

    for i, text in enumerate(story_chunks):
        # --- Generate placeholder image ---
        img_path = f"images/{i+1}.png"
        image_paths.append(img_path)

        # --- Generate TTS audio ---
        audio_path = f"audio/{i+1}.mp3"
        tts = gTTS(text=text, lang="en")
        tts.save(audio_path)

        # Get audio duration
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # in seconds
        audio_paths.append(audio_path)
        durations.append(duration)

    return story_chunks, image_paths, audio_paths, durations


# --- Generate Story & Picture Book ---
if st.button("Generate Story & Picture Book"):
    with st.spinner("Generating story, images, and audio..."):
        story_chunks, image_paths, audio_paths, durations = generate_story_and_images(list_of_chunks)

    st.session_state.story_chunks = story_chunks
    st.session_state.image_paths = image_paths
    st.session_state.audio_paths = audio_paths
    st.session_state.durations = durations
    st.success("✅ Story and picture book generated!")

    # Show preview
    st.subheader("Generated Story Preview")
    for text, img_path, audio_path in zip(story_chunks, image_paths, audio_paths):
        st.image(img_path, use_container_width=True)
        st.write(text)
        st.audio(audio_path)


# --- Storytelling Mode ---
if "story_chunks" in st.session_state and st.button("Start Storytelling"):
    story_chunks = st.session_state.story_chunks
    image_paths = st.session_state.image_paths
    audio_paths = st.session_state.audio_paths
    durations = st.session_state.durations

    storytelling_placeholder = st.empty()
    image_placeholder = st.empty()
    audio_placeholder = st.empty()

    for text, img_path, audio_path, duration in zip(story_chunks, image_paths, audio_paths, durations):
        with storytelling_placeholder:
            st.markdown(f"### {text}")
        with image_placeholder:
            st.image(img_path, use_container_width=True)
        with audio_placeholder:
            st.audio(audio_path, autoplay=True)

        # Wait for audio to finish before moving on
        time.sleep(duration)
