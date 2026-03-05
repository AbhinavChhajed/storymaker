import streamlit as st
import pvporcupine
import pyaudio
import struct
import os
import wave
from dotenv import load_dotenv
import time
import whisper # Added import
import time
from pathlib import Path
from PIL import Image
from pydub import AudioSegment

# --- Configuration ---
load_dotenv()
ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
OUTPUT_FILE = "recorded_audio.wav"
audio_text = None
st.title("🎙️ StoryMaker")

# --- Initialize session state ---
if "listening" not in st.session_state:
    st.session_state.listening = False

# --- Helper Functions ---

@st.cache_data
def get_input_devices():
    """Gets a dictionary of available audio input devices {name: index}."""
    pa = pyaudio.PyAudio()
    devices = {}
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        if device_info.get('maxInputChannels') > 0:
            name = f"Index {i}: {device_info.get('name')}"
            devices[name] = i
    pa.terminate()
    return devices

@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model, caching it for performance."""
    # Using "base" is a good balance of speed and accuracy.
    # Other options: "tiny", "small", "medium", "large"
    model = whisper.load_model("base")
    return model

def transcribe_audio(file_path):
    st.info("Transcribing audio... ")
    model = load_whisper_model()
    try:
        result = model.transcribe(file_path)
        transcribed_text = result["text"]
        st.session_state.transcribed_text = transcribed_text
        st.text_area("Result", transcribed_text, height=150)
    except Exception as e:  
        st.error(f"Error during transcription: {e}")


# --- Core Application Logic ---

def start_listening(device_index):
    """
    Initializes Porcupine and PyAudio to listen for a wake word on a specific device.
    """
    porcupine = None
    pa = None
    audio_stream = None
    status_placeholder = st.empty()

    try:
        porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=["jarvis"])
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
            input_device_index=device_index
        )
        
        last_update_time = time.time()
        dot_count = 0

        while st.session_state.get("listening", False):
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm_unpacked)
            
            current_time = time.time()
            if current_time - last_update_time > 0.4:
                dot_count = (dot_count + 1) % 4
                dots = "." * dot_count
                status_placeholder.info(f"Listening for 'Porcupine'{dots.ljust(3)}")
                last_update_time = current_time

            if keyword_index >= 0:
                status_placeholder.success("✅ Wake word detected!")
                
                audio_stream.stop_stream()
                audio_stream.close()
                audio_stream = None
                
                porcupine.delete()
                porcupine = None

                record_audio(device_index,pa, duration=10) 
                
                st.session_state.listening = False

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check your microphone connection and Picovoice Access Key.")
    finally:
        if audio_stream is not None and not audio_stream.is_stopped():
            audio_stream.stop_stream()
        if audio_stream is not None:
            audio_stream.close()
        if porcupine is not None:
            porcupine.delete()
        if pa is not None:
            pa.terminate()
        
        status_placeholder.empty()

def record_audio(device_index,pa, duration=10):
    """Record audio and then transcribe it."""
    st.write(f"🎤 Recording for {duration} seconds...")
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
        input_device_index=device_index
    )

    frames = []
    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))

    st.success(f"🎧 Audio recorded and saved as {OUTPUT_FILE}")
    st.audio(OUTPUT_FILE)

    # Call the transcription function
    transcribe_audio(OUTPUT_FILE)

# --- UI Controls ---
st.markdown("### 1. Select your microphone")
input_devices = get_input_devices()
device_names = list(input_devices.keys())

try:
    default_selection = next(i for i, s in enumerate(device_names) if 'Realtek' in s and 'Array' in s)
except StopIteration:
    default_selection = 0

selected_device_name = st.selectbox(
    "Choose your microphone from the list below.",
    options=device_names,
    index=default_selection,
    help="Your built-in laptop microphone is likely the 'Microphone Array (Realtek...)'."
)
selected_device_index = input_devices[selected_device_name]

st.markdown("### 2. Start Listening")

col1, col2 = st.columns(2)

if col1.button("▶️ Start Listening", use_container_width=True):
    if not st.session_state.listening:
        st.session_state.listening = True
        start_listening(device_index=selected_device_index)

if col2.button("⏹️ Stop Listening", use_container_width=True):
    st.session_state.listening = False
    st.warning("Stopped listening.")



############################################################################
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from gtts import gTTS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from huggingface_hub import InferenceClient
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
import os
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class VisualMemory(BaseModel):
    characters: list[dict] = Field(description="List of characters, each containing 'name' and 'description' keys.")
    setting: str = Field(description="Location or environment details of the scene.")
    objects: list[str] = Field(description="Important props or items mentioned.")
    style: str = Field(description="Artistic style or mood. Default to 'child-friendly storybook illustration' if not specified.")

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

correct_text_prompt = PromptTemplate(
    template="""
You are a professional transcript editor. Your task is to polish this raw speech-to-text output while keeping the original meaning.

Rules:
1. Fix punctuation and capitalization.
2. Correct grammar and spelling.
3. Resolve common homophone mistakes (their/there/they're, to/too/two, etc.).
4. Remove filler words (um, uh, like, you know) unless meaningful.
5. Improve readability (merge fragments, split run-ons).
6. Do not change meaning, tone, or vocabulary.

Transcript to correct:
{text}""",input_variables=["text"]
)

story_prompt = PromptTemplate(
    template="""
    You are a creative storyteller. Write a short, engaging story based on this idea:

"{idea}"

Guidelines:
- Audience: children aged 6-12.
- Length: 2-3 short paragraphs.
- Clear beginning, middle, and end.
- Simple vocabulary with vivid imagery.
- Include interesting characters with names.
- Emotional depth without being too complex.
- Ensure the story is uplifting and easy to follow based in a single setting.

    """
    ,input_variables=['idea']
)
story_chain = correct_text_prompt|llm|story_prompt|llm
if "transcribed_text" in st.session_state:
    user_idea = st.session_state.transcribed_text
else:
    user_idea = st.text_area("Or type your idea here:", "")

        
story = story_chain.invoke({"text":user_idea})
st.write("\n--- Your Story ---\n")
st.write(story.content)

tts = gTTS(story.content)
audio_file = "audio_file.mp3"
tts.save(audio_file)

parser = JsonOutputParser(pydantic_object=VisualMemory)

consistancy_prompt = PromptTemplate(
    template="""
You are an assistant helping to create a consistent picture book.
Task: From the following story text, extract a structured "visual memory".

{format_instructions}

story text:
{text}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

consistancy_chain = consistancy_prompt | llm | parser
try:
    extracted_info = consistancy_chain.invoke({"text": story.content})
except Exception as e:
    st.error(f"Failed to parse the visual memory: {e}")
    extracted_info = {"characters": [], "setting": "A whimsical landscape", "objects": [], "style": "storybook illustration"}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20,separators=["\n\n", ".", "!", "?", ",", " "])

chunks = text_splitter.split_text(story.content)

list_of_chunks = []
for c in chunks:
    list_of_chunks.append(c)


def make_image_prompt(extracted_info, chunk_text):
    """
    Build a high-quality image generation prompt from story chunk + extracted info.
    Ensures descriptive clarity, consistency, and alignment with narrative tone.
    """
    characters = "\n".join(
        [f"- {c['name']}: {c['description']}" for c in extracted_info.get("characters", [])]
    ) or "None specified"
    setting = extracted_info.get("setting", "Unspecified setting")
    objects = ", ".join(extracted_info.get("objects", [])) or "No specific objects"
    style = extracted_info.get("style", "Illustrated storybook style")

    prompt = f"""
    Illustration prompt:

A colorful storybook illustration of the following scene:
"{chunk_text}"

Include:
- Characters: {characters}
- Setting: {setting}
- Objects: {objects}

Style: {style}, soft, warm, child-friendly, consistent with earlier images.

Requirements:
- Maintain character consistency (appearance, clothing).
- Show key objects clearly.
- Balanced composition.
- No text, dialogue, or unrelated elements.

    """

    return prompt.strip()


image_generation_prompt_list = []
for i in list_of_chunks:
    image_generation_prompt_list.append(make_image_prompt(extracted_info,i))


# # Pick device automatically
# device = "cuda" if torch.cuda.is_available() else "cpu"
# pipeline = StableDiffusionPipeline.from_pretrained(
#     "./sd-v1-5-local",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32
# ).to(device)

# pipeline.enable_attention_slicing()

# # Ensure output directory exists
# os.makedirs("images", exist_ok=True)
# # Keep prompts short (<77 tokens each)
# for i, prompt in enumerate(image_generation_prompt_list):
#     # Truncate prompt for safety (77 tokens ~ 200 characters as rough cutoff)
#     prompt = prompt[:200]  
    
#     # Generate image
#     image = pipeline(
#         prompt,
#         height=512,
#         width=512
#     ).images[0]
    
#     # Save output
#     image.save(f"images/{i}.png")

hf_token = ""

client = InferenceClient(token=hf_token)
os.makedirs("images", exist_ok=True)

for i, prompt in enumerate(image_generation_prompt_list):
    prompt = prompt[:600]  
    
    try:
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-2-1" 
        )
        
        image.save(f"images/{i+1}.png")
        
    except Exception as e:
        st.error(f"Error generating image {i+1}: {e}")

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

