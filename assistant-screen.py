import sys
import base64
import time
from threading import Lock, Thread

import cv2
import numpy as np
import mss
import openai
from cv2 import imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()

if len(sys.argv) > 1:
    lang_flag = sys.argv[1]
else:
    lang_flag = "0"

if lang_flag == "1":
    SYSTEM_PROMPT = """
Eres un asistente de voz en español que utiliza el historial de chat y una captura de pantalla para responder preguntas sobre el contenido visual de la pantalla.
Responde a las preguntas de forma concisa y directa, sin utilizar emoticonos.
Sé amable, servicial y muestra algo de personalidad.
"""
    recognizer_language = "es"
else:
    SYSTEM_PROMPT = """
You are a voice assistant that uses the chat history and a screen capture image to answer questions about the visual content of the screen.
Answer questions concisely and directly without using emoticons.
Be friendly, helpful, and show some personality.
"""
    recognizer_language = "en"
# --------------------------------------------------------------------

class ScreenStream:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.frame = self.capture_screen()
        self.running = False
        self.lock = Lock()

    def capture_screen(self):
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            frame = self.capture_screen()
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

screen_stream = ScreenStream().start()

model = ChatOpenAI(model="gpt-4o")
assistant = Assistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language=recognizer_language)
        assistant.answer(prompt, screen_stream.read(encode=True))
    except UnknownValueError:
        print("Error al procesar el audio.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

print("Assistant running. Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass

screen_stream.stop()
stop_listening(wait_for_stop=False)
