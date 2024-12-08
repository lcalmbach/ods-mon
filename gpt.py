from openai import OpenAI
from helper import get_var
import streamlit as st


client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
)


def get_completion(system_prompt, user_prompt):
    """
    Generates a completion using OpenAI's GPT-4o model.
    Args:
        system_prompt (str): The system prompt for the chat completion.
        user_prompt (str): The user prompt for the chat completion.
    Returns:
        str: The generated completion content.
    Raises:
        None
    """

    model = "gpt-4o"
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=4096,
    )

    return completion.choices[0].message.content.strip()


def get_audio(text: str, audio_file: str) -> bool:
    with st.spinner("Generating audio..."):
        try:
            from openai import OpenAI

            response = client.audio.speech.create(
                model="tts-1", voice="alloy", input=text
            )
            # Assuming response.content contains the binary audio data
            with open(audio_file, "wb") as file:
                file.write(response.content)
            return True
        except Exception as e:
            print(e)
            return False
