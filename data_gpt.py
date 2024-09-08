from openai import OpenAI
from helper import get_var


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
    client = OpenAI(
        api_key=get_var("OPENAI_API_KEY"),
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=4096,
    )

    return completion.choices[0].message.content.strip()
