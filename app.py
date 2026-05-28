import chainlit as cl

from core.retrieve import retrieve
from core.generate import generate_stream

_GREETINGS = {"hello", "hi", "hey", "salut", "bonjour", "coucou", "yo", "bonsoir"}


def _is_greeting(text: str) -> bool:
    return text.strip().strip("!?.").lower() in _GREETINGS


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content="Bienvenue, camarade. Pose ta question sur les dictateurs et les regimes autoritaires."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    greeting = _is_greeting(message.content)

    if greeting:
        docs = []
        context = ""
    else:
        docs = retrieve(message.content)
        context = "\n\n".join(doc.page_content for doc in docs)

    msg = cl.Message(content="")
    await msg.send()

    for token in generate_stream(context, message.content):
        await msg.stream_token(token)

    # Append sources as a footer after the response
    if docs:
        source_names = set()
        source_lines = []
        for doc in docs:
            name = doc.metadata.get("source", "inconnu")
            if name not in source_names:
                source_names.add(name)
                source_lines.append(f"- {name}")
        msg.content += "\n\n---\n**Sources :**\n" + "\n".join(source_lines)

    await msg.update()
