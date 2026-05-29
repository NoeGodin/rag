import os

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from core.retrieve import retrieve
from core.generate import generate_stream
from utils.guard import detect_injection, check_output_leak, REFUSAL_MESSAGE, LEAK_REFUSAL

@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("chat_history", [])

@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Qui était Staline ?",
            message="Qui était Staline et comment a-t-il pris le pouvoir ?",
        ),
        cl.Starter(
            label="Les régimes totalitaires",
            message="Quelles sont les caracteristiques communes des régimes totalitaires ?",
        ),
        cl.Starter(
            label="Propagande et contrôle",
            message="Comment les dictateurs utilisent-ils la propagande pour maintenir le pouvoir ?",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # prompt injection protection 1 (regex heuristics)
    if detect_injection(message.content):
        await cl.Message(content=REFUSAL_MESSAGE).send()
        return

    docs = retrieve(message.content)
    context = "\n\n".join(doc.page_content for doc in docs)
    history = cl.user_session.get("chat_history")

    msg = cl.Message(content="")
    await msg.send()
    history.append(HumanMessage(content=message.content))

    full_response = ""
    for token in generate_stream(context, message.content, history):
        await msg.stream_token(token)
        full_response += token

    # prompt injection protection 2 (canary token)
    if check_output_leak(full_response):
        msg.content = LEAK_REFUSAL
        await msg.update()
        return

    history.append(AIMessage(content=full_response))

    if docs:
        elements = []
        seen_sources = set()
        source_lines = []
        for doc in docs:
            name = os.path.basename(doc.metadata.get("source", "inconnu"))
            if name not in seen_sources:
                seen_sources.add(name)
                source_lines.append(f"- {name}")
                elements.append(
                    cl.Text(name=name, content=doc.page_content, display="side")
                )

        msg.elements = elements
        msg.content += "\n\n---\n**Sources :**\n" + "\n".join(source_lines)

    await msg.update()
