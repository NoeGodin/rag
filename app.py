import os

import chainlit as cl

from core.retrieve import retrieve
from core.generate import generate_stream


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Qui etait Staline ?",
            message="Qui etait Staline et comment a-t-il pris le pouvoir ?",
        ),
        cl.Starter(
            label="Les regimes totalitaires",
            message="Quelles sont les caracteristiques communes des regimes totalitaires ?",
        ),
        cl.Starter(
            label="Propagande et controle",
            message="Comment les dictateurs utilisent-ils la propagande pour maintenir le pouvoir ?",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    docs = retrieve(message.content)
    context = "\n\n".join(doc.page_content for doc in docs)

    msg = cl.Message(content="")
    await msg.send()

    for token in generate_stream(context, message.content):
        await msg.stream_token(token)

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
