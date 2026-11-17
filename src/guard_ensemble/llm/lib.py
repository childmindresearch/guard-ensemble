from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.renderers.chat_utils import ChatCompletionMessageParam


def create_message(text: str) -> list["ChatCompletionMessageParam"]:
    param: ChatCompletionMessageParam = {"role": "user", "content": text}
    return [param]


def create_messages(texts: list[str]):
    return [create_message(t) for t in texts]
