import os
import textwrap
import time
# 简单的消息类，无需依赖 langchain
class SystemMessage:
    def __init__(self, content: str):
        self.content = content
        self.type = "system"

class HumanMessage:
    def __init__(self, content: str):
        self.content = content
        self.type = "human"

from rich import print
from dilu.driver_agent.model_provider import build_chat_model, get_model_label


class ReflectionAgent:
    def __init__(
        self, temperature: float = 0.0, verbose: bool = False
    ) -> None:
        print(f"Reflection uses {get_model_label()}")
        self.llm = build_chat_model(
            temperature=temperature,
            max_tokens=1000,
            request_timeout=60,
            streaming=False,
        )

    def reflection(self, human_message: str, llm_response: str) -> str:
        delimiter = "####"
        system_message = textwrap.dedent(f"""\
        You are a mature driving assistant powered by a large language model. You can give accurate and correct advice for human drivers in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with the available actions allowed to take. 

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)
        human_message = textwrap.dedent(f"""\
            ``` Human Message ```
            {human_message}
            ``` Model Response ```
            {llm_response}

            Now, you know this action caused a collision after taking this action, which means there are mistakes in the reasoning that caused the wrong action.
            Please carefully check every reasoning step in the model response, find the mistake in the reasoning process, and output your corrected version of the model response.
            Your answer should use the following format:
            {delimiter} Analysis of the mistake:
            <Your analysis of the mistake in the reasoning process>
            {delimiter} What should the driving assistant do to avoid such errors in the future:
            <Your answer>
            {delimiter} Corrected version of the model response:
            <Your corrected version of the model response>
        """)

        print("Self-reflection is running, make take time...")
        start_time = time.time()
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        response = self.llm(messages)
        target_phrase = f"{delimiter} What should the driving assistant do to avoid such errors in the future:"
        substring = response.content[response.content.find(
            target_phrase)+len(target_phrase):].strip()
        corrected_memory = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"
        print("Reflection done. Time taken: {:.2f}s".format(
            time.time() - start_time))
        print("corrected_memory:", corrected_memory)

        return corrected_memory
