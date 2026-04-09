import base64
import json
from rich import print
import sqlite3

# 兼容不同版本的 LangChain
try:
    from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
except ImportError:
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from dilu.driver_agent.model_provider import build_chat_model, get_model_label
from scenario.scenario import Scenario


class OutputParser:
    def __init__(self, sce: Scenario, temperature: float = 0.0) -> None:
        self.sce = sce
        self.temperature = temperature
        print(f"Output parser uses {get_model_label()}")
        self.llm = build_chat_model(
            temperature=temperature,
            max_tokens=1000,
            request_timeout=60,
            streaming=False,
        )
        # todo: put into a yaml file
        self.response_schemas = [
            ResponseSchema(
                name="action_id", description=f"output the id(int) of the decision. The comparative table is:  {{ 0: 'change_lane_left', 1: 'keep_speed or idle', 2: 'change_lane_right', 3: 'accelerate or faster',4: 'decelerate or slower'}} . For example, if the ego car wants to keep speed, please output 1 as a int."),
            ResponseSchema(
                name="action_name", description=f"output the name(str) of the decision. MUST consist with previous \"action_id\". The comparative table is:  {{ 0: 'change_lane_left', 1: 'keep_speed', 2: 'change_lane_right', 3: 'accelerate',4: 'decelerate'}} . For example, if the action_id is 3, please output 'Accelerate' as a str."),
            ResponseSchema(
                name="explanation", description=f"Explain for the driver why you make such decision in 40 words.")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas)
        self.format_instructions = self.output_parser.get_format_instructions()

    def agentRun(self, final_results: dict) -> str:
        print('[green]Output parser is running...[/green]')
        prompt_template = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "parse the problem response follow the format instruction.\nformat_instructions:{format_instructions}\n response: {answer}")
            ],
            input_variables=["answer"],
            partial_variables={"format_instructions": self.format_instructions}
        )
        input = prompt_template.format_prompt(
            answer=final_results['answer']+final_results['thoughts'])
        output = self.llm(input.to_messages())

        self.parseredOutput = self.output_parser.parse(output.content)
        self.dataCommit()
        print("Finish output agent.")
        return self.parseredOutput

    def dataCommit(self):
        conn = sqlite3.connect(self.sce.database)
        cur = conn.cursor()
        parseredOutput = json.dumps(self.parseredOutput)
        base64Output = base64.b64encode(
            parseredOutput.encode('utf-8')).decode('utf-8')
        cur.execute(
            """UPDATE decisionINFO SET outputParser ='{}' WHERE frame ={};""".format(
                base64Output, self.sce.frame
            )
        )
        conn.commit()
        conn.close()
