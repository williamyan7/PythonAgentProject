from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import china_engine, japan_engine




population_data_path = os.path.join("data", "WorldPopulation2023.csv")
population_df = pd.read_csv(population_data_path)
anime_data_path = os.path.join("data", "anime_ratings.csv")
anime_df = pd.read_csv(anime_data_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
anime_query_engine = PandasQueryEngine(
    df=anime_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
anime_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This gives information about world population and demographics"
        )
    ),
    QueryEngineTool(
        query_engine=anime_query_engine,
        metadata=ToolMetadata(
            name="anime_data",
            description="This datasets contains data on popular animes and how they are rated"
        )
    ),
    QueryEngineTool(
        query_engine=china_engine,
        metadata=ToolMetadata(
            name="China info",
            description="This document contains information about China"
        )
    ),
    QueryEngineTool(
        query_engine=japan_engine,
        metadata=ToolMetadata(
            name="Japan info",
            description="This document contains information about Japan"
        )
    )
]

llm = OpenAI(model="gpt-3.5-turbo") # test
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True,context=context)

while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        print(agent.query(prompt))
    except:
        print("I'm sorry, I don't have the tools or data needed to answer that question. Please ask another question.")