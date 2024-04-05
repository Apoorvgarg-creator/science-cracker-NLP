import os
import warnings
from typing import Dict
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


# create pieline for QA
from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from util import answer_question



############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []
    print(request)
    for text in request.text:
        qa_response = answer_question(text, modelMethod="GEMMA")
        response = qa_response
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))
