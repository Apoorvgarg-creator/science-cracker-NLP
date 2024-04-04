import os
import warnings
from typing import Dict

import torch
from transformers import pipeline


# create pieline for QA
from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass


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
        print(text)
        qa_response = answer(text)
        print(qa_response)
        response = qa_response
        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))


def answer(question):
    hf = pipeline(
        model="EleutherAI/gpt-neo-2.7B",
        task="text-generation",
        max_length=200,
        pad_token_id=50256
    )
    response = hf(question)
    return response[0]['generated_text']
