from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM,AutoModelForCausalLM, AutoTokenizer

def answer_question(question, modelMethod="FLAN"):
    if modelMethod == "CHATGPT":
        client = OpenAI(api_key="")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": question}
                ])
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    elif modelMethod == "FLAN":
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    elif modelMethod == "PHI2":
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        inputs = tokenizer(question, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=200)
        text = tokenizer.batch_decode(outputs)
        return text



# answer_question("how to build a magnifying glass at home",modelMethod="")