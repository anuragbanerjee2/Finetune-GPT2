## create data 

import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import os
from pypdf import PdfReader
import json

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get output from GPT api
def get_answer(prompt, model_name='gpt-4o'):
    output = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return output.choices[0].message.content

# Prompt GPT to give QA pairs from one page of text
def get_qa_pair(text, count):
    prompt = f"Given the whole page text from some research paper, give {count} questions from the page text only and the answer.\
    Give it strictly in list of json format with keys question and answer. Donot format the json with newlines. Page text:\n{text}"
    return get_answer(prompt=prompt)

# Generate Question-Answer pairs for the all the PDFs pagewise in JSON format
# using JSON for standarising the training data
def get_qa_pairs_from_papers(folder_path,page_pairs = 1):
    qa_pairs = []
    file_list = os.listdir(folder_path)
    for file in file_list:
        full_path = os.path.join(folder_path,file)
        reader = PdfReader(full_path)
        for page in reader.pages:
            page_text = page.extract_text()
            try:
                qa_pairs.extend(json.loads(get_qa_pair(page_text,page_pairs)))
            except:
                pass
    return qa_pairs

# Convert the JSON into text format to train. 
# Intermediate JSON format is used to make the data consistent as LLMs tends to make mistakes in formats
def json_to_text(j):
    data = ''
    for pair in j:
        data+='[Q] '+pair['question']+' [A] '+pair['answer']+'\n'
    return data

# Input PDF folder and output training data in txt
def papers_to_data(folder_path,output_file,page_pairs):
    data = get_qa_pairs_from_papers(folder_path=folder_path,page_pairs=page_pairs)
    text_data = json_to_text(data)
    with open(output_file,'w') as f:
        f.write(text_data)
