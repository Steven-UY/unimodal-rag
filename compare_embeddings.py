from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
from langchain.evaluation import load_evaluator
import openai
import os

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']


embedding_function = OpenAIEmbeddings()
vector = embedding_function.embed_query("apple")

print(vector)
print(len(vector))

evaluator = load_evaluator("pairwise_embedding_distance")

#run evalutation
x = evaluator.evaluate_string_pairs(prediction="apple", prediction_b="orange")
