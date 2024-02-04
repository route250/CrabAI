import sys, os, time
import re
import unittest
from openai import OpenAI, OpenAIError
import tiktoken
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
from test_data_download import download_and_extract_zip

from crab.embeddings import create_embeddings

client:OpenAI = OpenAI()
# text-embedding-3-small
# text-embedding-3-large
# text-embedding-ada-002
model = 'text-embedding-3-small'
text = '今日の天気は晴れ'
text_list = [text]
embs,tokens = create_embeddings( client, text_list, model )
