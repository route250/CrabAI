import sys,os,time, json
import hashlib
import numpy as np
from openai import OpenAI, OpenAIError
import tiktoken

# embeddingキャッシュの保存先
basedir = "tmp/embcache"

def isEmpty(value) ->bool:
    if isinstance(value,str):
        return value.strip()==''
    return False if value else True

def to_md5( text:str ) ->str:
    """文字列からmd5を計算する"""
    hash_md5 = hashlib.md5()
    hash_md5.update(text.encode())
    return hash_md5.hexdigest()

def to_path( model:str, dimensions:int, text:str )->str:
    """モデル名と次元数とテキストから保存先パスへ変換する"""
    md5 = to_md5(text)
    if dimensions:
        model = f"{model}_{dimensions}"
    return os.path.join(basedir,model,md5[:4],md5)

def load_emb(filepath) -> list[float]:
    """ファイルが存在していればembeddingをロードする"""
    try:
        with open(filepath, 'r') as file:
            embedding = json.load(file)
            return embedding
    except FileNotFoundError:
        return None

def save_emb(filepath, emb: list[float]):
    """embeddingを指定されたパスへ保存する"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(emb, file)
    except:
        pass

def create_embeddings( client:OpenAI, input, model:str, dimensions:int=None, timeout:float=None ):
    """キャッシング付きembedding関数"""
    # 入力テキストの型判定
    if not isinstance(input,list):
        input_list=[input]
    else:
        input_list = input

    ll:int = len(input_list)
    md5path_list:list[str] = [None] * ll
    emb_list:list[list[float]] = [None] * ll
    reqest_list=[]
    index_convert=[]
    for i in range(0,len(input_list)):
        text = input_list[i]
        if not isinstance(text,str):
            continue
        md5path = md5path_list[i] = to_path(model,dimensions,text)
        emb = emb_list[i] = load_emb(md5path)
        if not emb:
            reqest_list.append( input_list[i] )
            index_convert.append(i)

    tokens:int = 0
    if reqest_list:

        if dimensions:
            res = client.embeddings.create( input=input_list, model=model, dimensions=dimensions, timeout=timeout )
        else:
            res = client.embeddings.create( input=input_list, model=model, timeout=timeout )
        tokens:int = res.usage.total_tokens
        for ee in res.data:
            idx = index_convert[ee.index]
            emb_list[idx] = ee.embedding
            save_emb( md5path_list[idx], ee.embedding )

    if not isinstance(input,list):
        emb_list=emb_list[0]

    return emb_list,tokens

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def split_text(text: str, tokens: int = 1024) -> list[str]:
    result = []
    tkenc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tk = tkenc.encode(text)
    ll = len(tk)

    s = 0
    while s < ll:
        e = min(s+tokens,ll)
        for e in range(e,e-4,-1):
            segment = tkenc.decode(tk[s:e])
            if text.startswith(segment):
                break
        result.append(segment)
        text=text[len(segment):]
        s = e

    return result
