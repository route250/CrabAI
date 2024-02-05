import sys,os,time, json
import hashlib
import numpy as np
from openai import OpenAI, OpenAIError
import tiktoken

# embeddingキャッシュの保存先
base_cachedir = os.path.join( "tmp","embcache" )

def isEmpty(value) ->bool:
    if isinstance(value,str):
        return value.strip()==''
    return False if value else True

def to_md5( text:str ) ->str:
    """文字列からmd5を計算する"""
    hash_md5 = hashlib.md5()
    hash_md5.update(text.encode())
    return hash_md5.hexdigest()

def to_path( model:str, dimensions:int, text:str, cachedir=None )->str:
    """モデル名と次元数とテキストから保存先パスへ変換する"""
    if not cachedir:
        cachedir = base_cachedir
    md5 = to_md5(text)
    if dimensions:
        model = f"{model}_{dimensions}"
    return os.path.join(cachedir,model,md5[:4],md5)

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

def create_embeddings( client:OpenAI, input, model:str, dimensions:int=None, timeout:float=None, cachedir=None ):
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
        md5path = md5path_list[i] = to_path(model,dimensions,text,cachedir=cachedir)
        emb = emb_list[i] = load_emb(md5path)
        if not emb:
            reqest_list.append( input_list[i] )
            index_convert.append(i)

    tokens:int = 0
    if reqest_list:

        if dimensions:
            res = client.embeddings.create( input=reqest_list, model=model, dimensions=dimensions, timeout=timeout )
        else:
            res = client.embeddings.create( input=reqest_list, model=model, timeout=timeout )
        tokens:int = res.usage.total_tokens
        for ee in res.data:
            idx = index_convert[ee.index]
            emb_list[idx] = ee.embedding
            save_emb( md5path_list[idx], ee.embedding )

    if not isinstance(input,list):
        emb_list=emb_list[0]

    return emb_list,tokens

class EmbeddingFunction:

    def __init__( self, client:OpenAI, model:str, dimensions:int=None, timeout:float=None ):
        self.client:OpenAI = client
        self.model:str = model
        self.dimensions:int = dimensions
        self.timeout:float = timeout

    def __str__(self) ->str:
        if self.dimensions:
            return f"{self.model}_{self.dimensions}"
        else:
            return self.model

    def simple_name(self) ->str:
        return str(self).replace('text-embedding-','')

    def __call__( self, input, timeout:float=None ):
        tm:float = timeout if timeout else self.timeout
        return create_embeddings( self.client, input=input, model=self.model, dimensions=self.dimensions, timeout=tm )

def cosine_similarity(A, B):
    """
    Calculate the cosine similarity between two vectors A and B using NumPy.
    
    Parameters:
    A (numpy array): The first vector.
    B (numpy array): The second vector.
    
    Returns:
    float: The cosine similarity between vectors A and B.
    """
    # Convert lists to numpy arrays if they are not already
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Calculate the cosine similarity using NumPy operations
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_similarity = dot_product / (norm_A * norm_B)
    return cosine_similarity

def inner_product(A, B):
    """
    Calculate the inner product between two vectors A and B using NumPy.
    
    Parameters:
    A (numpy array): The first vector.
    B (numpy array): The second vector.
    
    Returns:
    float: The inner product between vectors A and B.
    """
    # Convert lists to numpy arrays if they are not already
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Calculate the inner product using NumPy operations
    product = np.dot(A, B)
    return product

def squared_l2_distance(A, B):
    """
    Calculate the squared L2 distance between two vectors A and B using NumPy.
    
    Parameters:
    A (numpy array): The first vector.
    B (numpy array): The second vector.
    
    Returns:
    float: The squared L2 distance between vectors A and B.
    """
    # Convert lists to numpy arrays if they are not already
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Calculate the squared L2 distance using NumPy operations
    distance = np.sum((A - B) ** 2)
    return distance

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
