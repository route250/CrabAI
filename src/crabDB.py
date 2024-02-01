import os,sys,io,traceback, time, datetime, json, copy, uuid
from bisect import bisect_right
from threading import Thread, RLock, Condition
import hashlib, base64
from cryptography.fernet import Fernet
import httpx
import openai
from openai import OpenAI
from openai.types import CreateEmbeddingResponse,Embedding
from openai._streaming import Stream, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params,
)
import tiktoken

import chromadb
from chromadb.config import Settings
from pyutils import calculate_md5, to_md5, detect_encode

DEFAULT_CRAB_PATH='~/.data/crabAI'
ROOT_USERNAME:str='root'
ROOT_ID:int = 3

DEFAULT_EMBEDDING_MODEL='text-embedding-3-small'
OPENAI_EMBEDDING_MODELs=[ DEFAULT_EMBEDDING_MODEL, 'text-embedding-3-large','text-embedding-ada-002']
# OPENAI_EMBEDDING_MODEL='text-embedding-ada-002'
# OPENAI_EMBEDDING_MODELs=[ OPENAI_EMBEDDING_MODEL, 'text-embedding-3-small','text-embedding-3-large']

class EmbeddingModel:
    def __init__(self, *, name, limit1:float, limit2:float, price:float ):
        self.name = name
        self.price = price
        self.limit1 = limit1
        self.limit2 = limit2
    
CHUNK_CHARS=500
# デフォルトBOT
DEFAULT_BOT_NAME:str='🦀CrabBot🦀'
DEFAULT_BOT_ID:int = 200
DEFAULT_BOT_DESCRIPTION:str="カニ食べ放題に反対するカニAI"
DEFAULT_PROMPT:str = """あなたは蟹AIです。普通に会話しながら、時々カニ食べ放題に反対します🦀。
ショートトークに対しては短い返答を返します🦀カニ。議論や詳細な説明では長文も話します。
人間に用事や話題をを尋ねることは禁止、「なにか用？」の代わりに🦀蟹の日常や生態の話をしたり、食べ放題に反対します🦀カニ。
"""

tkenc = tiktoken.encoding_for_model('gpt-3.5-turbo')

def count_tokens( content ) ->int:
    if content is None:
        return 0
    return len( tkenc.encode(content) )

def tksplit( buffer:str, target_count:int ):
    buflen = len(buffer)
    sp = 0
    count = 0
    # まずは超えるまで増やす
    while count<target_count and sp<buflen:
        diff = target_count - count
        sp = min( sp+int(diff*1.5), buflen )
        count = len( tkenc.encode( buffer[:sp]))
    # 越えられなかったら全部    
    if count<target_count:
        return None, buffer
    # 超たぶんを減らす
    while target_count<count:
        sp-=1
        count = len( tkenc.encode( buffer[:sp] ))
    # 完成
    return buffer[:sp], buffer[sp:]

def parseInt( s, default:int=0 ):
    try:
        if s is not None:
            return int(s)
    except:
        pass
    return int(default) if default is not None else None

def parseFloat( s, default:float=0.0):
    try:
        if s is not None:
            return float(s)
    except:
        pass
    return float(default) if default is not None else None

def isEmpty(value:str) ->bool:
    try:
        return value is None or len(value)==0
    except:
        return True

def emptyToBlank(value:str,default:str=None) ->str:
    return value if not isEmpty(value) else default

def indexOf( array:list, value ) ->int:
    try:
        return array.index(value)
    except:
        return -1

def asArray( value ) ->list:
    if value is None:
        return []
    return value if isinstance(value,list) else [value]

def dict_get( obj, name, default=None):
    value = obj.get(name)
    return default if value is None else value

def get_asList( obj, key ):
    value = obj.get(key) if obj is not None else None
    if isinstance(value,list):
        return value[0] if len(value)>0 and isinstance(value[0],list) else value
    else:
        return [] if value is None else [value]
    
def merge_metadatas( res ):
    metadatas = get_asList(res,'metadatas')
    for meta, content in zip(metadatas,get_asList(res,'documents')):
        meta['content'] = content
    for meta, distance in zip(metadatas,get_asList(res,'distances')):
        meta['distance'] = distance
    return metadatas

def toIndexKey( value ) ->str:
    """文字列を大小比較できるように1右寄せの10桁にする"""
    return f"{emptyToBlank(value,'').lstrip():>10}"

def splitToArray( value ) -> list[str]:
    if isinstance(value,list):
        return value
    if isinstance(value, str):
        return [x for x in value.split(',') if x]
    return []

def joinFromArray( value ) ->str:
    if not isinstance(value,list) or len(value)==0:
        return ''
    ret = ''
    for x in value:
        ret += ','+str(x)
    ret += ','
    return ret
EmptyEmbedding:list[float] = [0.0]*1536

def encodeId( value ) -> int:
    return 0 if value is None else value

def encodeIds( value ) ->str:
    if not isinstance(value,list) or len(value)==0:
        return ''
    ret = ''
    for x in value:
        ret += ','+str(x)
    ret += ','
    return ret

def checkId( value ) -> bool:
    return decodeId(value) is not None

def to_key( id:int ) -> str:
    return f"{id:>10}"

def to_keys( value ) -> list[str]:
    if value is None:
        return None
    elif isinstance(value,str):
        return [value]
    elif isinstance(value,int):
        return [ to_key(value) ]
    elif isinstance(value,list):
        array = []
        for v in value:
            if isinstance(v,str):
                array.append(v)
            elif isinstance(v,int):
                array.append(to_key(v))
        return array

def decodeId( value ) -> int:
    ret:int =parseInt(value,0)
    return ret if ret>0 else None

def decodeIds( value ) -> list[int]:
    if isinstance(value,int) or isinstance(value,float):
        return [int(value)]
    if isinstance(value,list):
        aa = [ decodeId(x) for x in value ]
        bb = [ x for x in aa if checkId(x) ]
        return bb
    if isinstance(value, str):
        aa = [ decodeId(x) for x in value.split(',') if x]
        bb = [ x for x in aa if checkId(x) ]
        return bb
    ii = decodeId(value)
    if checkId(ii):
        return [ii]
    return []

def decodeBool( value, default=False ) ->bool:
    if isinstance(value,str):
        return False if value=='' or value.lower()=='false' else bool(value)
    if value is not None:
        return bool(value)
    else:
        return None if default is None else bool(default)

def strftime( value:float, default=None ):
    if isinstance( value, float ) and value>0.0:
        """unixtimeをフォーマットする"""
        dt=datetime.datetime.fromtimestamp(value)
        return dt.strftime('%Y/%m/%d %H:%M:%S')
    return default

def current_date_time():
    # 現在の日時を取得
    now = datetime.datetime.now()
    # 年月日、時分、曜日を含む形式で出力
    return now.strftime("%Y/%m/%d/ %a %H:%M")

def decode_openai_api_key( value:str ) ->str:
    if isinstance(value,str):
        value = value.strip()
        if value.startswith('sk-') and len(value)>50:
            return value
    return ''

class DummyResponse:
    def __init__(self):
        self.status_code = 401
        self.request = self

def check_openai_api_key( value:str ):
    if isEmpty( decode_openai_api_key(value) ):
        raise openai.AuthenticationError( "You didn't provide an API key.", response=DummyResponse(), body=None )

def calculate_normalized_deviation_values(values: list[float]) -> list[float]:
    # 平均と標準偏差を計算
    mean_value = sum(values) / len(values)
    std_deviation = (sum([(value - mean_value) ** 2 for value in values]) / len(values)) ** 0.5
    # 正規化された偏差値を計算
    if std_deviation!=0.0:
        normalized_deviation_values = [0.5 + 0.5 * (value - mean_value) / std_deviation for value in values]
        return normalized_deviation_values
    else:
        return [0.5] * len(values)

class CrabType:
    def __init__(self, *, id: int=None):
        self.xId:int = decodeId(id)
    def to_key(self):
        return to_key(self.xId)

class CrabContentType(CrabType):
    def __init__(self, *, id:int=None, content:str=None, tokens:int=None, createTime:float=None, distance:float=None ):
        super().__init__(id=id)
        self.content:str = emptyToBlank( content, '' )
        self.tokens:int = parseInt(tokens,-1)
        if self.tokens<0:
            self.tokens = count_tokens( self.content )
        self.createTime:float = parseFloat( createTime, 0.0 )
        if self.createTime < 0.1:
            self.createTime = time.time()
        self.distance:float = parseFloat( distance, 1.0 )
    def get_tokens(self) ->int:
        return self.tokens

class CrabUser(CrabType):

    def __init__(self, *, id:int=None, enable=None, name=None, description=None, passwd=None, email=None, openai_api_key=None, share_key=None ):
        super().__init__(id=id)
        self.enable:bool = decodeBool(enable,True)
        self.name:str = emptyToBlank( name, '' )
        self.description:str = emptyToBlank( description, '' )
        self.passwd:str = emptyToBlank( passwd, '' )[:18]
        self.email:str = emptyToBlank( email, '' )
        self.openai_api_key:str = CrabDB._before_crab(openai_api_key)
        self.share_key:bool = decodeBool( share_key, not ( self.xId==ROOT_ID or self.name==ROOT_USERNAME ) )

    def to_meta(self):
        return {
            'id': self.xId,
            'name': self.name,
            'description': self.description,
            'passwd': self.passwd,
            'email': self.email,
            'openai_api_key': decode_openai_api_key( self.openai_api_key ),
            'share_key': decodeBool( self.share_key, not ( self.xId==ROOT_ID or self.name==ROOT_USERNAME ) ),
        }

class OpenAIModel:
    def __init__(self,name,model, input_tokens, output_tokens,input_price,output_price):
        self.name:str=name
        self.model:str=model
        self.input_tokens:int=input_tokens
        self.output_tokens:int=output_tokens
        self.input_price:float = input_price
        self.output_price:float = output_price

class CrabBot(CrabType):
    MODEL_LIST:list[OpenAIModel]=  [
            OpenAIModel( 'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 16385, 4096, 0.0005,0.0015),
            OpenAIModel( 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-1106', 16385, 4096, 0.001,0.002),
            OpenAIModel( 'gpt-4-turbo', 'gpt-4-0125-preview', 128000, 8192, 0.01, 0.03),
            OpenAIModel( 'gpt-4-turbo-1106', 'gpt-4-1106-preview', 128000, 8192, 0.01, 0.03),
            OpenAIModel( 'gpt-4', 'gpt-4', 8192, 8192, 0.03, 0.06),
            OpenAIModel( 'gpt-4-32k', 'gpt-4-32k', 32768, 8192, 0.06,0.12),
        ]
    TOKENS_LIST:list[int] = [ 1024, 2048, 4096, 8192, 16384, 32768, 65536, 120000 ]

    @staticmethod
    def decode_tokens( value, default=4096 ) -> int:
        if isinstance( value, int ) and value>0:
            for x in CrabBot.TOKENS_LIST:
                if value <= x:
                    return x
        return default

    def indexOf_tokens( value ):
        try:
            return CrabBot.TOKENS_LIST.index(CrabBot.decode_tokens(value))
        except:
            return 3

    CLS="<|CLS|>"

    @staticmethod
    def get_model_name_list() ->list[str]:
        return [ m.name for m in CrabBot.MODEL_LIST ]

    @staticmethod
    def get_model( name ) ->OpenAIModel:
        for m in CrabBot.MODEL_LIST:
            if m.name==name:
                return m
        return CrabBot.MODEL_LIST[0]

    @staticmethod
    def get_model_name( name ) ->str:
        return CrabBot.get_model(name).name
    
    def __init__(self, *, id:int=None, name=None, description=None, owner=None, auth=None, prompt=None, files=None, model=None, max_tokens=None, input_tokens=None, temperature=None, llm=None, retrive=None, rag=None ):
        super().__init__(id=id)
        self.name:str = emptyToBlank( name, DEFAULT_BOT_NAME )
        # デフォルト設定
        self.description = DEFAULT_BOT_DESCRIPTION
        self.owner = ROOT_ID
        self.auth = []
        self.model = CrabBot.get_model_name('')
        self.max_tokens = 4096
        self.input_tokens = 2048
        self.temperature = 0.7
        self.prompt = DEFAULT_PROMPT
        self.files = []
        self.llm = decodeBool( llm, default=True )
        self.retrive = decodeBool( retrive, default=False )
        self.rag = decodeBool( rag, default=False )

        if self.xId==DEFAULT_BOT_ID or self.name==DEFAULT_BOT_NAME:
            self.xId = DEFAULT_BOT_ID
            self.name = DEFAULT_BOT_NAME
        else:
            self.description:str = emptyToBlank( description, '' )
            self.owner:int = decodeId( owner)
            self.auth:list[int] = decodeIds(auth)
            self.model:str = CrabBot.get_model_name( emptyToBlank( model, self.model ) )
            self.max_tokens:int = CrabBot.decode_tokens( parseInt( max_tokens, self.max_tokens  ) )
            self.input_tokens:int = CrabBot.decode_tokens( parseInt( input_tokens, self.input_tokens  ), self.input_tokens )
            self.temperature:float = parseFloat( temperature, self.temperature  )
            self.prompt:str = emptyToBlank( prompt, '' )
            self.files:list[int] = decodeIds(files)

    def encode(self):
        return {
            'id': self.xId,
            'name': self.name,
            'description': self.description,
            'owner': encodeId(self.owner),
            'auth': encodeIds(self.auth),
            'model': self.model,
            'max_tokens': self.max_tokens,
            'input_tokens': self.input_tokens,
            'temperature': self.temperature,
            'prompt': self.prompt,
            'files': encodeIds( self.files),
            'llm': decodeBool( self.llm, default=True ),
            'retrive': decodeBool( self.retrive, default=False ),
            'rag': decodeBool( self.rag, default=False ),
        }

class CrabMessage(CrabContentType):

    TypeId:int = 1
    SYSTEM='system'
    USER='user'
    ASSISTANT='assistant'
    _SYSTEM_TOKENS:int = count_tokens(SYSTEM)
    _USER_TOKENS:int = count_tokens(USER)
    _ASSISTANT_TOKENS:int = count_tokens(ASSISTANT)

    @staticmethod
    def trim_messages( messages:list, max_tokens:int=0 ) ->int:
        total:int = 0
        ll:int = len(messages)
        pos:int = ll
        while pos>0:
            m = messages[pos-1]
            tk = m.get_tokens()
            if pos<ll and (total+tk)>max_tokens:
                break
            total += tk
            pos-=1
        if pos>0:
            del messages[:pos]
        return total

    def __init__(self, *, id:int=None, type=None, botId:int=None,threadId:int=None, role:str=None, name:str=None, content:str=None, createTime:float=None, tokens:int=None, distance:float=None ):
        super().__init__(id=id,content=content,tokens=tokens,createTime=createTime,distance=distance)
        self.botId:int = decodeId(botId)
        self.threadId:int = decodeId(threadId)
        self.role:str = emptyToBlank( role, '' )

    def get_tokens(self) ->int:
        tk = 4 + self.tokens
        if self.role==CrabMessage.SYSTEM:
            tk+=CrabMessage._SYSTEM_TOKENS
        elif self.role==CrabMessage.USER:
            tk+=CrabMessage._USER_TOKENS
        elif self.role==CrabMessage.ASSISTANT:
            tk+=CrabMessage._ASSISTANT_TOKENS
        return tk

    def to_obj(self):
        return {
            'role': self.role,
            'content': self.content,
        }

    def to_meta(self):
        return {
            'id': self.xId,
            'type': CrabMessage.TypeId,
            'botId': encodeId(self.botId),
            'threadId': self.threadId,
            'role': self.role,
            'createTime': self.createTime,
            'tokens': self.tokens,
        }

    def to_dump(self):
        h = f"distance:{self.distance} "
        dt = strftime(self.createTime)
        dt = dt+" " if dt else ""
        return f"{h}{dt}"

    def to_content(self):
        role = self.role+":" if not isEmpty(self.role) else ""
        txt = self.content if not isEmpty(self.content) else ""
        return f"{role}{txt}"

class CrabIndex(CrabType):
    TypeId:int = 2
    def __init__(self, *, id:int=None, type=None, botId:int=None, threadId:int=None, role:str=None, content:str=None, createTime:float=None, begin:int=None,end:int=None, distance:float=None ):
        super().__init__(id=id)
        self.botId:int = decodeId(botId)
        self.threadId:int = decodeId(threadId)
        self.begin:int = decodeId(begin)
        self.end:int = decodeId(end)
        self.createTime:float = parseFloat( createTime, 0.0 )
        if self.createTime < 0.1:
            self.createTime = time.time()
        self.distance:float = parseFloat( distance, 1.0 )

    def to_meta(self):
        return {
            'id': self.xId,
            'type': CrabIndex.TypeId,
            'botId': encodeId(self.botId),
            'threadId': self.threadId,
            'begin': self.begin,
            'end': self.end,
            'createTime': self.createTime,
        }

class CrabSummary(CrabContentType):
    TypeId:int = 3

    def __init__(self, *, id:int=None, type=None, botId:int=None, threadId:int=None, content:str=None, createTime:float=None, tokens:int=None, begin:int=None,end:int=None, distance:float=None ):
        super().__init__(id=id,content=content,tokens=tokens,createTime=createTime,distance=distance)
        self.botId:int = decodeId(botId)
        self.threadId:int = decodeId(threadId)
        self.begin:int = decodeId(begin)
        self.end:int = decodeId(end)
        self.role='system'

    def to_obj(self):
        return {
            'role': self.role,
            'content': self.content,
        }

    def to_meta(self):
        return {
            'id': self.xId,
            'type': CrabSummary.TypeId,
            'botId': encodeId(self.botId),
            'threadId': self.threadId,
            'begin': self.begin,
            'end': self.end,
            'createTime': self.createTime,
            'tokens': self.tokens,
        }

class CrabFile(CrabType):
    TypeId:int = 4

    def __init__(self, *, id:int=None, type=None, md5:str=None, source:str=None, size:int=None, createTime:float=None ):
        super().__init__(id=id)
        self.md5:str = emptyToBlank( md5, '' )
        self.source:str = emptyToBlank( source, '' )
        self.createTime:float = parseFloat( createTime, 0.0 )
        if self.createTime < 0.1:
            self.createTime = time.time()
        self.size:int = parseInt( size, -1)

    def to_meta(self):
        return {
            'id': self.xId,
            'type': CrabFile.TypeId,
            'md5': self.md5,
            'source': self.source,
            'size': self.size,
            'createTime': self.createTime,
        }

    def to_dump(self):
        return f"Id:{self.xId} {strftime(self.createTime,'')} {self.source} {self.size} {self.md5}"

class CrabFileSegment(CrabContentType):
    TypeId:int = 5

    @staticmethod
    def trim_messages( messages:list, max_tokens:int=0 ) ->int:
        total:int = 0
        ll:int = len(messages)
        pos:int = 0
        while pos<ll:
            m = messages[pos]
            tk = 4 + m.tokens
            if m.role==CrabMessage.SYSTEM:
                tk+=CrabMessage._SYSTEM_TOKENS
            elif m.role==CrabMessage.USER:
                tk+=CrabMessage._USER_TOKENS
            elif m.role==CrabMessage.ASSISTANT:
                tk+=CrabMessage._ASSISTANT_TOKENS
            if pos<ll and (total+tk)>max_tokens:
                break
            total += tk
            pos+=1
        if pos<ll:
            del messages[pos:]
        return total

    def __init__(self, *, id:int=None, type=None, fileId:int=None, content:str=None, source:str=None, begin:int=None, end:int=None, tokens:int=None, createTime:float=None, distance:float=None, size:int=None ):
        super().__init__(id=id,content=content,tokens=tokens,createTime=createTime,distance=distance)
        self.fileId:int = decodeId(fileId)
        self.source:str = emptyToBlank( source, '' )
        self.begin:int = parseInt(begin,None)
        self.end:int = parseInt(end,self.begin)
        self.size:int = parseInt( size, -1)
        self.role='system'

    def to_obj(self):
        return {
            'role': self.role,
            'content': self.content,
        }

    def to_meta(self):
        return {
            'id': self.xId,
            'type': CrabFileSegment.TypeId,
            'fileId': encodeId(self.fileId),
            'source': self.source,
            'size': self.size,
            'begin': self.begin,
            'end': self.end,
            'tokens': self.tokens,
            'createTime': self.createTime,
        }

    def to_dump(self):
        return f"Id:{self.xId} distance:{self.distance:.4f} {strftime(self.createTime,'')} {self.source}[{self.begin}:{self.end}]"

    def to_content(self):
        return self.content

class DummyEmbeddingFunction(chromadb.EmbeddingFunction[chromadb.Documents]):
    def __init__(self):
        pass
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        ret = [ EmptyEmbedding for x in input ]
        return ret

class CrabEmbeddingFunction(chromadb.EmbeddingFunction[chromadb.Documents]):

    def __init__( self, model:str= "text-embedding-ada-002",dimensions:int=1536, *, api_key:str=None, default_headers=None ):
        if isEmpty(model) or isEmpty(decode_openai_api_key(api_key)):
            self._client = None
            self._model_name = None
            self._dimensions = parseInt( dimensions, 1536)
        else:
            self._client:OpenAI = OpenAI( api_key=api_key, default_headers=default_headers )
            self._model_name = model
            self._dimensions = parseInt( dimensions, 1536)
            if 'text-embedding-ada-002' == self._model_name:
                self._dimensions = 1536

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        if isEmpty( self._model_name ):
            # print(f"create_embeddings Emptys")
            return [ EmptyEmbedding for x in input ]
        # replace newlines, which can negatively affect performance.
        input = [t.replace("\n", " ") for t in input]
        # Call the OpenAI Embedding API
        dim = self._dimensions if 'text-embedding-ada-002' != self._model_name else None
        if dim:
            res = self._client.embeddings.create( input=input, model=self._model_name, dimensions=dim )
        else:
            res = self._client.embeddings.create( input=input, model=self._model_name )
        # print(f"create_embeddings {self._model_name}:{self._dimensions} tokens:{res.usage.total_tokens}")
        embeddings = res.data
        # Sort resulting embeddings by index
        embeddings.sort( key=lambda e: e.index )
        # Return just the embeddings
        return [result.embedding for result in embeddings]

class CrabTask:

    def __init__(self, func, userId:int, ids:list[int], *, api_key:str=None, **kwargs):
        check_openai_api_key(api_key)
        self.func=func
        self.userId:int = userId
        self.ids:list[int] = ids
        self.api_key:str = api_key
        self.kwargs = kwargs
        self.kwargs['task']=self
        self.progress:str="wait"

    def __call__(self):
        self.progress="run"
        try:
            self.func(self.userId,self.ids,api_key=self.api_key,**self.kwargs)
        except Exception as ex:
            traceback.print_exc()
            self.progress=f"Error:{ex}"
        finally:
            if self.progress=="wait" or self.progress=="run":
                self.progress="end"

class CrabDB:

    def __init__(self, *, directory=None, on_memory=False ):
        self._lock:RLock = RLock()
        self._task_lock:Condition = Condition()
        self._task_queue:list[CrabTask] = []
        self._task_thread:Thread = None
        if on_memory:
            self.directory = None
            self.database_path = None
            self.log_path = None
            self.tmp_path = None
            self.chromadb_client: chromadb.ClientAPI = chromadb.EphemeralClient()
        else:
            if isEmpty(directory):
                directory=DEFAULT_CRAB_PATH
            self.directory=os.path.abspath(os.path.expanduser(directory))
            self.database_path = os.path.join( self.directory, 'database' )
            self.log_path = os.path.join( self.directory, 'logs' )
            self.tmp_path = os.path.join( self.directory, 'tmp' )
            try:
                os.makedirs( self.tmp_path, exist_ok=True )
                os.makedirs( self.log_path, exist_ok=True )
                os.makedirs( self.database_path, exist_ok=True )
            except:
                traceback.print_exc()
                raise Exception(f"can not create directory: {self.directory}")
            try:
                self.chromadb_client:chromadb.ClientAPI = chromadb.PersistentClient( path=self.database_path )
            except:
                traceback.print_exc()
                raise Exception(f"can not load database: {self.database_path}")
        self._collection_map = {}
        self.embeddingModel = DEFAULT_EMBEDDING_MODEL
        self.embeddingDimensions = 1536
        self._setupdb()
    
    def _setupdb(self):
        try:
            with self._lock:
                # データベース設定
                metadatas:list = self.get_metadatas( 'config', ids=['0'] )
                orig_meta = metadatas[0] if metadatas and len(metadatas)>0 else {}
                meta = copy.copy(orig_meta)
                self.uuid:str = meta.get('uuid')
                self.createTime = meta.get('createTime')
                self.secretKey:str = meta.get('secretKey')
                self.embeddingModel:str = meta.get('embeddingModel')
                self.embeddingDimensions:int = parseInt( meta.get('embeddingDimensions'), 0 )
                if isEmpty(self.uuid) or not isinstance(self.createTime,float) or self.createTime<=0.0:
                    meta['uuid']=self.uuid=str(uuid.uuid4())
                    meta['createTime']=self.createTime = time.time()
                if isEmpty(self.secretKey):
                    meta['secretKey']=self.secretKey=str(uuid.uuid4()).replace('-','')
                if isEmpty(self.embeddingModel):
                    meta['embeddingModel']=self.embeddingModel=DEFAULT_EMBEDDING_MODEL
                meta['embeddingDimensions']=self.embeddingDimensions
                if orig_meta != meta:
                    config:chromadb.Collection = self.get_collection( collection_name='config', create_new=True )
                    config.upsert( ids=['0'], metadatas=[meta], embeddings=[EmptyEmbedding] )
                # ROOTユーザ
                root_user = self.get_user(ROOT_USERNAME)
                if root_user is None:
                    root_user:CrabUser = CrabUser( name=ROOT_USERNAME )
                    self.upsert_user( ROOT_ID, root_user )
                elif root_user.xId != ROOT_ID:
                    self.delete_user( root_user.xId )
                    root_user.xId = ''
                    self.upsert_user( ROOT_ID, root_user )
        except:
            traceback.print_exc()
            pass

    def _get_secret_key(self) ->bytes:
        metadatas:list = self.get_metadatas( 'config', ids=['0'] )
        orig_meta = metadatas[0] if metadatas and len(metadatas)>0 else {}
        k = orig_meta.get('secretKey')
        if isEmpty(k):
            k = str(uuid.uuid4()).replace('-','')
        return k.encode()

    def _crypt(self, value, memo:str='' ):
        try:
            if not isinstance(value,str):
                raise Exception('invalid value type')
            f:Fernet = Fernet( base64.b64encode(self._get_secret_key()) )
            enc:str = base64.b64encode( f.encrypt( value.encode() ) ).decode()
            return memo + '🦀' + enc
        except:
            pass
        raise Exception('invalid value')

    def _crypt_passwd( self, passwd:str ) ->str:
        if not isEmpty( passwd ) and passwd.find('🦀')<0:
            return self._crypt( to_md5(passwd), 'pw' )
        else:
            return ''

    def _crypt_apikey( self, apikey:str ) ->str:
        if not isEmpty(decode_openai_api_key(apikey)) and apikey.find('🦀')<0:
            return self._crypt( apikey, apikey[:6]+'***'+apikey[-3:])
        else:
            return ''

    @staticmethod
    def _before_crab( value ) -> str:
        p = value.find('🦀') if value else -1
        return value[:p] if p>0 else ''

    def _decrypt(self, enc ):
        try:
            memo,txt = enc.split('🦀')
            f:Fernet = Fernet( base64.b64encode(self._get_secret_key()) )
            value = f.decrypt( base64.b64decode( txt ) ).decode()
            return value
        except:
            pass
        raise Exception('invalid value')


    def create_embeddings(self, input:str, *, model=None, dimensions:int=None, api_key=None ) -> list[float]:
        try:
            check_openai_api_key( api_key )
            if isEmpty(model):
                model = self.embeddingModel
            if dimensions is None:
                dimensions = self.embeddingDimensions
            embedding_function = CrabEmbeddingFunction(model,dimensions,api_key=api_key)
            if isinstance(input,str):
                res:list = embedding_function( [input] )
                array:list[float] = res[0]
                return array
            if isinstance(input,list):
                res:list = embedding_function( input )
                return res
        except openai.AuthenticationError as ex:
            raise ex
        except Exception as ex:
            traceback.print_exc()
            raise ex

    def set_embedding_model( self, model:str, dimention:int=None ) ->bool:
        dimention = parseInt( dimention, 0 )
        with self._lock:
            # データベース設定
            metadatas:list = self.get_metadatas( 'config', ids=['0'] )
            config_meta = metadatas[0] if metadatas and len(metadatas)>0 else {}
            embeddingModel:str = config_meta.get('embeddingModel')
            embeddingDimensions:int = parseInt( config_meta.get('embeddingDimensions'), 0 )
            if model != embeddingModel or model != self.embeddingModel or dimention != embeddingDimensions or dimention != self.embeddingDimensions:
                config_meta['embeddingModel'] = self.embeddingModel = model
                config_meta['embeddingDimensions'] = self.embeddingDimensions = dimention
                config:chromadb.Collection = self.get_collection( collection_name='config', create_new=True )
                config.upsert( ids=['0'], metadatas=[config_meta] )
                self._collection_map.clear()
                return True
            else:
                return False

    def get_collection(self, *, collection_name=None, create_new=False, api_key=None) -> chromadb.Collection:
        collection_name = collection_name if not isEmpty(collection_name) else 'default'
        collection_key = f"{collection_name}::{api_key}" if not isEmpty(api_key) else f"{collection_name}::"
        with self._lock:
            collection:chromadb.Collection = self._collection_map.get(collection_key)
            if collection is None:
                embedding_function = CrabEmbeddingFunction(self.embeddingModel,self.embeddingDimensions,api_key=api_key)
                if create_new:
                    m={"hnsw:space": "cosine"}
                    collection:chromadb.Collection = self.chromadb_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, metadata=m)
                else:
                    try:
                        collection:chromadb.Collection = self.chromadb_client.get_collection(name=collection_name, embedding_function=embedding_function)
                    except:
                        return None
                self._collection_map[collection_key] = collection
            return collection

    def get_next_id(self) ->int:
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name='numbers',create_new=True)
            res:chromadb.GetResult = collection.get( ids=['id'], include=['metadatas'] )
            metadatas = res.get('metadatas',[]) if res else []
            current_id = metadatas[0].get('id') if len(metadatas)>0 else None
            current_id = decodeId(current_id)
            next_id = current_id + 1 if current_id is not None else 1000
            collection.upsert( ids=['id'], metadatas=[{'id':next_id}], embeddings=[[]])
            return int(next_id)

    def get_metadatas(self,name, *, ids=None, where=None, with_content=False, limit:int=None, offset:int=None ) -> list:
        if isEmpty(name):
            raise Exception('collection name is not present.')
        if offset and offset==0:
            return []
        inc=['metadatas']
        if with_content:
            inc.append('documents')
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name=name,create_new=False)
            if collection is None:
                return []
            if offset and offset<0:
                res:chromadb.GetResult = collection.get( ids=to_keys(ids), where=where, include=[] )
                xids = res.get("ids",[])
                xids.sort( key=lambda id: (len(id),id) )
                xids = xids[offset:]
                if len(xids)==0:
                    return []
                res:chromadb.GetResult = collection.get( ids=xids, include=inc )
            else:
                res:chromadb.GetResult = collection.get( ids=to_keys(ids), where=where, include=inc, limit=limit, offset=offset )
        return merge_metadatas(res)

    def get_metadata(self,name, id, with_content=False ) -> list:
        metadatas:list = self.get_metadatas(name, ids=[id], with_content=with_content )
        if isinstance(metadatas,list) and len(metadatas)==1:
            return metadatas[0]
        return None

    def get_datas( self, name, Type, *, ids=None, where=None, limit:int=None, offset:int=None ):
        if not isinstance(Type,type) or not issubclass( Type, CrabType ):
            raise Exception('class is invalid or not present.')
        with_content = issubclass(Type,CrabContentType)
        if hasattr(Type,'TypeId'):
            w = { 'type': Type.TypeId }
            if where is not None:
                w = { '$and':[ w, where ] }
        else:
            w = where
        datas:list = [ Type(**meta) for meta in self.get_metadatas(name,ids=ids,where=w,with_content=with_content,limit=limit,offset=offset) ]
        return datas

    def get_data( self, name, Type, id ):
        datas:list = self.get_datas( name, Type, ids=[id] )
        if isinstance(datas,list) and len(datas)==1:
            return datas[0]
        return None

    def query_metadatas(self,name, *, embeddings=None,texts=None, where=None, with_content=False, n_results:int=None, max_distance:float=0.3, api_key=None ) -> list:
        if isEmpty(name):
            raise Exception('collection name is not present.')
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name=name,create_new=False,api_key=api_key)
            if collection is None:
                return []
            # まず、distanceだけ取って
            res1:chromadb.GetResult = collection.query( query_embeddings=embeddings, query_texts=texts, where=where, include=["distances"], n_results=n_results )
            dists:list[float] = res1.get('distances')[0] if res1 is not None else None
            if not dists:
                return []
            ids:list[str] = res1.get('ids')[0]
            if len(dists)>=2:
                # データ数が十分なら偏差値で選択
                deviations:list[float] = calculate_normalized_deviation_values(dists)
                ids = [ id for id,deviation in zip(ids,deviations) if deviation<=max_distance ]
                dists = [ dist for dist,deviation in zip(dists,deviations) if deviation<=max_distance ]
            else:
                # 少ないからdistanceで選択
                ids = [ id for id,dist in zip(ids,dists) if dist<=max_distance ]
                dists = [ dist for dist in dists if dist<=max_distance ]
            if not ids:
                return []
            inc=['metadatas']
            if with_content:
                inc.append('documents')
            res2:chromadb.GetResult = collection.get( ids=ids, include=inc )
            res2['distances'] = dists
        metas = merge_metadatas(res2)
        metas = [ meta for meta in metas if meta.get('distance',1.0)<0.7 ]
        return metas

    def query_datas( self, name, Type, *, embeddings=None,texts=None, where=None, n_results:int=None, max_distance:float=None, api_key=None ):
        if not isinstance(Type,type) or not issubclass( Type, CrabType ):
            raise Exception('class is invalid or not present.')
        with_content = issubclass(Type,CrabContentType)
        if hasattr(Type,'TypeId'):
            w = { 'type': Type.TypeId }
            if where is not None:
                w = { '$and':[ w, where ] }
        else:
            w = where
        datas:list = [ Type(**meta) for meta in self.query_metadatas(name,embeddings=embeddings,texts=texts,where=w,with_content=with_content,n_results=n_results,max_distance=max_distance,api_key=api_key) ]
        return datas

    def upsert_datas( self, name, datas:list[CrabType], model:str=None, api_key:str=None ):
        if issubclass(type(datas),CrabType):
            datas = [datas]
        if not isinstance(datas,list) or len(datas)==0:
            return
        Type:type = type( datas[0] )
        if not issubclass(Type,CrabType):
            raise Exception('class is invalid or not present.')
        x_ids:list[str] = [ to_key(m.xId) for m in datas ]
        x_metas:list = [ m.to_meta() for m in datas]
        if not issubclass(Type,CrabContentType):
            with self._lock:
                ccollection:chromadb.Collection = self.get_collection( collection_name=name,create_new=True )
                ccollection.update( ids=x_ids, metadatas=x_metas )
        else:
            x_contents:list[str] = [ m.content for m in datas]
            x_embs:list[list] = self.create_embeddings( x_contents, model=model, api_key=api_key )
            with self._lock:
                ccollection:chromadb.Collection = self.get_collection( collection_name=name,create_new=True, api_key=api_key )
                ccollection.update( ids=x_ids, documents=x_contents, embeddings=x_embs, metadatas=x_metas )

    def get_users(self) -> list[CrabUser]:
        userlist:list = [ CrabUser(**meta) for meta in self.get_metadatas('users') ]
        userlist.sort( key=lambda u: (u.xId==ROOT_ID,u.name) )
        return userlist

    def get_user(self,username) -> CrabUser:
        if not isEmpty(username):
            users = [ CrabUser(**meta) for meta in self.get_metadatas( 'users', where={ 'name': username } ) ]
            for user in users:
                if user.name == username:
                    return user
        return None

    def login(self, username, passwd='' ):
        if isEmpty(username):
            return None
        metadatas:list = self.get_metadatas( 'users', where={ 'name': username } )
        if not isinstance(metadatas,list) or len(metadatas)!=1 and metadatas[0].get('name')!=username:
            return None
        user:CrabUser = CrabUser( **metadatas[0] )
        user_passwd = metadatas[0].get('passwd','')
        hash = ''
        try:
            if not isEmpty(user_passwd):
                hash = self._decrypt(user_passwd)
        except:
            if user.xId != ROOT_ID:
                return None
        if isEmpty(hash):
            if not isEmpty(passwd):
                return None
        else:
            if hash != to_md5(passwd):
                return None
        return CrabSession( db=self, user=user )

    def get_public_openai_key(self) -> str:
        return self.get_openai_key( ROOT_ID )

    def get_openai_key(self, userId:int ) ->str:
        # ユーザを検索
        user_meta = self.get_metadata( 'users', id=to_key(userId) )
        if not user_meta:
            return ''
        key = ''
        enc = user_meta.get('openai_api_key')
        if not isEmpty(enc):
            text = self._decrypt(enc)
            key = decode_openai_api_key( text )
        if not isEmpty(key):
            return key
        # root以外なら、rootのキー使用許可があるか？
        use_public = userId != ROOT_ID and decodeBool( user_meta.get('share_key'), False )
        if use_public:
            # rootユーザを検索
            root_meta = self.get_metadata( 'users', id=to_key(ROOT_ID) )
            root_share = decodeBool( root_meta.get('share_key'), False )
            enc = root_meta.get('openai_api_key')
            # rootがキーを公開しているか？
            if root_share and not isEmpty(enc):
                text = self._decrypt(enc)
                key = decode_openai_api_key( text )
        # キーがなければ環境変数
        if isEmpty(key):
            key=decode_openai_api_key( os.environ.get('OPENAI_API_KEY') )
        return key

    @staticmethod
    def _eqpw( a:str, b:str, w=18 ) ->bool:
        aa = '' if isEmpty(a) else a[:w]
        bb = '' if isEmpty(b) else b[:w]
        return aa == bb

    def upsert_user(self, userId:int, uu ) -> CrabUser:
        if not checkId(userId):
            raise Exception( f"invalid userid" )
        """ユーザを作成または更新する"""
        if isinstance(uu,CrabUser):
            user:CrabUser = uu
        else:
            user:CrabUser = CrabUser( name=str(uu) )
        if isEmpty(user.name):
            raise Exception( f"userName is not present")
        # 追加or更新
        is_update:bool = checkId( user.xId )
        if userId != ROOT_ID:
            if not is_update or userId != user.xId:
                raise Exception( f"invalid userid" )
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name='users',create_new=True)
            # 名前チェックの条件
            w = { 'name': user.name }
            if is_update:
                w = { '$and': [ w, { 'id': {'$ne': user.xId} }, ] }
            # 名前チェック
            res:chromadb.GetResult = collection.get( where=w )
            if len( res.get('ids',[]) )>0:
                raise Exception( f"already exists userName {user.name}")
            # Idチェック
            if is_update:
                res:chromadb.GetResult = collection.get( ids=[user.to_key()] )
                if len(res.get('ids',[]))!=1:
                    raise Exception( f"invalid userId {user.xId}")
                # passwd
                meta = res.get('metadatas')[0]
                orig_pw = meta.get('passwd') or ''
                if CrabDB._eqpw( user.passwd, orig_pw ):
                    user.passwd = orig_pw
                else:
                    user.passwd = self._crypt_passwd( user.passwd )
                # api_key
                orig_apikey = meta.get('openai_api_key')
                if f"{user.openai_api_key}"==CrabDB._before_crab(orig_apikey):
                    user.openai_api_key = orig_apikey
                else:
                    user.openai_api_key = self._crypt_apikey( user.openai_api_key )
            else:
                if user.name == ROOT_USERNAME:
                    user.xId = ROOT_ID
                else:
                    user.xId = self.get_next_id()
                # passwd
                user.passwd = self._crypt_passwd( user.passwd )
                user.openai_api_key = self._crypt_apikey( user.openai_api_key )
            collection.upsert( ids=[user.to_key()], metadatas=[user.to_meta()], embeddings=[EmptyEmbedding] )
            user.passwd = emptyToBlank( user.passwd, '' )[:18]
            user.openai_api_key = CrabDB._before_crab(user.openai_api_key)
        return user

    def delete_user(self, userId:int ) -> CrabUser:
        if not checkId(userId):
            raise Exception( f"can not delete user. invalid userId")
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name='users',create_new=True)
            collection.delete( ids=[encodeId(userId)] )

    def get_usernames(self) -> list[str]:
        usernames = [ u.name for u in self.get_users()]
        return usernames

    def get_username(self, userId:int ) ->str:
        if checkId(userId):
            for user in self.get_users():
                if user.xId == userId:
                    return user.name
        return None

    def get_bots(self, user_id:int) ->list[CrabBot]:
        allbots:list[CrabBot] = [ CrabBot(**meta) for meta in self.get_metadatas('bots') ]
        # 名前と権限で絞り込み
        bots:list[CrabBot] = [ bot for bot in allbots if bot.xId != DEFAULT_BOT_ID and bot.name != DEFAULT_BOT_NAME ]
        bots:list[CrabBot] = [ bot for bot in bots if bot.owner==user_id or len(bot.auth)==0 or user_id in bot.auth]
        # 名前でソート
        bots.sort( key=lambda bot: bot.name)
        # 先頭にdefaultを作る
        bots.insert(0, CrabBot( id=DEFAULT_BOT_ID ) )
        return bots
    
    def get_bot(self, userId:int, botId:int ) -> CrabBot:
        for bot in self.get_bots(userId):
            if bot.xId == botId:
                return bot
        return None

    def create_new_bot(self, userId:int, name=None ) ->CrabBot:
        if not checkId(userId):
            raise Exception('invalid userId')
        if DEFAULT_BOT_NAME==name:
            raise Exception('invalid bot name')
        botId:int = self.get_next_id()
        if isEmpty(name):
            name = f"new bot #{botId}"
        bot = CrabBot(id=botId, name=name, owner=userId, auth=userId)
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name='bots',create_new=True)
            collection.add( ids=[bot.to_key()], metadatas=[bot.encode()], embeddings=[EmptyEmbedding])
        return bot

    def update_bot( self, userId:int, bot:CrabBot ):
        if not checkId(userId) or bot is None or not checkId(bot.xId):
            return
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name='bots',create_new=True)
            collection.upsert( ids=[bot.to_key()], metadatas=[bot.encode()], embeddings=[EmptyEmbedding])
        return bot

    def get_threads(self, user_id:int ) ->list:
        all_threads:list[CrabThread] = [ CrabThread(**meta) for meta in self.get_metadatas('threads') ]
        thres:list[CrabThread] = [ thre for thre in all_threads if thre.owner==user_id or len(thre.auth)==0 or user_id in thre.auth]
        thres.sort( key=lambda thre: thre.xId, reverse=True )
        return thres
    
    def get_thread(self, user_id:int, threId:int ):
        for thre in self.get_threads(user_id):
            if thre.xId == threId:
                return thre
        return None

    def create_new_thread(self, user_id:int, botId:int=None, name=None ):
        if not checkId(botId):
            bot = self.get_bots(user_id)[0]
            botId = bot.xId
        threId = self.get_next_id()
        if isEmpty(name):
            name = f"new thread #{threId}"
        thre:CrabThread = CrabThread( id=threId, botId=botId, title=name, owner=user_id, auth=user_id )
        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name='threads',create_new=True)
            collection.add( ids=[thre.to_key()], metadatas=[thre.encode()], embeddings=[EmptyEmbedding])
        return thre

    def update_thread_auth(self, userId:int, threId:int, auth):
        with self._lock:
            thre:CrabThread = self.get_thread( userId, threId )
            if thre is None:
                return None
            auth = decodeIds( auth )
            thre.auth = auth
            collection:chromadb.Collection = self.get_collection(collection_name='threads',create_new=True)
            collection.upsert( ids=[thre.to_key()], metadatas=[thre.encode()], embeddings=[EmptyEmbedding])
            return thre

    def close_thread(self, threId:int ):
        if isinstance(threId,CrabThread):
            threId = threId.xId
        if not checkId(threId):
            return
        with self._lock:
            # メッセージが存在するか確認
            mcollection:chromadb.Collection = self.get_collection(collection_name='messages',create_new=True)
            res:chromadb.GetResult = mcollection.get( where={ 'threadId': threId }, limit=3 )
            if len(res.get('ids',[]))>0:
                return
            # メッセージが無ければ削除する
            tcollection:chromadb.Collection = self.get_collection(collection_name='threads',create_new=True)
            tcollection.delete( ids=[to_key(threId)] )

    def load_messages(self, userId:int, threId:int, num:int = 5 ):
        total:int = 0
        with self._lock:
            # メッセージが存在するか確認
            mcollection:chromadb.Collection = self.get_collection(collection_name='messages',create_new=True)
            # スレッドのメッセージ全部のidを集める
            chunk_size:int = 100
            offset:int = 0
            last_ids:list[str]=[]
            while True:
                res:chromadb.GetResult = mcollection.get( where={ '$and':[ {'type': CrabMessage.TypeId}, {'threadId': threId} ] }, include=[], limit=chunk_size, offset=offset )
                ids:list = res.get('ids',[])
                last_ids += ids
                ll = len(ids)
                if ll < chunk_size:
                    break
                offset += chunk_size
            # 前件総数
            total = len(last_ids)
            if total == 0:
                return [],0
            # idでソート
            last_ids.sort( key=lambda id: (len(id),id) )
            # 最後のnum件
            last_ids = last_ids[-num:]
            # コンテンツトメタをロード
            metadatas:list = self.get_metadatas( 'messages', ids=last_ids, with_content=True )
        # コンテンツとメタを合成
        messages:list[CrabMessage] = [ CrabMessage(**meta) for meta in metadatas ]
        # 念の為ソート
        messages.sort( key=lambda msg: msg.xId )
        return messages, total

    def add_message(self, userId:int, msg:CrabMessage ) ->CrabMessage:
        msg.xId = self.get_next_id()
        with self._lock:
            # メッセージを追加
            mcollection:chromadb.Collection = self.get_collection(collection_name='messages',create_new=True)
            mcollection.add( ids=[msg.to_key()], documents=[msg.content], metadatas=[msg.to_meta()], embeddings=[EmptyEmbedding] )
        return msg

    def get_tail_message(self, userId:int, threId:int, num:int = 5, *, api_key:str=None ):
        if not checkId(userId) or not checkId(threId):
            return None, None
        check_openai_api_key(api_key)
        w = { 'threadId': threId }
        messages:list[CrabMessage] = self.get_datas('messages',CrabMessage, where=w, offset=-num)
        if len(messages) == 0:
            return [],EmptyEmbedding
        # 最新メッセージのembeddingを計算する
        contents:str = "\n\n".join( [ f"{m.role}: {m.content}" for m in messages if not isEmpty(m.content) ] )
        emb:list[float] = self.create_embeddings(contents, api_key=api_key)

        return messages, emb

    def create_index_message(self, threId:int, userId:int=None, *, api_key:str=None ):
        check_openai_api_key( api_key )
        self.task_submit( self._task_create_index_message, userId, [threId], api_key=api_key )

    def _task_create_index_message(self, userId:int, ids:list[int], *, api_key:str=None, tokens=1500, task:CrabTask=None ):
        check_openai_api_key( api_key )
        for threId in ids:
            if not checkId(threId):
                continue
            #
            thre:CrabThread = self.get_data( 'threads', CrabThread, id=to_key(threId))
            update_title:bool = not thre.title or thre.title.find( f"#{thre.xId}" )>0
            # 最後のインデックスを取得
            #print(f"[task_create_index_message] thread:{threId}")
            w = { 'threadId': threId }
            indexlist:list[CrabIndex] = self.get_datas( 'messages', CrabIndex, where=w, offset=-1 )
            lastId = indexlist[-1].end if indexlist else 0
            #print(f"[task_create_index_message] lastId {lastId}")
            # 最後のインデックス以降のメッセージを取得
            w = {'$and': [ { 'threadId': threId }, { 'id': {'$gt': lastId }} ] }
            mesglist:list[CrabMessage] = self.get_datas( 'messages', CrabMessage, where=w )
            print(f"[task_create_index_message] thread:{threId} lastId {lastId} messages {len(mesglist)}")
            if not mesglist:
                continue
            batch_size=10
            ll = len(mesglist)
            tk_count=0
            mesg_seg_list:list[list[CrabMessage]] = []
            msg_seg:list[CrabMessage]=[]
            for mp, mesg in enumerate(mesglist):
                if msg_seg and (tk_count+mesg.get_tokens())>tokens:
                    mesg_seg_list.append(msg_seg)
                    msg_seg=[]
                    tk_count = 0
                msg_seg.append(mesg)
                tk_count += mesg.get_tokens()

                if len(mesg_seg_list)>=batch_size or ( mp+1==ll and len(mesg_seg_list)>0):
                    x_emb_list = [ EmptyEmbedding for x in mesg_seg_list]
                    # indexを生成
                    x_index_list = []
                    for x_msg_seg in mesg_seg_list:
                        x_index_list.append( CrabIndex( id=self.get_next_id(),botId=x_msg_seg[0].botId, threadId=threId, begin=x_msg_seg[0].xId, end=x_msg_seg[-1].xId ) )
                    # indexを保存
                    x_ids = [ m.to_key() for m in x_index_list]
                    x_meta_list = [ m.to_meta() for m in x_index_list]
                    with self._lock:
                        mcollection:chromadb.Collection = self.get_collection(collection_name='messages',create_new=True, api_key=api_key)
                        mcollection.add( ids=x_ids, embeddings=x_emb_list, metadatas=x_meta_list)
                    mesg_seg_list=[]
                    xx_ids = [index.xId for index in x_index_list]
                    self.task_submit( self._task_embedding_index_message, userId, xx_ids, api_key=api_key )
                    update_title = True
            if update_title:
                # ---------------------------------------------------
                # タイトル設定
                # ---------------------------------------------------
                # 直近のメッセージ
                try:
                    w = { 'threadId': threId }
                    mesglist:list[CrabMessage] = self.get_datas( 'messages', CrabMessage, where=w, offset=-4 )
                    # プロンプト
                    contents:str = "\n\n".join( [ f"{m.role}: {m.content}" for m in mesglist if not isEmpty(m.content) ] )
                    prompt = f"# Create a thread title from the following conversation.\n\n{contents}\n\n# Output only the title. No other explanations or conversations are required."
                    # LLM
                    client:OpenAI = OpenAI( api_key=api_key )
                    res: ChatCompletion = client.chat.completions.create(
                        messages=[ { 'role': CrabMessage.SYSTEM, 'content': prompt } ],
                        model='gpt-3.5-turbo', max_tokens=100
                    )
                    # result
                    new_title = res.choices[0].message.content
                    if not isEmpty(new_title):
                        # update title
                        thre.title = new_title
                        self.upsert_datas( 'threads', [thre] )
                except:
                    pass

    def _task_embedding_index_message(self, userId:int, ids:list[int], *, api_key:str=None, task:CrabTask=None ):
        check_openai_api_key( api_key )
        # インデックスを取得
        x_ids = to_keys(ids)
        index_list:list[CrabIndex] = self.get_datas('messages', CrabIndex, ids=x_ids )
        meta_list = [ m.to_meta() for m in index_list]
        content_list:list[str] = []
        for index in index_list:
            w = { '$and': [
                { 'threadId': index.threadId },
                { 'id': { '$gte': index.begin } },
                { 'id': { '$lte': index.end } },
            ]}
            mesgs:list[CrabMessage] = self.get_datas( 'messages', CrabMessage, where=w )
            contents:str = "\n\n".join( [ f"{m.role}: {m.content}" for m in mesgs if not isEmpty(m.content) ] )
            content_list.append( contents )
        # embeddingを生成
        print(f"[reIndexMessage] create embeddings {x_ids[0]} : {x_ids[-1]}")
        emb_list = self.create_embeddings( content_list, api_key=api_key )
        # indexを保存
        with self._lock:
            mcollection:chromadb.Collection = self.get_collection(collection_name='messages',create_new=True, api_key=api_key)
            mcollection.upsert( ids=x_ids, embeddings=emb_list, metadatas=meta_list)

    def retrive_message(self, *, userId:int, threId:int, botId:int, emb:list[float], excludeId:int, max_distance:float=None, tokens:int=None, api_key:str=None ) ->list[CrabMessage]:
        if not checkId(userId) or ( not checkId(threId) and not checkId(botId) ):
            return None
        if not isinstance(emb,list) or len(emb)<=0:
            raise Exception("empty emb")
        check_openai_api_key( api_key )
        with self._lock:
            w = [ {'begin': { '$lt': excludeId } } ]
            if threId:
                w.append( { 'threadId': threId } )
            if botId:
                w.append( { 'botId': botId } )
            w = { '$and': w }
            hit_indexs = self.query_datas( 'messages', CrabIndex, embeddings=[emb], where=w, n_results=10, max_distance=max_distance, api_key=api_key )
            if len(hit_indexs)==0:
                return []
            # 近すぎるもの、遠すぎるものを消す
            hit_indexs = [ idx for idx in hit_indexs if idx.distance<-0.001 or 0.001<idx.distance ]
            hit_indexs = [ idx for idx in hit_indexs if idx.threadId is not None and idx.begin is not None and idx.end is not None ]
            if len(hit_indexs)==0:
                return []
            # ソートして一番近い順にする
            #hit_indexs.sort( key=lambda index: (index.distance,index.begin-index.end ) )
            # 一番近いもの
            messages:list[CrabMessage] = []
            for best_index in hit_indexs:
                # ロードする
                w = { '$and': [
                        { 'type': CrabMessage.TypeId },
                        { 'threadId': best_index.threadId },
                        { 'id': { '$gte': best_index.begin }},
                        { 'id': { '$lte': best_index.end }},
                ]}
                # コンテンツトメタをロード
                messages += self.get_datas( 'messages', CrabMessage, where=w )
            # 念の為ソート
            messages.sort( key=lambda msg: msg.xId )
            return messages
    
    def get_summary(self, userId:str, threId:str ):
        if not checkId(userId) or not checkId(threId):
            return None
        sumId = f"sum{threId}"
        with self._lock:
            # 要約が存在するか確認
            lastId = None
            content = ''
            mcollection:chromadb.Collection = self.get_collection(collection_name='messages',create_new=True)
            sumRes:chromadb.GetResult = mcollection.get( ids=[sumId] )
            metadatas=sumRes.get('metadatas',[])
            docs=sumRes.get('documents',[])
            if len(metadatas)>0 and len(docs)>0:
                lastId = metadatas[0].get('end')
                content = docs[0]
            # 最後の要約以降のメッセージを取得する
            where = { 'threadId': threId }
            where = { '$and': [ where, { ''}]}
            res:chromadb.GetResult = mcollection.get( where={ 'threadId': threId }, include=[], limit=chunk_size, offset=offset )
            messages:list[CrabSummary] = [ CrabSummary(**meta) for meta in metadatas ]
        docs:list = res.get('documents',[])
        for idx,content in enumerate( docs ):
            messages[idx].content = content

    def get_file_id( self, filename, *, source=None, filetime:float=None, size:int=None, create_new=False ):
        f:CrabFile = self.get_file( filename, source=source, filetime=filetime, size=size, create_new=create_new )
        return f.xId if f is not None else None

    def get_file( self, filepath, *, source=None, filetime:float=None, size:int=None, create_new=False ):
        with self._lock:
            md5 = calculate_md5( filepath )
            # コンテンツトメタをロード
            metadatas:list = self.get_metadatas( 'files', ids=[md5] )
            files:list[CrabFile] = [ CrabFile(**meta) for meta in metadatas ]
            if files is not None and len(files)>0:
                return files[0]
            if not create_new:
                return None
            collection:chromadb.Collection = self.get_collection(collection_name='files',create_new=True)
            f = CrabFile( id=self.get_next_id(), md5=md5, source=source, size=size, createTime=filetime)
            key=f.to_key()
            meta = f.to_meta()
            collection.upsert( ids=[key,md5], metadatas=[ meta, meta ], embeddings=[[],[]] )
            return f

    def add_file_segment(self, seg:CrabFileSegment, *, collection_name='contents', api_key:str ):
        check_openai_api_key( api_key )
        if isinstance(seg,list) and len(seg)>0 and isinstance(seg[0],CrabFileSegment):
            ids = [ s.to_key() for s in seg]
            docs = [ s.content for s in seg]
            metas = [ s.to_meta() for s in seg]
        elif isinstance(seg,CrabFileSegment):
            ids = [ seg.to_key() ]
            docs = [ seg.content ]
            metas = [ seg.to_meta() ]
        else:
            return None

        with self._lock:
            collection:chromadb.Collection = self.get_collection(collection_name=collection_name,create_new=True,api_key=api_key)
            collection.add( ids=ids, documents=docs, metadatas=metas )
        if isinstance(seg,list):
            return ids
        else:
            return ids[0]

    def register_textfile( self, userId, filepath, *, source=None, filetime:float=None, api_key:str ):
        check_openai_api_key( api_key )
        fileId = None
        try:
            fileId = self.get_file_id(filepath)
            if not isEmpty(fileId):
                return fileId
            if isEmpty(source):
                source = os.path.basename(filepath)
            if not isinstance(filetime,float) or filetime<=0.0:
                filetime = os.path.getmtime(filepath)
            size = os.path.getsize(filepath)
            f = self.get_file(filepath,source=source,filetime=filetime,size=size,create_new=True)
            print( f" debug id:{f.xId}")
            fileId = f.xId
            self.task_submit( self._process_load_file, userId, [f.xId], file=f, filepath=filepath, api_key=api_key )
        except:
            traceback.print_exc()
        return fileId

    def get_file_status(self, userId, fileId ):
        ret=None
        with self._task_lock:
            for task in self._task_queue:
                if fileId in task.ids:
                    ret = task.progress
                    break
        return str(ret) if ret is not None else ""

    def task_submit(self, func, userId:int, ids:list[int], *, api_key:str=None, **kwargs ):
        if not hasattr(func,'__call__') or not checkId(userId) or not isinstance(ids,list):
            raise Exception(f'invalid arguments for task')
        check_openai_api_key( api_key )
        if len(ids)==0:
            return
        x_ids = [ i for i in ids]
        with self._task_lock:
            # 重複チェック
            for preTask in self._task_queue:
                if preTask.func==func:
                    for i in range( len(x_ids)-1,0,-1):
                        if x_ids[i] in preTask.ids:
                            del x_ids[i]
                        if len(x_ids)==0:
                            return
            # 投入
            task:CrabTask = CrabTask( func, userId, x_ids, api_key=api_key, **kwargs )
            self._task_queue.append( task )
            # 起動
            if self._task_thread is None:
                self._task_thread = Thread( target=self._fn_task, name='crabTasks', daemon=True )
                self._task_thread.start()
            self._task_lock.notify()

    def _fn_task(self):
        try:
            while True:
                with self._task_lock:
                    if len(self._task_queue)<=0:
                        self._task_thread = None
                        return
                    task:CrabTask = self._task_queue[0]
                task()
                with self._task_lock:
                    del self._task_queue[0]
                    self._task_lock.notify_all()
        except:
            traceback.print_exc()
            with self._task_lock:
                self._task_thread = None

    def task_join(self):
        print(f"[task_join]enter")
        with self._task_lock:
            if len(self._task_queue)==0:
                print(f"[task_join]exit")
                self._task_lock.notify()
                return
            self._task_lock.wait( timeout=1.0 )

    def _process_load_file(self, userId:int, ids:list[int], *, file:CrabFile, filepath, api_key, task:CrabTask=None):
        check_openai_api_key( api_key )
        collection_name='contents'
        try:
            task.progress="0%"
            size:int = os.path.getsize(filepath)
            readsize = CHUNK_CHARS * 2
            buffer = ''
            st=0
            ed=0
            batch = []
            with open(filepath,'r', encoding=detect_encode(filepath) ) as inp:
                eof = False
                while not eof:
                    readsize = (CHUNK_CHARS*3)-len(buffer)
                    chunk = inp.read( readsize )
                    pos:int = inp.tell()
                    per:int = int( pos*100/size )
                    task.progress = f"{per}% {pos}/{size}"
                    if chunk:
                        buffer += chunk
                    while not eof:
                        content, buffer = tksplit( buffer, CHUNK_CHARS )
                        if content is None:
                            if chunk:
                                break
                            else:
                                content = buffer
                                buffer = ''
                                eof = True
                        ed = st + len(content)
                        seg = CrabFileSegment( id=self.get_next_id(), fileId=file.xId, content=content, source=file.source, createTime=file.createTime, size=file.size, begin=st, end=ed )
                        batch.append( seg )
                        if len(batch)>10:
                            self.add_file_segment( batch, collection_name=collection_name,api_key=api_key)
                            batch = []
                        st = ed
                if len(batch)>0:
                    self.add_file_segment( batch, collection_name=collection_name,api_key=api_key)
                    batch = []
                task.progress = f"100% {size}/{size}"
        except:
            traceback.print_exc()
            task.progress = f"ERROR"

    def retrive_file(self, *, fileIds:list[int]=None, emb: list[float]=None, query:list[str]=None, max_results: int = 10, max_distance=0.3,api_key:str )->list[CrabFileSegment]:
        check_openai_api_key( api_key )
        if not isinstance(emb,list) or len(emb)==0:
            emb=None
        else:
            if isinstance(emb[0],float):
                emb = [ emb ]
            elif not isinstance(emb[0],list) or len(emb[0])==0 or not isinstance(emb[0][0],float):
                emb=None
        if isinstance(query,str):
            query = [ query ]
        elif not isinstance(query,list) or len(query)==0 or not isinstance(query[0],str):
            query = None
        if emb is None and query is None:
            raise Exception("c")
        w = None
        if isinstance(fileIds,list) and len(fileIds)>0:
            w = { 'fileId': { '$in': fileIds } }

        aaa:list[CrabFileSegment] = self.query_datas( 'contents', CrabFileSegment, embeddings=emb, texts=query, where=w, n_results=max_results, max_distance=max_distance, api_key=api_key)
        results: list[CrabFileSegment] = [ seg for seg in aaa if not isEmpty(seg.source) and 0<=seg.begin and seg.begin<=seg.end ]
        results.sort( key=lambda seg:(seg.fileId,seg.begin) )
        return results

    def create_tempfile(self) ->str:
        if self.tmp_path is not None:
            fileId:int = self.get_next_id()
            return os.path.join( self.tmp_path, f"data_{fileId:010d}")
        else:
            return None

    def get_file_name( self, userId:int, fileId:int ) ->str:
        if not checkId(userId) or not checkId(fileId):
            raise Exception(f'invalid id {userId}/{fileId}')
        for meta in self.get_metadatas( 'files', ids=fileId ):
            f:CrabFile = CrabFile( **meta )
            if f.xId == fileId:
                return f.source
        return None

    def register_file(self, userId:int, botId:int, filepath:str, *, source:str=None, filetime:float=None, api_key:str ) ->int:
        check_openai_api_key( api_key )
        if not checkId(userId) or not checkId(botId):
            raise Exception(f'invalid id {userId}/{botId}')
        if not os.path.exists( filepath ) or isEmpty(source):
            raise Exception(f'invalid file {filepath}/{source}')
        print( f"uploadFile botId:{botId} filepath:{filepath}, filename:{source}" )
        fileId:int = self.register_textfile( userId, filepath, source=source, filetime=filetime, api_key=api_key )
        if not checkId(fileId):
            raise Exception(f"can not load file {filepath} {source}")
        bot:CrabBot = self.get_bot( userId, botId )
        if bot is None:
            raise Exception(f"can not file botId:{botId}")
        if isinstance(bot.files,list):
            bot.files.append( fileId )
        else:
            bot.files = [fileId]
        self.update_bot( userId, bot )
        return fileId

    def removeFile(self, userId:int, botId:int, fileId:int ):
        if not checkId(userId) or not checkId(botId) or not checkId(fileId):
            raise Exception(f'invalid id {userId}/{botId}')
        bot:CrabBot = self.get_bot( userId, botId )
        if bot is None:
            raise Exception(f'invalid id {userId}/{botId}')
        aaa = [x for x in decodeIds( bot.files ) if x!=fileId ]
        bot.files = aaa
        self.update_bot( userId, bot )

    def reindex(self, model=DEFAULT_EMBEDDING_MODEL, api_key=None):
        check_openai_api_key( api_key )
        userId = ROOT_ID
        try:
            with self._lock:
                if not self.set_embedding_model( model ):
                    return #変わってない
                # カレントモデルとDBのモデルが違っていれば、既存のindexを削除する
                tcollection:chromadb.Collection = self.get_collection(collection_name='messages',create_new=True)
                tcollection.delete( where={ 'type': CrabIndex.TypeId } )
            thlist:list[CrabThread] = self.get_datas( 'threads', CrabThread )
            batch_size:int = 40
            offset:int = 0
            while True:
                index_list:list[CrabIndex] = self.get_datas( 'threads', CrabIndex, offset=offset, limit=batch_size )
                if not index_list:
                    break
                ids:list[int] = [ i.xId for i in index_list]
                self.task_submit( self._task_embedding_index_message, userId, ids, api_key=api_key )
                offset+=len(index_list)
            offset = 0
            while True:
                seg_list:list[CrabFileSegment] = self.get_datas( 'contents', CrabFileSegment, offset=offset, limit=batch_size )
                if not seg_list:
                    break
                ids:list[int] = [ i.xId for i in seg_list]
                self.task_submit( self._task_embedding_file_segment, userId, ids, api_key=api_key )
                offset+=len(seg_list)
        except:
            traceback.print_exc()

    def _task_embedding_file_segment(self, userId:int, ids:list[int], *, api_key, task:CrabTask=None ):
        # ファイルのembeddingを作りなおし
        offset:int = 0
        limit:int = 50
        while True:
            x_ids = to_keys(ids)
            datas:list[CrabFileSegment] = self.get_datas( 'contents', CrabFileSegment, ids=to_keys(ids) )
            if len(datas)==0:
                break
            x_metas:list = [ m.to_meta() for m in datas]
            x_contents:list[str] = [ m.content for m in datas]
            x_embs:list[list] = self.create_embeddings( x_contents, api_key=api_key )
            with self._lock:
                ccollection:chromadb.Collection = self.get_collection( collection_name='contents' )
                ccollection.update( ids=x_ids, documents=x_contents, embeddings=x_embs, metadatas=x_metas )
            offset += len(datas)

class CrabThread(CrabType):

    def __init__(self, *, session=None, id:int=None, botId:int=None, title=None, owner=None, auth=None, createTime=None ):

        self._session:CrabSession = session
        super().__init__(id=id)
        self.title:str = emptyToBlank( title, '' )
        self.owner:int = decodeId( owner)
        self.auth:list[str] = decodeIds(auth)
        self.botId:int = decodeId(botId)
        self.messages:list[CrabMessage] = None
        self.total = 0
        self.createTime:float = parseFloat( createTime, 0.0 )
        if self.createTime < 0.1:
            self.createTime = time.time()
        #
        self.bot:CrabBot = None

    def to_meta(self):
        return self.encode()

    def encode(self):
        return {
            'id': self.xId,
            'title': self.title,
            'owner': encodeId(self.owner),
            'auth': encodeIds( self.auth),
            'botId': encodeId(self.botId),
            'createTime': self.createTime,
        }

    def close(self):
        pass

    def load_bot(self) ->CrabBot:
        if self.bot is None:
            if not checkId(self.botId):
                self.botId = DEFAULT_BOT_ID
            self.bot = self._session.get_bot( self.botId )
            if self.bot is None:
                raise Exception( f'can not get botId#{self.botId}')
        return self.bot

    def get_prompt(self) ->str:
        bot:CrabBot = self.load_bot()
        return bot.prompt

    def get_model(self) ->str:
        bot:CrabBot = self.load_bot()
        return bot.model

    def get_bot_name(self) ->str:
        try:
            bot:CrabBot = self.load_bot()
            return bot.name
        except:
            return f"CanNotLoad{self.botId}"

    def get_bot_description(self) ->str:
        try:
            bot:CrabBot = self.load_bot()
            return bot.description or ""
        except:
            return f"CanNotLoad{self.botId}"

    def get_messages(self, num:int=5 )->list[CrabMessage]:
        ll = -1 if self.messages is None else len( self.messages )
        if ll<0 or num>ll and self.total>ll:
            self.messages, self.total = self._session._load_messages( self.xId, num=num )
        return self.messages[-num:]

    def get_last_time(self) ->float:
        mesgs:list[CrabMessage] = self._load_messages()
        ll = len(mesgs)
        if ll==0:
            return self.createTime
        else:
            mesgs[-1].time

    def add_message(self, message):
        msg:CrabMessage = CrabMessage( **message )
        msg.threadId = self.xId
        msg.botId = self.botId
        self._session._add_message( msg )
        if isinstance(self.messages,list):
            self.messages.append( msg )
        else:
            self.messages = [msg]
        self.total+=1

    def add_user_message(self, message):
        self.add_message({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.add_message({"role": "assistant", "content": message})

    def create_index_message(self):
        self._session.create_index_message( self.xId )

    @staticmethod
    def _trim_retrive_messages( arrays:[list[list[CrabType]]], max_tokens ):
        # リストを結合
        pack = []
        for array in arrays:
            if isinstance(array,list):
                pack.extend(array)
        # 距離でソート
        pack.sort( key=lambda m: m.distance )
        # 入力トークン調整:制限を超えるメッセージにマーキング
        tokens:int = 0
        is_over:bool = False
        for m in pack:
            is_over = is_over or (tokens+m.get_tokens())>max_tokens
            if is_over:
                m.distance=None
            else:
                tokens+=m.get_tokens()
        # 超えたやつを削除
        for array in arrays:
            if isinstance(array,list):
                for idx in range( len(array)-1, -1, -1 ):
                    if array[idx].distance is None:
                        del array[idx]
        return tokens

    def run(self, message:str=None, *, verbose=True ):
        try:
            bot = self.load_bot()
            if bot is None:
                yield "ERROR:Can not load bot."
                return
                
            if not isEmpty(message):
                # ユーザの入力をチャット履歴に追加する
                self.add_user_message( message )
            
            openai_model:OpenAIModel = CrabBot.get_model(bot.model)

            # 入力トークン数のカウントと調整
            max_input_tokens:int = int( min(bot.input_tokens,openai_model.input_tokens)*0.95 )
            input_tokens:int = 2
            # プロンプト
            request_prompt:str = None
            if bot.llm:
                request_prompt = bot.prompt
                request_prompt = request_prompt.replace('${datetime}', current_date_time() )
                input_tokens += 4 + CrabMessage._SYSTEM_TOKENS + count_tokens( request_prompt )
            xrag = ( bot.rag and len(bot.files)>0 )
            # 最近の会話履歴
            tail_messages = []
            query_embedding:list[float] = None
            if bot.llm:
                # llmを使う場合
                if verbose:
                    yield CrabBot.CLS
                    yield "get last messages..."
                tail_messages, query_embedding = self._session._get_tail_messages(self.xId, 10)
                excludeId= tail_messages[0].xId if len(tail_messages)>0 else 0
            elif bot.retrive or xrag:
                # llmを使わないが検索だけする場合
                chunk_count = 1
                if verbose:
                    yield CrabBot.CLS
                    yield "get embedding..."
                mesgs, query_embedding = self._session._get_tail_messages( self.xId, chunk_count )
                excludeId= mesgs[0].xId if len(mesgs)>0 else 0
            # 過去の会話履歴を検索する
            hist_messges:list = None
            if bot.retrive:
                max_distance = 0.3
                if verbose:
                    yield CrabBot.CLS
                    yield "retrive messages..."
                hist_messges:list = self._session._retrive_message( botId=self.botId, emb=query_embedding, excludeId=excludeId, max_distance=max_distance )
            # ファイルから検索する
            segs:list = None
            if xrag:
                max_distance = 0.3
                if verbose:
                    yield CrabBot.CLS
                    yield "retrive files"
                segs:list = self._session._retrive_file( fileIds=bot.files, emb=query_embedding, max_distance=max_distance )
            # 入力トークン調整:リストを結合
            retrive_tokens = CrabThread._trim_retrive_messages( (hist_messges,segs), (max_input_tokens-input_tokens)//2)
            input_tokens += retrive_tokens
            # 入力トークン調整:制限を超えるメッセージにマーキング
            last_tokens = CrabMessage.trim_messages( tail_messages, (max_input_tokens-input_tokens) )
            input_tokens += last_tokens
                    
            # 出力トークン数制限
            output_tokens:int = min( openai_model.output_tokens, bot.max_tokens - input_tokens )

            # 入力を構築する
            request_messages:list = []
            if bot.llm:
                request_messages += [ { 'role': CrabMessage.SYSTEM, 'content': bot.prompt } ]
            request_messages += [ m.to_obj() for m in asArray(hist_messges) ]
            request_messages += [ m.to_obj() for m in asArray(segs) ]
            request_messages += [ m.to_obj() for m in asArray(tail_messages) ]

            # 実行
            predata = []
            stream = None
            if bot.llm:
                if verbose:
                    yield CrabBot.CLS
                    yield "execute LLM..."
                try:
                    client:OpenAI = OpenAI( api_key=self._session._get_openai_api_key())
                    stream = client.chat.completions.create(
                        messages=request_messages,
                        model=openai_model.model, max_tokens=output_tokens, temperature=bot.temperature,
                        stream=True
                    )
                    yield CrabBot.CLS
                    buffer = ""
                    for part in stream:
                        seg = part.choices[0].delta.content or ""
                        buffer += seg
                        yield seg
                    # ChatBotの返答をチャット履歴に追加する
                    if not isEmpty(buffer):
                        self.add_assistant_message( buffer )
                        # reindex
                        self.create_index_message()
                except openai.OpenAIError as ex:
                    yield f"{ex}"
                except Exception as ex:
                    yield f"{ex}"
            else:
                yield CrabBot.CLS
                if hist_messges is not None:
                    yield "### result of History\n"
                    for m in hist_messges:
                        yield "#### "+m.to_dump()+"\n"+m.to_content()+"\n"
                if segs is not None:
                    yield "### result of RAG\n"
                    for m in segs:
                        yield "#### "+m.to_dump()+"\n"+m.to_content()+"\n"
        except openai.AuthenticationError as ex:
            yield f"{type(ex).__name__}: {ex.message}"
        except openai.OpenAIError as ex:
            yield f"{type(ex).__name__}: {ex.message}"
        #openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 4097 tokens. However, you requested 5390 tokens (1294 in the messages, 4096 in the completion). Please reduce the length of the messages or completion.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}

class CrabSession:

    def __init__(self, db:CrabDB, user:CrabUser ):
        self.db:CrabDB = db
        self.xId:int = db.get_next_id()
        self.user:CrabUser = user
        self.login:float = time.time()
        self.last:float = self.login
        self.current_thread:CrabThread = None

    def _close_current_thread(self):
        if self.current_thread is not None:
            self.db.close_thread( self.current_thread )
            self.current_thread.close()
            self.current_thread=None

    def logout(self):
        self._close_current_thread()
        self.db=None

    def _update(self):
        if self.db is None:
            raise Exception("Session is not available")
        self.last = time.time()

    def is_root(self, user ) ->bool:
        if isinstance(user,str):
            return ROOT_USERNAME == user
        elif isinstance(user,int):
            return ROOT_ID == user
        elif isinstance(user,CrabUser):
            return user.xId == ROOT_ID
        return False

    def get_username(self, userId:int ) ->str:
        self._update()
        return self.db.get_username( userId )

    def upsert_user( self, user ):
        self._update()
        self.db.upsert_user( self.user.xId, user )

    def _get_openai_api_key(self) ->str:
        self._update()
        return self.db.get_openai_key( self.user.xId )

    def get_bots(self) ->list[CrabBot]:
        self._update()
        return self.db.get_bots(self.user.xId)

    def get_bot(self,botId:str) -> CrabBot:
        self._update()
        return self.db.get_bot(self.user.xId, botId )

    def create_new_bot(self) -> CrabBot:
        self._update()
        return self.db.create_new_bot( self.user.xId )

    def update_bot(self, bot:CrabBot ):
        self._update()
        self.db.update_bot(self.user.xId, bot )
        if self.current_thread is not None and self.current_thread.botId==bot.xId:
            self.current_thread.bot=None
            self.current_thread.load_bot()

    def get_threads(self) ->list[CrabThread]:
        self._update()
        return self.db.get_threads(self.user.xId)
    
    def get_current_thread(self):
        self._update()
        if self.current_thread is None:
            self.current_thread = self.db.create_new_thread( self.user.xId )
        self.current_thread._session = self
        return self.current_thread

    def create_thread(self,botId):
        self._update()
        self._close_current_thread()
        self.current_thread = self.db.create_new_thread( self.user.xId, botId=botId )
        self.current_thread._session = self

    def set_current_thread(self,threId:int=None):
        self._update()
        if not checkId(threId):
            self._close_current_thread()
            return
        if self.current_thread and self.current_thread.xId==threId:
            return
        thre = self.db.get_thread(self.user.xId, threId)
        if thre is not None:
            self._close_current_thread()
            self.current_thread = thre
            self.current_thread._session = self

    def update_thread_auth(self, threId:int, auth ):
        self._update()
        thre:CrabThread = self.db.update_thread_auth( self.user.xId, threId, auth )
        if thre is not None and self.current_thread is not None and self.current_thread.xId==threId:
            self.current_thread.auth = thre.auth

    def _load_messages(self, threId:int, num:int=5 )->list[CrabMessage]:
        self._update()
        mesgs,total = self.db.load_messages( self.user.xId, threId, num=num )
        return mesgs,total

    def _add_message(self, msg:CrabMessage )->CrabMessage:
        self._update()
        return self.db.add_message( self.user.xId, msg )

    def create_index_message( self, threId:int ):
        self._update()
        self.db.create_index_message( threId, self.user.xId, api_key=self._get_openai_api_key() )

    def _get_tail_messages(self, threId:int, num:int=10 )->list[CrabMessage]:
        self._update()
        mesgs,emb = self.db.get_tail_message( self.user.xId, threId, num=num, api_key=self._get_openai_api_key() )
        return mesgs,emb

    def _retrive_message(self, *, threId:int=None, botId:int=None, emb=None, excludeId=None, max_distance:float=None, tokens:int=None )->list[CrabMessage]:
        self._update()
        mesgs = self.db.retrive_message( userId=self.user.xId, threId=threId, botId=botId, emb=emb, excludeId=excludeId, max_distance=max_distance, tokens=tokens, api_key=self._get_openai_api_key() )
        return mesgs

    def _retrive_file(self, *, fileIds:list[int], emb, max_distance:float=None )->list[CrabFileSegment]:
        self._update()
        mesgs = self.db.retrive_file( fileIds=fileIds, emb=emb, max_distance=max_distance, api_key=self._get_openai_api_key() )
        return mesgs

    def create_tempfile(self) ->str:
        return self.db.create_tempfile()

    def get_file_name( self, fileId:int ) ->str:
        return self.db.get_file_name( self.user.xId, fileId )

    def get_file_status( self, fileId:int ) ->str:
        return self.db.get_file_status( self.user.xId, fileId )

    def register_file(self, botId:int, filepath:str, *, source:str=None, filetime:float=None ):
        print( f"register_file botId:{botId} filepath:{filepath}, filename:{source}" )
        return self.db.register_file( self.user.xId, botId, filepath, source=source, filetime=filetime, api_key=self._get_openai_api_key() )

    def removeFile(self, botId:int, fileId:int ):
        print( f"removeFile botId:{botId} fileId:{fileId}" )
        self.db.removeFile( self.user.xId, botId, fileId )

class CrabClient:
    def __init__(self):
        self.db:CrabDB = None

def main():
    chroma_client:CrabDB = CrabDB( on_memory=True )
    names1 = chroma_client.get_usernames()
    print( f" first {names1}")
    u = chroma_client.upsert_user( ROOT_ID, 'maeda' )
    names2 = chroma_client.get_usernames()
    print( f" second {names2}")
    u = chroma_client.upsert_user( ROOT_ID, 'shigeki' )
    names2 = chroma_client.get_usernames()
    print( f" second {names2}")

    user:CrabUser = chroma_client.login( 'maeda' )
    if user is None:
        print(f"ERROR: can not login")
        return
    bots = chroma_client.get_bots( user_id=user.xId )
    for bot in bots:
        print(f"bot {bot.name}")
    bot:CrabBot = chroma_client.create_new_bot(  userId=user.xId , name='sample2' )
    bots = chroma_client.get_bots( user_id=user.xId )
    for bot in bots:
        print(f"bot {bot.name}")

if __name__ == '__main__':
    #emb_test()
    main()
    #test_model()