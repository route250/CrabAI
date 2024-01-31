import sys, os, time, traceback, inspect
import unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
from src.pyutils import isEmpty
from testutils import test_title, Fail
from test_data_download import download_and_extract_zip
from crabDB import CrabDB, CrabSession, CrabThread, CrabBot, CrabMessage, CrabFileSegment
import crabDB

class TestEmbeddings(unittest.TestCase):

    def xtest_list_models(self):
        case:str = test_title()
        client:CrabDB = CrabDB( on_memory=True )
        models=[ x for x in CrabBot.MODEL_LIST if x.name.startswith('gpt-')]
        mess=[ { 'role':'user', 'content': 'hello' } ]
        for model in models:
            print( f"--------------------" )
            print( f"{model.name}:{model.model}")
            # try:
            #     ret = client.chat.completions.create( model=model, messages=mess, max_tokens=100000000 )
            # except Exception as ex:
            #     print( ex )

    def xtest_get_embeddings(self):
        case:str = test_title()
        result:bool = True
        # os.environ['OPENAI_API_KEY']
        client:CrabDB = CrabDB( on_memory=True )
        txt='abc'
        try:
            emb = client.create_embeddings( txt )
            result = Fail('')
        except Exception as ex:
            print(ex)
        try:
            emb = client.create_embeddings( txt, api_key='invalid debug test' )
            result = Fail('')
        except Exception as ex:
            print(ex)
        try:
            api_key= client.get_public_openai_key()
            emb = client.create_embeddings( txt, api_key=api_key )
            print( f"emb len:{len(emb)}")
            print( emb[:20])
        except:
            traceback.print_exc()
            result = Fail('')

    def test_retrive_meros(self):
        case:str = test_title()
        result:bool = True
        # os.environ['OPENAI_API_KEY']
        client:CrabDB = CrabDB( on_memory=True )
        api_key = client.get_public_openai_key()
        ROOT_ID=client.get_user('root').xId
        filename_list = [ 'hashire_merosu.txt' ]
        for filename in filename_list:
            print( f"Load {filename}")
            file_path = download_and_extract_zip(filename)
            before_id = client.get_file_id( file_path )
            if before_id is not None:
                raise Exception("hate?")
            fileId = client.register_textfile( ROOT_ID, file_path, api_key=api_key )
            if fileId is None:
                raise Exception("invalid fileId?")
            while True:
                st = client.get_file_status( ROOT_ID, fileId )
                if isEmpty(st):
                    print(f"{fileId} done")
                    break
                print(f"{fileId} {st}")
                time.sleep(2.0)
        print("-------------------------")
        MODELS=['text-embedding-3-small','text-embedding-ada-002']
        # MODELS=['text-embedding-ada-002','text-embedding-3-small']
        query_list=['セリヌンティウスの頬を殴ったのは誰ですか？','○○氏は、セリヌンティウスの頬を殴った。'
                    ,'王様の名前は？','王様の名前は鈴木です。',
                    'セリヌンティウスの職業は？','セリヌンティウスはエンジニアです。']
        for embmodel in MODELS:
            if client.embeddingModel != embmodel:
                break
                print("-------------------------")
                print("  REINDEX {model}")
                print("-------------------------")
                client.reindex( embmodel, api_key=api_key )
            for query in query_list:
                print(f"\nRETRIVE {query}")
                reslist:list[CrabFileSegment] = client.retrive_file( query=query,api_key=api_key)
                print(f"---- Result {query}")
                if len(reslist)>0:
                    for seg in reslist:
                        print(f"dist:{seg.distance} source:{seg.source} range:{seg.begin}-{seg.end}\n{seg.content}")
                else:
                    print("    No Result")
                for model in MODELS:
                    print(f"\nRETRIVE {model} {query}")
                    emb = client.create_embeddings(query,model=model,api_key=api_key)
                    reslist:list[CrabFileSegment] = client.retrive_file( emb=emb, api_key=api_key)
                    print(f"---- Result {model} {query}")
                    if len(reslist)>0:
                        for seg in reslist:
                            print(f"dist:{seg.distance} source:{seg.source} range:{seg.begin}-{seg.end}\n{seg.content}")
                    else:
                        print("    No Result")

# def main2():
#     #create_emb( 'abc123' )
#     api_key=os.environ['OPENAI_API_KEY']
#     client:CrabDB = CrabDB( on_memory=True )
#     filename_list = [ 'hashire_merosu.txt', 'gingatetsudono_yoru.txt', 'gakumonno_susume.txt', 'bocchan.txt' ]
#     filename_list = filename_list[:1]
#     for filename in filename_list:
#         print( f"Load {filename}")
#         file_path = f"testData/{filename}"
#         before_id = client.get_file_id( file_path )
#         fileId = client.register_textfile( ROOT_ID, file_path, api_key=api_key )
#         while True:
#             st = client.get_file_status( ROOT_ID, fileId )
#             if isEmpty(st):
#                 print(f"{fileId} done")
#                 break
#             print(f"{fileId} {st}")
#             time.sleep(2.0)
#         after_id = client.get_file_id( file_path )
#         print( f"    fileId {before_id}  {fileId}  {after_id}")
#     print("-------------------------")    
#     query_list=['セリヌンティウスの頬を殴ったのは誰ですか？','銀河鉄道に乗ったのは誰？']
#     for query in query_list:
#         print(f"\nQuery: {query}")
#         print(f"RETRIVE")
#         reslist:list[CrabFileSegment] = client.retrive_file( query=query,api_key=api_key)
#         for seg in reslist:
#             print("-----------------------")
#             print(f"dist:{seg.distance} source:{seg.source} range:{seg.begin}-{seg.end}\n{seg.content}")
#         print(f"EMBEDDING")
#         emb = client.create_embedding(query,api_key=api_key)
#         reslist:list[CrabFileSegment] = client.retrive_file( emb=emb,api_key=api_key)
#         for seg in reslist:
#             print("-----------------------")
#             print(f"dist:{seg.distance} source:{seg.source} range:{seg.begin}-{seg.end}\n{seg.content}")

if __name__ == '__main__':
    unittest.main()