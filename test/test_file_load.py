import sys, os, time
import re
import unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
import tiktoken
from crabDB import isEmpty, ROOT_ID, CrabTask, CrabDB, CrabSession, CrabThread, CrabBot, CrabMessage, CrabFileSegment

def main2():
    #create_emb( 'abc123' )
    api_key=os.environ['OPENAI_API_KEY']
    chroma_client:CrabDB = CrabDB( on_memory=True )
    filename_list = [ 'hashire_merosu.txt', 'gingatetsudono_yoru.txt', 'gakumonno_susume.txt', 'bocchan.txt' ]
    filename_list = filename_list[:1]
    for filename in filename_list:
        print( f"Load {filename}")
        file_path = f"testData/{filename}"
        before_id = chroma_client.get_file_id( file_path )
        fileId = chroma_client.register_textfile( ROOT_ID, file_path, api_key=api_key )
        while True:
            st = chroma_client.get_file_status( ROOT_ID, fileId )
            if isEmpty(st):
                print(f"{fileId} done")
                break
            print(f"{fileId} {st}")
            time.sleep(5.0)
        after_id = chroma_client.get_file_id( file_path )
        print( f"    fileId {before_id}  {fileId}  {after_id}")
    print("-------------------------")    
    query_list=['セリヌンティウスの頬を殴ったのは誰ですか？','銀河鉄道に乗ったのは誰？']
    for query in query_list:
        print(f"\nQuery: {query}")
        print(f"RETRIVE")
        reslist:list[CrabFileSegment] = chroma_client.retrive_file( query=query,api_key=api_key)
        for seg in reslist:
            print("-----------------------")
            print(f"dist:{seg.distance} source:{seg.source} range:{seg.begin}-{seg.end}\n{seg.content}")
        print(f"EMBEDDING")
        emb = chroma_client.create_embeddings(query,api_key=api_key)
        reslist:list[CrabFileSegment] = chroma_client.retrive_file( emb=emb,api_key=api_key)
        for seg in reslist:
            print("-----------------------")
            print(f"dist:{seg.distance} source:{seg.source} range:{seg.begin}-{seg.end}\n{seg.content}")

if __name__ == '__main__':
    #emb_test()
    main2()
    #test_model()