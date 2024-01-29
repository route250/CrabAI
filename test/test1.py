import sys, os, time
import re
import unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
import tiktoken
from crabDB import CrabTask, CrabDB, CrabSession, CrabThread, CrabBot, CrabMessage, CrabFileSegment

def str_dump( value, max:int=100 ) ->str:
    dmp=str(value).replace('\n','\\n')
    ll = len(dmp)
    if ll<=max:
        return dmp
    seg = (max-5)//2
    return dmp[:seg]+"****"+dmp[-seg:]

def text_split( text:str ) ->str:
    # 後ろに改行がない改行、または前に空白がない空白、または後ろに空白がない空白、または丸
    pattern = r'\n(?!\n)|(?<!\s)|\s(?!\s)|[、。](?![\n])'
    pattern = r'[\n、。](?![\n、。」])'
    # パターンに一致する位置を探す
    start_pos:int=0
    for match in re.finditer(pattern, text):
        end_pos = match.end()
        if start_pos<end_pos:
            yield text[start_pos:end_pos]
            start_pos=end_pos
    if start_pos<len(text):
        yield text[start_pos:]

def tk_text_split( text:str, tokens:int ) ->list:
    tkenc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    buf=[]
    for subtext in text_split(text):
        subtk = tkenc.encode(subtext)
        if len(buf)+len(subtk)> tokens:
            yield buf
            buf=subtk
        else:
            buf+=subtk
    if len(buf)>0:
        yield buf

def tksplit(text: str, tokens: int = 1024) -> list[str]:
    result = []
    tkenc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tk = tkenc.encode(text)
    ll = len(tk)

    # 後ろに改行または空白がない、改行または空白の正規表現パターン
    pattern = r'[\n\s](?![\n\s])'
    aa = text.split(pattern,text)
    # パターンに一致する位置を探す
    matches = re.finditer(pattern, text)

    i=text.find('')
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
class TestA(unittest.TestCase):

    def xxtest_sample(self):
        print(f"test sample")
        key=1000
        hist_messages=[]
        for i in range(0,10):
            role = CrabMessage.USER if i%2==0 else CrabMessage.ASSISTANT
            hist_messages.append( CrabMessage( id=key, role=role, content=f"メッセージ{i}", distance=float((10-i)/10) ) )
            key+=1
        file_segments=[]
        for i in range(0,10):
            file_segments.append( CrabFileSegment( id=key, fileId=1, content=f"セグメント{i}", distance=float((10-i)/10) ) )
            key+=1
        total_tokens = 0
        print("[INPUT]hist_messages")
        for m in hist_messages:
            total_tokens += m.get_tokens()
            print( m.to_meta() )
        print("[INPUT]file_segments")
        for m in file_segments:
            total_tokens += m.get_tokens()
            print( m.to_meta() )
        print("[TRIM]")
        print(f"total_tokens:{total_tokens}")
        target_tokens = total_tokens//2
        ret_tokens = CrabThread._trim_retrive_messages( (hist_messages,file_segments), target_tokens )
        print(f"ret_tokens:{ret_tokens}")

        input_tokens = 0
        print("[OUTPUT]hist_messages")
        for m in hist_messages:
            input_tokens += m.get_tokens()
            print( m.to_meta() )
        print("[OUTPUT]file_segments")
        for m in file_segments:
            input_tokens += m.get_tokens()
            print( m.to_meta() )
        print(f"input_tokens:{input_tokens}")

    def test_retrive_message(self):
        print("xxx")
        file_path='testData/hashire_merosu.txt'
        # login
        client:CrabDB = CrabDB( on_memory=True )
        Session:CrabSession = client.login( 'root' )
        Thre:CrabThread = Session.get_current_thread()
        # load text
        with open(file_path,'r',encoding='cp932') as file:
            file_contents=file.read()
        # split and add message
        rolesw:bool=False
        tkenc = tiktoken.encoding_for_model('gpt-3.5-turbo')
        for tk in tk_text_split( file_contents, 512 ):
            line = tkenc.decode(tk)
            print(f"add_message {str_dump(line)}")
            if rolesw:
                Thre.add_user_message( line )
            else:
                Thre.add_assistant_message( line )
            rolesw = not rolesw
            # reindex
            #client.task_submit( ( Thre, None, Session._get_openai_api_key() ) )
            Thre.create_index_message()
            #Thre.re_index_message()
        time.sleep(10.0)
        client.task_join()
        # add query
        Thre.add_user_message( 'メロスの親友の名前は？' )
        mesgs, query_embedding = Session._get_tail_messages( Thre.xId, 1 )
        excludeId= mesgs[0].xId if len(mesgs)>0 else 0
        # retrive
        messages:list[CrabMessage] = Session._retrive_message( threId=Thre.xId, botId=Thre.botId, emb=query_embedding, excludeId=excludeId, max_distance=0.3)
        for msg in messages:
            print("---")
            print( msg.content )

if __name__ == '__main__':
    unittest.main()