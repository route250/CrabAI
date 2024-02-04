import sys, os, time
import re
import unittest
from openai import OpenAI, OpenAIError
import tiktoken
import matplotlib.pyplot as plt
import japanize_matplotlib
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
from test_data_download import download_and_extract_zip

from crab.embeddings import EmbeddingFunction, create_embeddings, cosine_similarity, split_text

class EmbeddingFunc(EmbeddingFunction):
    def __init__(self, client, model, dim=None, *, color='r' ):
        super().__init__( client, model=model, dimensions=dim )
        self.color = color

client:OpenAI = OpenAI()
fn_3small = EmbeddingFunc(client,"text-embedding-3-small", color='blue')
fn_3small_256 = EmbeddingFunc(client,"text-embedding-3-small",256, color='aqua')
fn_3large = EmbeddingFunc(client,"text-embedding-3-large", color='crimson')
fn_3large_1536 = EmbeddingFunc(client,"text-embedding-3-large",1536, color='tomato')
fn_3large_256 = EmbeddingFunc(client,"text-embedding-3-large",256, color='brown')
fn_ada002 = EmbeddingFunc(client,"text-embedding-ada-002", color='green')
Function_list=[
    fn_3small,
    fn_3large,
    fn_3large_1536,
    fn_3small_256,
    fn_3large_256,
    fn_ada002,
]

input_text_file = download_and_extract_zip('hashire_merosu.txt')
Query_list = [
    {'query': 'メロスは何のためにシラクスの市にやって来たのですか？', 'segment':0 },
    {'query': 'メロスの職業は何ですか？', 'segment':0 },
    {'query': 'メロスにはどのような家族がいますか？', 'segment':0 },
    {'query': 'メロスは市に入ってから何が原因で捕縛されましたか？', 'segment':1 },
    {'query': 'メロスは王に対して、どのような非難をしましたか？', 'segment':1 },
    {'query': '王様の名前は何ですか？', 'segment':1 },
    {'query': 'メロスはなぜ処刑までに三日間の猶予を求めましたか？', 'segment':2 },
    {'query': '王はメロスにどのような条件を提示しましたか？', 'segment':2 },
    {'query': 'セリヌンティウスの職業は？', 'segment':2 },
    {'query': 'メロスが村に到着したのは、何時でしたか？', 'segment':3 },
    {'query': 'メロスの妹は、メロスの到着時に何をしていましたか？', 'segment':3 },
    {'query': '結婚式の日、どのような天候の変化が起こりましたか？', 'segment':3 },
    {'query': 'メロスは花婿にどのようなことを伝えましたか？', 'segment':4 },
    {'query': 'メロスはなぜ、祝宴の途中で眠りにつくことを決意しましたか？', 'segment':4 },
    {'query': 'メロスはどこで眠りにつきましたか？', 'segment':4 },
    {'query': 'メロスの進路に立ちはだかった障害は何でしたか？', 'segment':5 },
    {'query': 'メロスが川岸で何を祈ったのはなぜですか？', 'segment':5 },
    {'query': '橋が流されたのは何故ですか？', 'segment':5 },
]

test_case_list = [
    #セリヌンティウスは、すべてを察した様子で首肯《うなず》き、刑場一ぱいに鳴り響くほど音高くメロスの右頬を殴った
    { 
        'tokens': 1024,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'セリヌンティウスは、すべてを察した様子で首肯《うなず》き、刑場一ぱいに鳴り響くほど音高くメロスの右頬を殴った', 'segments':[174,175] },
        ]
    },
    { 
        'tokens': 1024,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'メロスを殴ったのは誰ですか？', 'segments':[174,175] },
            { 'text': 'メロスを殴ったのは誰', 'segments': [174,175] },
            { 'text': 'メロスを殴ったのは', 'segments': [174,175] },
        ]
    },
    { 
        'tokens': 128,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'メロスを殴ったのは誰ですか？', 'segments':[174,175] },
            { 'text': 'メロスを殴ったのは誰', 'segments': [174,175] },
            { 'text': 'メロスを殴ったのは', 'segments': [174,175] },
        ]
    },
    { 
        'tokens': 64,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'メロスを殴ったのは誰ですか？', 'segments':[174,175] },
            { 'text': 'メロスを殴ったのは誰', 'segments': [174,175] },
            { 'text': 'メロスを殴ったのは', 'segments': [174,175] },
        ]
    },
    { 
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'誰が彼を殴った', 'segments':[174,175] },
            { 'text':'誰がメロスを殴った', 'segments':[174,175] },
            { 'text':'彼がメロスを殴った', 'segments':[174,175] },
        ]
    },
    # メロスは腕に唸《うな》りをつけてセリヌンティウスの頬を殴った。
    { 
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'メロスは腕に唸《うな》りをつけてセリヌンティウスの頬を殴った。', 'segments':[177] },
        ]
    },
    { 
        'tokens': 1024,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'セリヌンティウスを殴ったのは誰ですか？', 'segments':[177] },
            { 'text': 'セリヌンティウスを生んだのは誰ですか？', 'segments': [] },
            { 'text': '焼き芋を食べたのは誰ですか？', 'segments': [] },
        ]
    },
    { 
        'tokens': 256,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'セリヌンティウスを殴ったのは誰ですか？', 'segments':[177] },
            { 'text': 'セリヌンティウスを生んだのは誰ですか？', 'segments': [] },
            { 'text': '焼き芋を食べたのは誰ですか？', 'segments': [] },
        ]
    },
    # segment 6
    #山賊たちは、ものも言わず一斉に棍棒《こんぼう》を振り挙げた。メロスはひょいと、からだを折り曲げ、飛鳥の如く身近かの一人に襲いかかり、その棍棒を奪い取って、
    #「気の毒だが正義のためだ！」と猛然一撃、たちまち、三人を殴り倒し、残る者のひるむ隙《すき》に、さっさと走って峠を下った。
    { 
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text':'山賊たちは、ものも言わず一斉に棍棒《こんぼう》を振り挙げた', 'segments':[106,107] },
            { 'text': '混棒を振り上げたのは誰？', 'segments': [106,107] },
            { 'text': '山賊たちは何を振り上げた？', 'segments': [106,107] },
        ]
    },
]

#query_markers
xxx = [
    { 'title': '●', 'marker': 'o' },
    { 'title': '■', 'marker': 's' },
    { 'title': '▲', 'marker': '^' },
    { 'title': '★', 'marker': '*' },
    { 'title': 'Ｘ', 'marker': 'x' },
]

def get_marker( aa:int ) :
    return xxx[ aa%len(xxx) ].get('marker','x')

def get_marker_title( aa:int ) :
    return xxx[ aa%len(xxx) ].get('title','x')

def main3():

    # テキストを分割
    with open( input_text_file, 'r', encoding='cp932') as file:
        text_data = file.read()
    
    segments_map = {
        64: split_text( text_data, 64 )
    }

    for i,txt in enumerate(segments_map.get(64)):
        print(f"# Segment:{i}")
        print(txt)
        print("\n")
    
    for case_no, test_case in enumerate(test_case_list):

        split_tokens:int = test_case.get('tokens',1024)
        segment_text_list = segments_map.get(split_tokens)
        if not segment_text_list:
            segment_text_list = segments_map[split_tokens] = split_text( text_data, split_tokens )
        xticks = [ (s*split_tokens)//64 for s in range(0,len(segment_text_list))]
        xticks = [ s for s in range(0,len(segment_text_list))]
        query_title = '\n'.join( [ f"{get_marker_title(qidx)}:{query.get('text')}" for qidx,query in enumerate(test_case.get('query',[]))] )

        plt.figure(figsize=(17, 9))
        plt.subplots_adjust(left=0.05)
        plt.suptitle( f'コサイン類似度の比較 Case{case_no} SegmentSize:{split_tokens}')
        plt.title( query_title,fontsize=10)
        plt.xlabel('Segment')
        plt.xticks( xticks )
        plt.ylabel('cosine_similarity')
        plt.ylim(0, 1)  # Keeping the y-axis range from 0 to 1
        plt.grid(True)

        for m,mm in enumerate( test_case.get('models',[]) ):
            plt.text( 0, 1.01+(0.03*m), f'{str(mm)}:{mm.color}', color=mm.color, fontsize=12)

        segments = [ (x*64)//split_tokens for query in test_case.get('query', []) for x in query.get('segments', [])]
        segments = sorted(set(segments))
        for x in segments:
            plt.axvline( x=x, color='r' )
    
        for fn in test_case.get('models',[]):
            segment_emb_list, tokens = fn( segment_text_list )
            for q,query in enumerate(test_case.get('query',[])):
                txt = query.get('text','')
                query_emb, tokens = fn( txt )
                sim_list = [ cosine_similarity(query_emb, seg_emb ) for seg_emb in segment_emb_list ]
                tt = '-' if query.get('segments') else '--'
                plt.plot(xticks,sim_list, label=f"Q:{q} {fn.simple_name()}", color=fn.color, linestyle=tt, marker=get_marker(q) )
                x=0
                y=sim_list[0]
                for i,s in enumerate(sim_list):
                    if s>y:
                        x=i
                        y=s
                plt.plot( x, y, 'o', ms=15, mfc='none', mec=fn.color, mew=1 )

        #plt.show()
        image_file_path = f'tmp/model_scores_plot_{case_no}.png'
        plt.savefig(image_file_path)

if __name__ == "__main__":
    main3()