import sys, os, time
import re
import unittest
from openai import OpenAI, OpenAIError
import tiktoken
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import japanize_matplotlib
#from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
from test_data_download import download_and_extract_zip

from crab.embeddings import EmbeddingFunction, create_embeddings, cosine_similarity, split_text

def adjust_saturation(color_name, saturation_level):
    """
    Adjust the saturation of a given color name to a specified level.
    
    Parameters:
    - color_name: The name of the color (e.g., 'red').
    - saturation_level: The desired saturation level (0 to 1).
    
    Returns:
    - A tuple representing the RGB values of the adjusted color.
    """
    # Convert color name to RGB
    original_rgb = mcolors.to_rgb(color_name)
    # Convert RGB to HSV
    original_hsv = mcolors.rgb_to_hsv(original_rgb)
    # Adjust the saturation
    adjusted_hsv = original_hsv.copy()
    adjusted_hsv[1] = saturation_level
    # Convert back to RGB
    adjusted_rgb = mcolors.hsv_to_rgb(adjusted_hsv)
    return adjusted_rgb

# Example usage: adjust the saturation of 'red' to 70%
adjusted_color = adjust_saturation('red', 0.7)
adjusted_color

def text_filter( input ):
    input_list = input if isinstance(input,list) else [input]
    result_list = [ re.sub( r'《[^》]*》', '', txt ) if isinstance(txt,str) else txt for txt in input_list ]
    return result_list if isinstance(input,list) else result_list[0]

class EmbeddingFunc(EmbeddingFunction):
    def __init__(self, client, model, dim=None, *, color='r', color2=None ):
        super().__init__( client, model=model, dimensions=dim )
        self.color = color
        self.color2 = color2 if color2 else adjust_saturation(color,0.4)

client:OpenAI = OpenAI()
fn_3small = EmbeddingFunc(client,"text-embedding-3-small", color='blue')
fn_3small_256 = EmbeddingFunc(client,"text-embedding-3-small",256, color='aqua')
fn_3large = EmbeddingFunc(client,"text-embedding-3-large", color='crimson')
fn_3large_1536 = EmbeddingFunc(client,"text-embedding-3-large",1536, color='tomato')
fn_3large_256 = EmbeddingFunc(client,"text-embedding-3-large",256, color='brown')
fn_ada002 = EmbeddingFunc(client,"text-embedding-ada-002", color='green')

class CrabSimFunc:
    def __str__(self)-> str:
        return "crab cosine_similarity"
    def __call__(self, v1, v2 ) ->float:
        return cosine_similarity(v1,v2)

# class SklearnSimFunc:
#     def __str__(self)-> str:
#         return "sklearn cosine_similarity"
#     def __call__(self, v1, v2 ) ->float:
#         return sklearn_cosine_similarity( [v1],[v2] )[0]

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
line_marker_list = [
    { 'title': '●', 'marker': 'o' },
    { 'title': '■', 'marker': 's' },
    { 'title': '▲', 'marker': '^' },
    { 'title': '★', 'marker': '*' },
    { 'title': 'Ｘ', 'marker': 'x' },
]

def get_marker( index:int ) :
    return line_marker_list[ index%len(line_marker_list) ].get('marker','x')

def get_marker_title( index:int ) :
    return line_marker_list[ index%len(line_marker_list) ].get('title','x')

def main():

    # テキストをロード
    with open( input_text_file, 'r', encoding='cp932') as file:
        text_data = file.read()
    # 分割キャッシュ
    split_cache = {}
   
    for case_no, test_case in enumerate(test_case_list):

        split_tokens:int = test_case.get('tokens',1024)
        segment_text_list = split_cache.get(split_tokens)
        if not segment_text_list:
            segment_text_list = split_cache[split_tokens] = split_text( text_data, split_tokens )
            # 分割結果をダンプ
            basename_without_ext = os.path.splitext(os.path.basename(input_text_file))[0]
            segments_dump_file = os.path.join( "tmp", f"{basename_without_ext}_split_{split_tokens}.txt" )
            with open(segments_dump_file, 'w', encoding='utf-8' ) as out:
                for i,txt in enumerate( segment_text_list or [] ):
                    out.write(f"# Segment {i}\n")
                    out.write(txt)
                    out.write("\n\n")

        xticks = [ s for s in range(0,len(segment_text_list))]
        query_title = '\n'.join( [ f"{get_marker_title(qidx)}:{ text_filter(query.get('text'))}" for qidx,query in enumerate(test_case.get('query',[]))] )

        #fn_sim = SklearnSimFunc()
        fn_sim = CrabSimFunc()
        plt.figure(figsize=(17, 9))
        plt.suptitle( f'コサイン類似度の比較 Case{case_no} SegmentSize:{split_tokens}')
        plt.title( query_title,fontsize=10)
        plt.xlabel('Segment')
        plt.xticks( xticks )
        plt.ylabel( str(fn_sim) )
        plt.ylim(0, 1)  # Keeping the y-axis range from 0 to 1
        plt.grid(True)

        for m,mm in enumerate( test_case.get('models',[]) ):
            plt.text( 0, 1.01+(0.03*m), f'{str(mm)}:{mm.color}', color=mm.color, fontsize=12)

        segments = [ (x*64)//split_tokens for query in test_case.get('query', []) for x in query.get('segments', [])]
        segments = sorted(set(segments))
        for x in segments:
            plt.axvline( x=x, color='r' )
    
        for fn in test_case.get('models',[]):
            segment_emb_list, tokens = fn( text_filter(segment_text_list) )
            for q,query in enumerate(test_case.get('query',[])):
                txt = text_filter( query.get('text','') )
                query_emb, tokens = fn( txt )
                sim_list = [ fn_sim(query_emb, seg_emb ) for seg_emb in segment_emb_list ]
                cl = fn.color if query.get('segments') else fn.color2
                tt = '-' if query.get('segments') else '--'
                lw = 1 #if query.get('segments') else 0.5
                plt.plot(xticks,sim_list, label=f"Q:{q} {fn.simple_name()}", color=cl, linestyle=tt, marker=get_marker(q),  linewidth=lw )
                x=0
                y=sim_list[0]
                for i,s in enumerate(sim_list):
                    if s>y:
                        x=i
                        y=s
                sw = fn.color if query.get('segments') else 'gray'
                plt.plot( x, y, 'o', ms=18, mfc='none', mec=cl, mew=lw )

        #plt.show()
        image_file_path = os.path.join( "tmp", f'model_scores_plot_{case_no}.png' )
        plt.savefig(image_file_path)

if __name__ == "__main__":
    main()