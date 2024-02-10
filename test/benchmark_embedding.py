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

from crab.embeddings import EmbeddingFunction, create_embeddings, squared_l2_distance, inner_product, cosine_similarity, split_text

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

emb_models:list[EmbeddingFunc] = [fn_3small, fn_3small_256, fn_3large, fn_3large_1536, fn_3large_256, fn_ada002]

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

txt_64_73:str = """メロスも、満面に喜色を湛《たた》え、しばらくは、王とのあの約束をさえ忘れていた。祝宴は、夜に入っていよいよ乱れ華やかになり、人々は、外の豪雨を全く気にしなくなった。メロスは、一生このままここにいたい、と思った。この佳い人たちと生涯暮して行きたいと願ったが、いまは、自分のからだで、自分のものでは無い。ままならぬ事である。メロスは、わが身に鞭打ち、ついに出発を決意した。あすの日没までには、まだ十分の時が在る。ちょっと一眠りして、それからすぐに出発しよう、と考えた。その頃には、雨も小降りになっていよう。少しでも永くこの家に愚図愚図とどまっていたかった。メロスほどの男にも、やはり未練の情というものは在る。今宵呆然、歓喜に酔っているらしい花嫁に近寄り、
「おめでとう。私は疲れてしまったから、ちょっとご免こうむって眠りたい。眼が覚めたら、すぐに市に出かける。大切な用事があるのだ。私がいなくても、もうおまえには優しい亭主があるのだから、決して寂しい事は無い。おまえの兄の、一ばんきらいなものは、人を疑う事と、それから、嘘をつく事だ。おまえも、それは、知っているね。亭主との間に、どんな秘密でも作ってはならぬ。おまえに言いたいのは、それだけだ。おまえの兄は、たぶん偉い男なのだから、おまえもその誇りを持っていろ。」
"""
txt_66_71:str = """メロスは、一生このままここにいたい、と思った。この佳い人たちと生涯暮して行きたいと願ったが、いまは、自分のからだで、自分のものでは無い。ままならぬ事である。メロスは、わが身に鞭打ち、ついに出発を決意した。あすの日没までには、まだ十分の時が在る。ちょっと一眠りして、それからすぐに出発しよう、と考えた。その頃には、雨も小降りになっていよう。少しでも永くこの家に愚図愚図とどまっていたかった。メロスほどの男にも、やはり未練の情というものは在る。今宵呆然、歓喜に酔っているらしい花嫁に近寄り、「おめでとう。私は疲れてしまったから、ちょっとご免こうむって眠りたい。眼が覚めたら、すぐに市に出かける。"""

txt_68_69:str = """ちょっと一眠りして、それからすぐに出発しよう、と考えた。その頃には、雨も小降りになっていよう。少しでも永くこの家に愚図愚図とどまっていたかった。メロスほどの男にも、やはり未練の情というものは在る。"""
txt01_nomal:str = """メロスも、顔いっぱいに喜びをあふれさせて、しばらくは王との約束をすっかり忘れていました。祝宴は夜になるとさらに賑やかになり、人々は外の激しい雨にも全く気を留めなくなりました。メロスは、一生このままここにいたいと思いました。この素晴らしい人たちと一生を共に過ごしたいと願ったけれど、今は自分の体が自分のものではありません。どうすることもできません。メロスは自分を奮い立たせ、ついに出発する決意をしました。明日の日没までには、まだたくさん時間があります。少し眠ってから、すぐに出発しようと思いました。その頃には、雨も小降りになっているでしょう。できるだけ長くこの家にいたかったです。メロスも、やはり未練があるのです。今宵、喜びに酔いしれている花嫁に近づき、
「おめでとう。私は疲れてしまったから、少し休ませてもらって眠りたい。目が覚めたら、すぐに町へ出かける。大切な用があるんだ。私がいなくても、あなたには優しい夫がいるから、寂しくなんかないよ。あなたの兄が一番嫌うのは、人を疑うことと、嘘をつくことだ。あなたもそれは知ってるよね。夫との間に、どんな秘密も作ってはいけない。言いたいのは、それだけだ。あなたの兄は、きっと素晴らしい人なんだから、あなたもその誇りを持ってね。」
花嫁は、夢見るようにうなずいた。メロスは、それから花婿の肩を叩いて、
「持ち物がないのはお互い様だ。私の家には、宝物と言ったら、妹と羊だけだよ。他には何もない。全部あげる。もう一つ、メロスの弟になったことを誇りに思ってね。」"""
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
        ],
        'functions': [cosine_similarity, inner_product, squared_l2_distance ]
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
    {
        'tokens': 1024,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text': '羊小屋にもぐり込んで、死んだように深く眠った。眼が覚めたのは翌る日の薄明の頃である。メロスは跳ね起き、南無三、寝過したか、いや、まだまだ大丈夫、これからすぐに出発すれば、約束の刻限までには十分間に合う。きょうは是非とも、あの王に、人の信実の存するところを見せてやろう。そうして笑って磔の台に上ってやる。メロスは、悠々と身仕度をはじめた。', 'segments':[76,77,78,79]},
            { 'text': 'メロスはどこで寝たの？', 'segments':[76,77]},
            { 'text': 'メロスは何時に起きたの？', 'segments':[77]},
        ]
    },
    {
        'tokens': 512,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text': '羊小屋にもぐり込んで、死んだように深く眠った。眼が覚めたのは翌る日の薄明の頃である。メロスは跳ね起き、南無三、寝過したか、いや、まだまだ大丈夫、これからすぐに出発すれば、約束の刻限までには十分間に合う。きょうは是非とも、あの王に、人の信実の存するところを見せてやろう。そうして笑って磔の台に上ってやる。メロスは、悠々と身仕度をはじめた。', 'segments':[76,77,78,79]},
            { 'text': 'メロスはどこで寝たの？', 'segments':[76,77]},
            { 'text': 'メロスは何時に起きたの？', 'segments':[77]},
        ]
    },
    {
        'tokens': 256,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text': '羊小屋にもぐり込んで、死んだように深く眠った。眼が覚めたのは翌る日の薄明の頃である。メロスは跳ね起き、南無三、寝過したか、いや、まだまだ大丈夫、これからすぐに出発すれば、約束の刻限までには十分間に合う。きょうは是非とも、あの王に、人の信実の存するところを見せてやろう。そうして笑って磔の台に上ってやる。メロスは、悠々と身仕度をはじめた。', 'segments':[76,77,78,79]},
            { 'text': 'メロスはどこで寝たの？', 'segments':[76,77]},
            { 'text': 'メロスは何時に起きたの？', 'segments':[77]},
        ]
    },
    {
        'tokens': 64,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text': '羊小屋にもぐり込んで、死んだように深く眠った。眼が覚めたのは翌る日の薄明の頃である。メロスは跳ね起き、南無三、寝過したか、いや、まだまだ大丈夫、これからすぐに出発すれば、約束の刻限までには十分間に合う。きょうは是非とも、あの王に、人の信実の存するところを見せてやろう。そうして笑って磔の台に上ってやる。メロスは、悠々と身仕度をはじめた。', 'segments':[76,77,78,79]},
            { 'text': 'メロスはどこで寝たの？', 'segments':[76,77]},
            { 'text': 'メロスは何時に起きたの？', 'segments':[77]},
        ]
    },
    {
        'tokens': 1024,
        'models': [fn_3small,fn_3large,fn_ada002],
        'query': [
            { 'text': txt_64_73, 'segments':[64,65,66,67,68,69,70,71,72,73]},
            { 'text': txt_66_71, 'segments':[66,67,68,69,70,71]},
            { 'text': txt_68_69, 'segments':[68,69]},
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

def text_trim( text:str, maxlen:int=70 ) ->str:
    text=text.replace('\n','').strip()
    ll = len(text)
    if ll<=maxlen:
        return text
    midstr=f' ...(省略)... '
    a = (maxlen-len(midstr))//2
    b = maxlen-len(midstr)-a
    return text[:a]+midstr+text[-b:]

def main():

    dpi:int = 126
    width_in: int = 16
    height_in:int = 9
    # テキストをロード
    with open( input_text_file, 'r', encoding='cp932') as file:
        text_data = file.read()
    # 分割キャッシュ
    split_cache = {}
    case_no:int = 0
    total_tks:int = 0
    for test_case in test_case_list:
        tks = sum( fn.tokens for fn in emb_models )
        test_case_query_list = test_case.get('query') or []
        test_case_model_list = test_case.get('models') or []
        if not test_case_model_list or not test_case_query_list:
            continue

        split_tokens:int = test_case.get('tokens',1024)
        segment_text_list = split_cache.get(split_tokens)
        if not segment_text_list:
            segment_text_list = split_cache[split_tokens] = split_text( text_data, split_tokens )
            # 分割結果をダンプ
            basename_without_ext = os.path.splitext(os.path.basename(input_text_file))[0]
            segments_dump_file = os.path.join( "tmp", f"{basename_without_ext}_split_{split_tokens}.txt" )
            with open(segments_dump_file, 'w', encoding='utf-8' ) as out:
                for i,query_text in enumerate( segment_text_list or [] ):
                    out.write(f"# Segment {i}\n")
                    out.write(query_text)
                    out.write("\n\n")

        xticks = [ s for s in range(0,len(segment_text_list))]

        fn_sim_list = test_case.get('functions') or cosine_similarity #CrabSimFunc()
        fn_sim_list = fn_sim_list if isinstance(fn_sim_list,list) else [fn_sim_list]

        for fn_sim in fn_sim_list:
            case_no += 1
            ylabel = fn_sim.__name__
            print(f"start case:{case_no} {ylabel}")

            is_not_distance = fn_sim != squared_l2_distance

            plt.figure(figsize=(width_in, height_in),dpi=dpi)
            plt.suptitle( f' Case{case_no} {ylabel}の比較')
            plt.xlabel('Segment')
            plt.xticks( xticks )
            plt.ylabel( ylabel )
            plt.grid(True)

            xpos2 = len(segment_text_list)*-0.03
            ybase = 1.01 if is_not_distance else -0.02
            yheight = 0.03 if is_not_distance else -0.06
            for qidx,query in enumerate(test_case_query_list):
                ypos = ybase + yheight*( len(test_case_query_list)-qidx-1)
                query_text = text_trim( text_filter(query.get('text')))
                plt.text( xpos2,ypos, f"{get_marker_title(qidx)}:{query_text}", fontsize=12 )

            xpos2 = len(segment_text_list)*0.8
            for midx,mm in enumerate( test_case_model_list ):
                ypos = ybase + yheight*( len(test_case_model_list)-midx-1)
                plt.text( xpos2, ypos, f'{str(mm)}', color=mm.color, fontsize=12)
            ypos = ybase + yheight*( len(test_case_model_list)-(-1)-1)
            plt.text( xpos2, ypos, f"分割サイズ:{split_tokens}", fontsize=12)

            segments = [ (x*64)//split_tokens for query in test_case_query_list for x in query.get('segments', [])]
            segments = sorted(set(segments))
            for x in segments:
                plt.axvline( x=x, color='r' )
        
            ms = 6 if len(segment_text_list)<40 else 3

            global_min = None
            global_max = None
            for fn in test_case_model_list:
                segment_emb_list = fn( text_filter(segment_text_list) )
                for q,query in enumerate(test_case_query_list):
                    query_text = text_filter( query.get('text','') )
                    query_emb = fn( query_text )
                    sim_list = [ fn_sim(query_emb, seg_emb ) for seg_emb in segment_emb_list ]
                    ymin = min(sim_list)
                    global_min = ymin if global_min is None or ymin<global_min else global_min
                    ymax = max(sim_list)
                    global_max = ymax if global_max is None or ymax>global_max else global_max
                    cl = fn.color if query.get('segments') else fn.color2
                    tt = '-' if query.get('segments') else '--'
                    lw = 0.5 #if query.get('segments') else 0.5
                    plt.plot(xticks,sim_list, label=f"Q:{q} {fn.simple_name()}", color=cl, linestyle=tt, marker=get_marker(q),  markersize=ms, linewidth=lw )
                    mark_y = ymax if fn_sim != squared_l2_distance else ymin
                    mark_x = sim_list.index(mark_y)
                    sw = fn.color if query.get('segments') else 'gray'
                    plt.plot( mark_x, mark_y, 'o', ms=18, mfc='none', mec=cl, mew=lw )
            if is_not_distance:
                plt.ylim(0, 1) 
            else:
                plt.ylim( int(global_max+1),0)
            plt.subplots_adjust( left=0.05, right=0.95 )
            #plt.tight_layout()  # 自動的に空白を最適化
            #plt.show()
            for ext in ['png','svg']:
                image_file_path = os.path.join( "tmp", ext, f'embedding_scores_plot_{case_no}_{ylabel}.{ext}' )
                os.makedirs( os.path.dirname(image_file_path),exist_ok=True)
                plt.savefig(image_file_path)

            tks = sum( fn.tokens for fn in emb_models ) - tks
            total_tks += tks
            print( f"    tokens: {tks} {total_tks}")

if __name__ == "__main__":
    main()