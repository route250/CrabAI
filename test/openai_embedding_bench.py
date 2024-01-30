import sys, os, time, traceback, json
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
import numpy as np
import openai
import tiktoken
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

from test_data_download import download_and_extract_zip

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def tksplit(text: str, tokens: int = 1024) -> list[str]:
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

class ZerosEmbedding:
    def __init__(self,index):
        self.index=index
        self.embedding = [0] * 1536

class ZerosFunction:

    def __init__(self,model="zeros",dimensions=None):
        self.zeros = [ ZerosEmbedding(i) for i in range(0,100) ]
        self.client:OpenAI = OpenAI()
        self.model=model
        self.dimensions=dimensions

    def __str__(self):
        return self.model + ( "" if self.dimensions is None else "_"+str(self.dimensions) )

    def __call__(self, input: list[str]) -> list[float]:
        # input = [t.replace("\n", " ") for t in input]
        embeddings = self.zeros[0:len(input)]
        embeddings.sort( key=lambda e: e.index )
        return [result.embedding for result in embeddings]

class EmbFunction(ZerosFunction):

    def __init__(self,model:str):
        super().__init__(model)

    def __call__(self, input: list[str]) -> list[float]:
        # input = [t.replace("\n", " ") for t in input]
        embeddings = self.client.embeddings.create( input=input, model=self.model ).data
        embeddings.sort( key=lambda e: e.index )
        return [result.embedding for result in embeddings]
    
class EmbDimFunction(ZerosFunction):

    def __init__( self,model:str,dimensions:int ):
        super().__init__(model,dimensions)

    def __call__(self, input: list[str]) -> list[float]:
        # input = [t.replace("\n", " ") for t in input]
        embeddings = self.client.embeddings.create( input=input, model=self.model, dimensions=self.dimensions ).data
        embeddings.sort( key=lambda e: e.index )
        return [result.embedding for result in embeddings]

Function_list=[
    EmbFunction("text-embedding-ada-002"),
    EmbFunction("text-embedding-3-small"),
    EmbFunction("text-embedding-3-large"),
    EmbDimFunction("text-embedding-3-large",1536),
    EmbDimFunction("text-embedding-3-small",256),
    EmbDimFunction("text-embedding-3-large",256),
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

def main():
    seg_tokens=1024
    os.makedirs( 'tmp', exist_ok=True )
    file_segments='tmp/segments.txt'
    # テキストファイルを分割する
    if not os.path.exists(file_segments):
        with open( input_text_file, 'r', encoding='cp932') as file:
            text_data = file.read()
        segment_list = tksplit( text_data, seg_tokens )
        with open( file_segments, 'w' ) as file:
            json.dump( segment_list, file, ensure_ascii=False )
    else:
        with open( file_segments, 'r' ) as file:
            segment_list = json.load( file )
    # 分割したテキストからembeddingを計算する
    segment_embedding_list=[]
    query_embedding_list=[]
    for func in Function_list:
        filename=f"tmp/{str(func)}.json"
        if not os.path.exists(filename):
            seg_emb = func( segment_list )
            q_emb = func( [ q['query'] for q in Query_list ])
            with open( filename, 'w' ) as file:
                json.dump( {'segments':seg_emb, 'querys': q_emb }, file, ensure_ascii=False )
        else:
            with open( filename, 'r' ) as file:
                data = json.load(file)
                seg_emb=data['segments']
                q_emb=data['querys']
        segment_embedding_list.append(seg_emb)
        query_embedding_list.append(q_emb)
    #
    csvfile='tmp/cosine_similarity.csv'
    if not os.path.exists(csvfile):
        with open( csvfile, 'w' ) as file:
            line="queryNo,queryText,actual,model"
            for s1 in range(0,len(segment_list)):
                line+=f",s{s1}"
            file.write(line+"\n")
            for q,query in enumerate(Query_list):
                text=query['query']
                actual=query['segment']
                for f,func in enumerate(Function_list):
                    seg_emb = segment_embedding_list[f]
                    query_emb = query_embedding_list[f][q]
                    sim_list = [ cosine_similarity(query_emb, emb) for emb in seg_emb ]
                    line=f"{q},\"{text}\",{actual},\"{str(func)}\""
                    for s in sim_list:
                        line+=f",{s:.3f}"
                    file.write( line+"\n" )

    # Filtering the data for queryNo 0 to 2
    cosine_similarity_data = pd.read_csv(csvfile)
    query_0_data = cosine_similarity_data[cosine_similarity_data['queryNo'] == 0]
    # Extracting column names for scores (s0, s1, ..., s11)
    score_columns = query_0_data.columns[4:]
    # Specifying marker styles for each queryNo
    model_colors_list = [ 
        {
            "text-embedding-ada-002": "gray",
            "text-embedding-3-small": "red",
            "text-embedding-3-large": "blue"
        },
        {
            "text-embedding-3-small": "gray",
            "text-embedding-3-small_256": "red",
            "text-embedding-3-large_256": "blue"
        },
    ]
    query_markers = {
        0: "o",  # Circle marker
        1: "s",  # Square marker
        2: "^"   # Triangle marker
    }

    qlist_list = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17]]
    for plotNo, queryIndexList in enumerate(qlist_list):
        query_0_to_2_data = cosine_similarity_data[cosine_similarity_data['queryNo'].isin(queryIndexList)]
        # Creating a line plot with specified colors and markers for the selected models and queryNos
        for mm,model_colors in enumerate(model_colors_list):
            plt.figure(figsize=(17, 9))
            plt.subplots_adjust(left=0.05)
            title_list=[]
            for query_no in queryIndexList:
                queryText = Query_list[query_no]['query']
                title_list.append( f"No{query_no} {queryText}" )
                mk = query_no % 3
                for model in model_colors:
                    lbl = model.replace('text-embedding-','')
                    model_data = query_0_to_2_data[(query_0_to_2_data['queryNo'] == query_no) & (query_0_to_2_data['model'] == model)]
                    if not model_data.empty:
                        plt.plot(score_columns, model_data.iloc[0, 4:], label=f"{lbl} No{query_no}", 
                                color=model_colors[model], marker=query_markers[mk])
            plt.suptitle('モデル別のコサイン類似度')
            plt.title( "\n".join(title_list),fontsize=10)
            plt.xlabel('Segment')
            plt.ylabel('cosine_similarity')
            plt.ylim(0, 1)  # Keeping the y-axis range from 0 to 1
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Moving the legend outside the plot
            plt.grid(True)
            #plt.show()
            image_file_path = f'tmp/model_scores_plot{plotNo}_{mm}.png'
            plt.savefig(image_file_path)


def main2():
    seg_tokens=1024
    with open( input_text_file, 'r', encoding='cp932') as file:
        text_data = file.read()
    print( f"test: {len(text_data)}(chars)")
    segments = tksplit( text_data, seg_tokens )
    print( f"segments:{len(segments)}")
    for i,seg in enumerate(segments):
        print(f"  {i:3d} {len(seg):6d}(chars)")

    batchs=[1,2,4,8,11,15,21,29,40]
    loop=5
    print()
    csvdata=[]
    csvdata.append("model,tokens/segment,batch,loop,totaltime(sec),time(sec)/request\n")
    for batch in batchs:
        inputs=[ segments[i%(len(segments))] for i in range(0,batch)]
        for func in Function_list:
            title = str(func)
            st=time.time()
            for k in range(0,loop):
                emb = func( inputs )
            et=time.time()
            total_time=et-st
            timePerReq = total_time/loop
            print(f"{title:20s},{batch:2d},{loop:2d},{total_time:8.3f}(sec)")
            csvdata.append(f"\"{title}\",{seg_tokens},{batch},{loop},{total_time:.3f},{timePerReq:.3f}\n")
    with open('tmp/result.csv','w') as file:
        file.writelines(csvdata)

if __name__ == "__main__":
    main()
