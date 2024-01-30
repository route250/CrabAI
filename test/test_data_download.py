


import os
import requests
import zipfile
from urllib.parse import urlparse

# 青空文庫
FILEMAP={
    'hashire_merosu.txt': "https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip",
    'gingatetsudono_yoru.txt': "https://www.aozora.gr.jp/cards/000081/files/456_ruby_145.zip",
    'gakumonno_susume.txt': "https://www.aozora.gr.jp/cards/000296/files/47061_ruby_28378.zip",
    'bocchan.txt': "https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip",
}

def detect_encode( filename ):
    encs=['utf-8','cp932','utf-16']
    for enc in encs:
        try:
            with open(filename,'r',encoding=enc) as inp:
                chunk = inp.read(1000)
                return enc
        except UnicodeDecodeError:
            continue
        except:
            pass
    return None

def tail_content(content):
    try:
        p=content.index('底本：')
        if p>0:
            print( content[p:] )
    except:
        pass

def tail_filepath(filepath):
    try:
        with open(filepath,'r', encoding=detect_encode(filepath) ) as fp:
            content = fp.read()
        tail_content(content)
    except:
        pass
    return filepath

def download_and_extract_zip( targetfile, download_dir='./tmp/download', aozora_dir='./tmp/aozora_bunko'):
    url = FILEMAP.get(targetfile)
    if not url:
        return None
    targetpath = os.path.join(aozora_dir,targetfile)
    if os.path.exists( targetpath ):
        return tail_filepath(targetpath)
    # ディレクトリが存在しない場合は作成
    os.makedirs(download_dir,exist_ok=True)
    os.makedirs(aozora_dir,exist_ok=True)
    # URLからファイル名（basename）を抽出
    zip_filename = os.path.basename(urlparse(url).path)
    zip_path = os.path.join(download_dir, zip_filename)
    # ZIPファイルのダウンロード
    response = requests.get(url)
    with open(zip_path, 'wb') as entry:
        entry.write(response.content)
    # ZIPファイルの展開
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # ZIP内の各ファイルに対して
        for entry in zip_ref.namelist():
            # テキストファイルのみを対象にする
            if entry==targetfile:
                zip_ref.extract(entry, aozora_dir)
                with open(targetpath,'r', encoding=detect_encode(targetpath) ) as fp:
                    content = fp.read()
                content = content.replace('\r\n','\n')
                with open(targetpath, 'w') as fp:
                    fp.write(content)
                tail_content(content)
                return targetpath
    return None

def main():
    # この関数はURLを引数として取りますが、今はダミーのURLを使用します
    filepath = download_and_extract_zip('hashire_merosu.txt')
    print(f"download:{filepath}")

if __name__ == "__main__":
    main()