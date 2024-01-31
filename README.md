# CrabAI🦀

カニ食べ放題に反対活動するカニ

OpenAIのAPIを使ったチャットボットシステムです。

## 🦀Requirements

|ライブラリ|要求バージョン|開発時バージョン|コメント|
|---|---|---|---|
|python|>= 3.10|3.11|おそらく3.9でも動くと思うが未確認|
|streamlit|>=1.24.0|1.30.0|st.chat_messageが1.24以降だが1.30でしか確認していない|
|openai|>=1.10|1.10|2024-01-25に発表されたtext-embedding-3-smallのために1.10以降が必要|
|tiktoken||0.5.2|依存性がよくわからないので要求バージョン不明|
|chromadb||0.4.22|依存性がよくわからないので要求バージョン不明|
|pysqlite3-binary||0.5.2.post3|sqlite3を使う場合は不要|
|sqlite3|>=3.35.0||pysqlite3-binaryを使う場合は不要|

## 🦀実行方法

Linux(ubuntu)では、添付のrun.shで起動できます。このスクリプトで後述の環境構築も自動で行います。

Windows環境の場合は、run.batで起動できます。環境は手動で構築して下さい。

streamlit標準の8501ポートで起動しますので、ブラウザで、http://127.0.0.1:8501を開いて下さい。

他のPCからアクセスする場合は、ファイアウォールの設定をお忘れなく。

## 🦀環境構築

実行に必要な環境は以下のように構築して下さい。run.shを実行したら自動で実施されます。

```bash:環境構築
python3 -m venv .venv --prompt CrabAI
source .venv/bin/activate
python3 -m pip install -U pip
pip install streamlit openai tiktoken chromadb
```

おそらく、pysqlite3-binaryも必要になります。
linuxの場合、OSにインストールされているsqlite3のバージョンが古い場合、以下のようなエラーメッセージが表示されます。

```text:エラーメッセージ
File "~/venv3.10/lib/python3.10/site-packages/chromadb/__init__.py", line 36, in <module> raise RuntimeError( RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
lease visit https://docs.trychroma.com/troubleshooting#sqlite to learn how to upgrade. )
```

chromadbのTroubleshootingページのSQLiteの項目を参照して下さい。

- [https://docs.trychroma.com/troubleshooting](https://docs.trychroma.com/troubleshooting)
- [https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300](https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300)

私の場合は、chromadb/__init__.pyを修正して対応しました。エラーをraiseしている箇所をコメントアウトして、代わりにpysqlite3を使うように設定を追加しています。

```python:chromadb/__init__.pyの修正箇所
            import sys
            __import__("pysqlite3")
            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            #raise RuntimeError(
            #    "\033[91mYour system has an unsupported version of sqlite3. Chroma \
            #        requires sqlite3 >= 3.35.0.\033[0m\n"
            #    "\033[94mPlease visit \
            #        https://docs.trychroma.com/troubleshooting#sqlite to learn how \
            #        to upgrade.\033[0m"
            #)
```
