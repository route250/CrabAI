# CrabAI🦀

カニ食べ放題に反対活動するカニ

CrabAIは、OpenAIのAPIを使用したチャットボットシステムです。GPTsに似た機能を備えています。

## 🦀Features

- 目的に応じてチャットボットを作成することができます。モデルとプロンプトでできます。
- ログインユーザ毎に、自分のチャットボットを作成できます。チャットボットは自分専用、もしくは、他のユーザに公開できます。
- 複数のチャットボットを定義して、それぞれにプロンプトやモデルなどを設定できます。
- RAG (Retrieval-Augmented Generation) により、チャットボットに追加情報を与えることができます。
- OpenAIのapikeyをユーザ間で共有、または、個別のキーを使用することができます。
- カニ食べ放題に反対します。


## 🦀Requirements

- OS: Ubuntsu 22.04 LTS, Rocky8(RHEL8), Windows10/11
- Python v3.10以降

  |ライブラリ|要求バージョン|開発時バージョン|コメント|
  |---|---|---|---|
  |streamlit|>=1.24.0|1.31.0|st.chat_messageが1.24以降だが1.31でしか確認していない|
  |openai|>=1.10|1.10|2024-01-25に発表されたtext-embedding-3-smallのために1.10以降が必要|
  |tiktoken||0.5.2|依存性がよくわからないので要求バージョン不明|
  |chromadb||0.4.22|依存性がよくわからないので要求バージョン不明|
  |pysqlite3-binary||0.5.2.post3|sqlite3を使う場合は不要|
  |sqlite3|>=3.35.0||pysqlite3-binaryを使う場合は不要|
  |cryptography||42.0.2|依存性がよくわからないので要求バージョン不明|


## 🦀実行方法

Linux(ubuntu)では、添付のrun.shで起動できます。このスクリプトで後述の環境構築も自動で行います。

Windows環境の場合は、run.batで起動できます。環境は手動で構築して下さい。

streamlit標準の8501ポートで起動しますので、ブラウザで、http://127.0.0.1:8501を開いて下さい。

他のPCからアクセスする場合は、ファイアウォールの設定をお忘れなく。

vscodeで起動する場合は、launch.jsonを設定してあるので、vscode画面の左端のツールバーから「実行とデバッグ」のアイコンをクリックして、"Streamlit crabai-st"を選択してF5キーで起動できるはずです。

## 🦀環境構築

実行に必要な環境は以下のように構築して下さい。run.shを実行したら自動で実施されます。

```bash:環境構築
python3 -m venv .venv --prompt CrabAI
source .venv/bin/activate
python3 -m pip install -U pip
pip install streamlit openai tiktoken chromadb cryptography
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

## 🦀Usage

### 初期設定
起動したら、ログイン画面が表示されます。初期状態では、rootユーザしかありませんので、rootユーザでログインして下さい。

rootでログインしたら、まずはサイドバーの一番上の「User:root」ボタンをクリックしてユーザ編集画面を開いて下さい。
rootユーザのパスワードと、openaiのapi-keyを入力し、share keyをチェックして「Save」ボタンで保存して下さい。

他のユーザを作成する場合は、サイドバーの一番下の「create user」をクリックして下さい。

### ユーザ編集画面の項目
|項目|説明|
|---|---|
|Name|ログインユーザ名|
|Passwd|パスワード|
|E-mail|メールアドレス。現仕様では使用しません|
|OpenAI api-key|openaiのapi-keyを設定して下さい。share Keyの項目も参照のこと|
|Shre key|api-keyをユーザ間で共有する設定です。rootユーザの場合、他のユーザにキーを使用させるかどうかの設定になります。他のユーザの場合、rootユーザのapi-keyを使用するかどうかの設定になります。
|Description|メモ欄。特に使用しません

### api-keyの優先順位

- rootユーザの場合

  1. rootユーザに設定されたキー
  2. 環境変数OPENAI_API_KEYに設定されたキー

- 他のユーザの場合

  1. ユーザに設定されたキー
  2. rootユーザに設定されたキー(ただしshare keyの設定による)
  3. 環境変数OPENAI_API_KEYに設定されたキー

### Botの作成・編集

初期状態では、CrabBotというBotが設定されています。こいつは、カニ食べ放題しか言わないので、別途Botを作成して下さい。

サイドバーのCrabsのところの「create」ボタンをクリックすると新規作成できます。また、Botの設定を変更するときは、Bot名の横の「...」ボタンをクリックして下さい。

### Bot編集画面の項目

|項目|説明|
|---|---|
|Name|Botの名前
|Owner|作成者(変更不可)|
|auth|自分専用ならprivate 他のユーザに公開するならpublic|
|Description|メモ欄|
|Model|OpenAIのモデルを選択|
|max_tokens|リクエストで送受信する最大トークン数。トークン数についてを参照|
|input_tokens|リクエストで送信する最大トークン数。トークン数についてを参照|
|temperture|OpenAIのAPIのtempertureパラメータ|
|LLM|OpenAIのAPIをコールするかどうか。チェックを外すと、RetriveとRAGだけを実行する。|
|Retrive|チェックをOnにすると、会話ログからEmbedding検索でRetriveする|
|RAG|チェックをOnにすると、Botに登録されているファイルからEmbedding検索でRetriveする|
|prompt|OpenAI LLMのプロンプトです。|
|select file|Botにテキストファイルを添付することができます。D&Dするか「Browse files」をクリックしてファイルを選択して下さい。streamlitの仕様により、いまいち使いにくいのですが。アップロードしたファイルが表示されない時は、一度Bot編集画面を閉じて開き直して見て下さい。|

### トークン数について

リクエストを生成する時、Botに設定されたinput_tokensを超えないように処理します。(たまに超えますが)

リクエストで送信するデータは以下のものが含まれます。
  1. プロンプト
  2. 過去ログからのRetrive (RetriveがOnの場合)
  3. ファイルからのRetrive (RAGがOnの場合)
  4. 直近の会話履歴

これらのデータのトークン数をtiktokenでカウントして調整します。
まず、input_tokensからプロンプトトークン数を引いた残りを半分づつRetriveと会話履歴に振り分けます。

Retreveトークン数 = ( input_tokens - プロンプトのトークン数 ) // 2
会話履歴トークン数 = ( input_tokens - プロンプトのトークン数 ) // 2

過去ログとファイルを検索した結果を、Retriveトークン数の範囲に収まるように調整します。

会話履歴から最新１０件を会話履歴トークン数の範囲に収めます。

処理した結果の入力トークン数合計をBotに設定されたmax_tokensから引いたものが最大出力トークン数となります。

最大出力トークン数 = max_tokens - 入力トークン数の合計

OpenAIへのリクエストのmax_tokensパラメータに最大出力トークン数を設定します。Botに設定されたmax_tokensとOpenAI APIへのmax_tokensは意味が異なることに注意して下さい。

## 🦀データ保存先

ユーザ設定、Bot設定、会話履歴は、すべてChromaDBのデータとして保存されます。保存先ディレクトリは、$HOME/.data/crabAIです。(Windowsの場合はC:\Users\ユーザ名\.data\crabAI)