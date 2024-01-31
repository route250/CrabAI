#!/bin/bash

START_FILE=./src/crabai-st.py
SCRDIR=$(cd $(dirname $0);pwd)
ST_OPT='--server.headless true --browser.gatherUsageStats false'

# 重複起動防止
RUNDIR=/tmp/crabAI
LOCKDIR=$RUNDIR/.lock
mkdir -p $RUNDIR
if ! mkdir $LOCKDIR >/dev/null 2>&1; then
    exit 9
fi
function cleanup() {
    echo ""
    echo "End `date`"
    echo ""
    rmdir $LOCKDIR
}
trap cleanup EXIT

exec >>$RUNDIR/crab.log 2>&1
echo ""
echo "Start `date`"
echo ""
cd $SCRDIR
if [ ! -r $START_FILE ]; then
    echo "Can't found $START_FILE"
    exit 7
fi

# 仮想環境から抜ける
if type deactivate>/dev/null 2>&1; then
    deactivate
fi

# 仮想環境を作る
if [ ! -d .venv ]; then
    echo "# Create venv"
    python3 -m venv .venv --prompt CrabAI
fi
if [ ! -f .venv/bin/activate ]; then
    echo "# can not found activate"
    exit 1
fi

# 仮想環境へ切り替え
source .venv/bin/activate
if type deactivate>/dev/null 2>&1; then
    :
else
    echo "# can not found activate"
    exit 1
fi

# 必要なパッケージをインストールする
python3 -m pip install -U pip
P=$(pip list | awk '$1=="openai"||$1=="streamlit"||$1=="chromadb"||$1=="tiktoken" {print $1}'|wc -l)
if [ "$P" != "4" ]; then
  pip install streamlit openai tiktoken chromadb pysqlite3-binary cryptography
fi
P=$(pip list | awk '$1=="openai"||$1=="streamlit"||$1=="chromadb"||$1=="tiktoken" {print $1}'|wc -l)
if [ "$P" != "4" ]; then
    echo "# can not install package"
    exit 1
fi

# chromadbにパッチを当てる
for file in .venv/lib/*python*/site-packages/chromadb/__init__.py; do
    cat <<'__EOT__' | patch -N $file >/dev/null 2>&1
--- before	2024-01-22 16:39:03.437971849 +0900
+++ after	2024-01-22 16:38:41.226667742 +0900
@@ -76,13 +76,16 @@ if not is_client:
             __import__("pysqlite3")
             sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
         else:
-            raise RuntimeError(
-                "\033[91mYour system has an unsupported version of sqlite3. Chroma \
-                    requires sqlite3 >= 3.35.0.\033[0m\n"
-                "\033[94mPlease visit \
-                    https://docs.trychroma.com/troubleshooting#sqlite to learn how \
-                    to upgrade.\033[0m"
-            )
+            import sys
+            __import__("pysqlite3")
+            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
+            #raise RuntimeError(
+            #    "\033[91mYour system has an unsupported version of sqlite3. Chroma \
+            #        requires sqlite3 >= 3.35.0.\033[0m\n"
+            #    "\033[94mPlease visit \
+            #        https://docs.trychroma.com/troubleshooting#sqlite to learn how \
+            #        to upgrade.\033[0m"
+            #)


 def configure(**kwargs) -> None:  # type: ignore
__EOT__
done

if [ -r ~/.bashrc ]; then
    source ~/.bashrc
fi
streamlit run ${START_FILE} --server.address 0.0.0.0 $ST_OPT

