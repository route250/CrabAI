import streamlit as st
from openai import OpenAI
from openai.types import CreateEmbeddingResponse
from crabDB import CrabDB, CrabSession, CrabUser, CrabBot, CrabThread

ai_title = '蟹カニAI'

def isEmpty( value ) ->bool:
    if isinstance(value,(str,list)):
        return len(value)==0
    return value is not None

def indexOf( options:list, value, default=None ):
    try:
        return options.index( value )
    except:
        return default

@st.cache_resource
def get_database()->CrabDB:
    DB = CrabDB()
    return DB

def handle_bot_close():
    st.session_state['edit']=None
    st.session_state['errormsg']=None

def handle_login( username ):
    DB:CrabDB = get_database()
    s = DB.login( username )
    if isinstance(s,CrabSession):
        st.session_state['session'] = s
        handle_bot_close()

def handle_logout():
    session:CrabSession = st.session_state.get('session')
    if session is not None:
        session.logout()
    st.session_state['session']=None
    handle_bot_close()

def main():

    # ログイン状態判定
    SESSION:CrabSession = st.session_state.get('session')
    if SESSION is None:
        # ログイン画面
        st.set_page_config(page_title=ai_title,layout="centered")
        st.title(ai_title)
        DB:CrabDB = get_database()
        userlist:list[str] = DB.get_usernames()
        username = st.selectbox( 'Select User:', userlist, placeholder='select username', key='username', index=None )
        st.button( label='Login', key='submit', on_click=handle_login, args=(username,) )
        return

    # ログイン済み
    UserName:str = SESSION.user.name
    UserId:int = SESSION.user.xId

    def handle_bot_click(bot):
        SESSION.create_thread(botId=bot.xId)
        handle_bot_close()

    def handle_thread_click(thre):
        SESSION.set_current_thread(thre.xId)
        handle_bot_close()

    def handle_bot_edit(bot):
        if bot is None:
            bot = SESSION.create_new_bot()
        SESSION.set_current_thread()
        st.session_state['edit']=bot
        st.session_state['errormsg']=None

    def handle_user_edit(user:CrabUser):
        if user is None:
            user = CrabUser()
        SESSION.set_current_thread()
        st.session_state['edit']=user
        st.session_state['errormsg']=None

    st.set_page_config(page_title=ai_title,layout="wide")

    edit_target:CrabBot = st.session_state.get('edit')
    current_thread:CrabThread = SESSION.get_current_thread() if edit_target is None else None

    def handle_thread_auth():
        if current_thread is not None:
            try:
                auth = st.session_state.get('thre_auth')
                if auth == 'public':
                    current_thread.auth=[]
                else:
                    current_thread.auth=[current_thread.owner]
                SESSION.update_thread_auth( current_thread.xId, current_thread.auth )
                st.session_state['errormsg']=None
            except Exception as ex:
                st.session_state['errormsg'] = str(ex)

    # サイドバー
    with st.sidebar:
        st.title(ai_title)

        colx1, colx2 = st.columns([3,1])
        with colx1:
            st.button( key='self_edit', label=f"User: {UserName}", use_container_width=True, on_click=handle_user_edit, args=(SESSION.user,) )
        with colx2:
            st.button( key='logout', label='logout', on_click=handle_logout )

        # スレッド設定
        if current_thread is not None:
            is_not_owner = UserId != current_thread.owner
            with colx1:
                st.write( f"{current_thread.title} - {current_thread.get_bot_name()}")
            with colx2:
                def handle_show_thread():
                    if "show_thread" in st.session_state:
                        del st.session_state['show_thread']
                    else:
                        st.session_state['show_thread']="x"
                st.button( "....", on_click=handle_show_thread )
            if "show_thread" in st.session_state:
                thre_model = current_thread.get_bot_name()
                st.text_input( label='Model', value=thre_model, disabled=True )
                thre_owner = SESSION.get_username( current_thread.owner )
                st.text_input( label='Owner', value=thre_owner, disabled=True )
                auth_idx = 0 if current_thread.auth is None or len(current_thread.auth)==0 else 1
                st.selectbox( label='auth', key='thre_auth', options=['public','private'], index=auth_idx, on_change=handle_thread_auth, disabled=is_not_owner )

        st.header('bots',divider='gray')
        bot_col1, bot_col2 = st.columns([3,1])
        for bot in SESSION.get_bots():
            with bot_col1:
                st.button( key=f"b{bot.xId}", label=bot.name, use_container_width=True, on_click=handle_bot_click, args=(bot,) )
            with bot_col2:
                st.button( key=f"e{bot.xId}", label='...', on_click=handle_bot_edit, args=(bot,) )
        with bot_col2:
            st.button( key='newbot', label='create', on_click=handle_bot_edit, args=(None,))

        st.header('threads',divider='gray')
        for thre in SESSION.get_threads():
            st.button( key=f"t{thre.xId}", label=thre.title, on_click=handle_thread_click, args=(thre,) )

        if SESSION.is_root(UserName):
            DB:CrabDB = get_database()
            st.header('Users',divider='gray')
            for user in DB.get_users():
                st.button( key=f"u{user.xId}", label=user.name, use_container_width=True, on_click=handle_user_edit, args=(user,) )
            st.button( key='newuser', label='create User', on_click=handle_user_edit, args=(None,))

    # 画面判定        

    if isinstance( edit_target, CrabBot ):
        SESSION.set_current_thread()
        is_not_owner = UserId != edit_target.owner
        # ボット編集
        def handle_bot_save( edit_bot:CrabBot ):
            if is_not_owner:
                return
            name = st.session_state.get('bot_name')
            if name is None or len(name)==0:
                return
            owner = st.session_state.get('bot_owner')
            auth = st.session_state.get('bot_auth')
            description = st.session_state.get('bot_description')
            model = st.session_state.get('bot_model')
            max_tokens = st.session_state.get('bot_max_tokens')
            input_tokens = st.session_state.get('bot_input_tokens')
            temperature = st.session_state.get('bot_temperature')
            prompt = st.session_state.get('bot_prompt')
            prompt = st.session_state.get('bot_prompt')
            llm = st.session_state.get('bot_llm')
            retrive = st.session_state.get('bot_retrive')
            rag = st.session_state.get('bot_rag')

            edit_bot.name = name
            if auth == 'public':
                edit_bot.auth=[]
            else:
                edit_bot.auth=[edit_bot.owner]
            edit_bot.description = description
            edit_bot.model = model
            edit_bot.max_tokens = max_tokens
            edit_bot.input_tokens = input_tokens
            edit_bot.temperature = temperature
            edit_bot.prompt = prompt
            edit_bot.llm = llm
            edit_bot.retrive = retrive
            edit_bot.rag = rag

            try:
                SESSION.update_bot( edit_bot )
                st.session_state['errormsg']=None
            except Exception as ex:
                st.session_state['errormsg'] = str(ex)

        # 編集画面
        col31,col32,col33 = st.columns([5,1,1])
        with col31:
            errmsg = st.session_state['errormsg']
            if errmsg is not None and len(errmsg)>0:
                st.error(errmsg)
        with col32:
            st.button( label='Save', on_click=handle_bot_save, args=(edit_target,), disabled=is_not_owner )
        with col33:
            st.button( label='Close', on_click=handle_bot_close )
        col41,col42,col43 = st.columns([3,1,1])
        with col41:
            st.text_input( label='Name', key='bot_name', value=edit_target.name, disabled=is_not_owner )
        with col42:
            username:str = SESSION.get_username( edit_target.owner )
            st.text_input( label='Owner', key='bot_owner', value=username, disabled=True )
        with col43:
            auth_idx = 0 if edit_target.auth is None or len(edit_target.auth)==0 else 1
            st.selectbox( label='auth', key='bot_auth', options=['public','private'], index=auth_idx, disabled=is_not_owner )
        st.text_area( label='description', key='bot_description', value=edit_target.description, height=120, disabled=is_not_owner )
        col51,col52,col53,col54 = st.columns([1,1,1,1])
        with col51:
            models=CrabBot.get_model_name_list()
            st.selectbox( label='model', key='bot_model', options=models, index=indexOf(models,edit_target.model,0), disabled=is_not_owner )
        with col52:
            st.selectbox( label='max_tokens', key='bot_max_tokens', options=CrabBot.TOKENS_LIST, index=CrabBot.indexOf_tokens( edit_target.max_tokens ), disabled=is_not_owner )
        with col53:
            st.selectbox( label='input_tokens', key='bot_input_tokens', options=CrabBot.TOKENS_LIST, index=CrabBot.indexOf_tokens( edit_target.input_tokens ), disabled=is_not_owner )
        with col54:
            st.number_input( label='temperature', key='bot_temperature', min_value=0.0, max_value=2.0, step=0.1, value=edit_target.temperature, disabled=is_not_owner )
        col55,col56,col57 = st.columns([1,1,1])
        with col55:
            st.checkbox( label='LLM', key='bot_llm', value=edit_target.llm, disabled=is_not_owner )
        with col56:
            st.checkbox( label='Retrive', key='bot_retrive', value=edit_target.retrive, disabled=is_not_owner )
        with col57:
            st.checkbox( label='RAG', key='bot_rag', value=edit_target.rag, disabled=is_not_owner )
        st.text_area( label='prompt', key='bot_prompt', value=edit_target.prompt, height=300, disabled=is_not_owner )

        if isinstance( edit_target.files,list):
            def handle_remove_file(fileId):
                SESSION.removeFile( edit_target.xId, fileId)
                next = SESSION.get_bot( edit_target.xId)
                if next is not None:
                    st.session_state['edit'] = next
            colF1,colF2 = st.columns([3,1])
            for fileId in edit_target.files:
                with colF1:
                    file_stat = SESSION.get_file_status( fileId )
                    st.write( f"{fileId:>10d}", SESSION.get_file_name( fileId ), file_stat )
                if not is_not_owner:
                    with colF2:
                        st.button( key=f'delfile_{fileId}', label='del', on_click=handle_remove_file, args=(fileId,) )
        if not is_not_owner:
            uploaded_file = st.file_uploader( key="bot_uploadFile", label="select file", type=['txt'])
            if uploaded_file is not None and st.session_state.get('prevfile') != uploaded_file:
                st.write( "uploaded file:", uploaded_file.name )
                tmp_file_name=SESSION.create_tempfile()
                with open(tmp_file_name,'wb') as inp:
                    inp.write( uploaded_file.getvalue())
                fileId = SESSION.register_file( edit_target.xId, tmp_file_name, source=uploaded_file.name )
                next = SESSION.get_bot( edit_target.xId)
                if next is not None:
                    st.session_state['edit'] = next
                st.session_state['prevfile'] = uploaded_file

    elif isinstance(edit_target,CrabUser):
        SESSION.set_current_thread()
        # 編集画面
        def handle_user_save( user:CrabUser):
            try:
                name = st.session_state.get('user_name')
                if name is None or len(name)==0:
                    raise Exception("invalid userName")
                passwd = st.session_state.get('user_passwd')
                email = st.session_state.get('user_email')
                description = st.session_state.get('user_description')
                openai_api_key = st.session_state.get('user_openai_api_key')
                share_key = st.session_state.get('user_share_key')
                user.name=name
                user.passwd=passwd
                user.email=email
                user.description=description
                user.openai_api_key=openai_api_key
                user.share_key=share_key
                SESSION.upsert_user(user)
                st.session_state['errormsg']=None
            except Exception as ex:
                st.session_state['errormsg'] = str(ex)

        col31,col32,col33 = st.columns([5,1,1])
        with col31:
            errmsg = st.session_state['errormsg']
            if errmsg is not None and len(errmsg)>0:
                st.error(errmsg)
        with col32:
            st.button( label='Save', on_click=handle_user_save, args=(edit_target,) )
        with col33:
            st.button( label='Close', on_click=handle_bot_close )
        st.text_input( label='Name', key='user_name', value=edit_target.name )
        st.text_input( label='Passwd', key='user_passwd', type='password', value=edit_target.passwd )
        st.text_input( label='E-mail', key='user_email', value=edit_target.email )
        st.text_input( label='OpenAI api-key', key='user_openai_api_key', type='password', value=edit_target.openai_api_key )
        st.checkbox( label='Share key', key='user_share_key', value=edit_target.share_key, disabled=not SESSION.is_root(UserName) )
        st.text_area( label='Description', key='user_description', value=edit_target.description, height=120 )

    elif edit_target is None and current_thread is not None:
        # チャット画面
        is_not_owner = UserId != current_thread.owner

        st.subheader( current_thread.get_bot_name() )
        desc=current_thread.get_bot_description()
        if not isEmpty(desc):
            st.write( desc )
        st.divider()

        # これまでのチャット履歴を全て表示する 
        for message in current_thread.get_messages(30):
            with st.chat_message(message.role):
                st.markdown(message.content)

        # 持ち主じゃなければ見るだけ
        if not is_not_owner:

            # ユーザーの入力が送信された際に実行される処理 
            user_input = st.chat_input("Your input?", disabled=is_not_owner)

            if user_input:
                # ユーザの入力を表示する
                with st.chat_message("user"):
                    st.markdown(user_input)
                # 非同期実行
                response = current_thread.run(user_input)
                # ChatBotの返答を表示する 
                with st.chat_message("assistant"):
                    assistant_msg:str = ""
                    assistant_response_area = st.markdown("")
                    for part in response:
                        # 回答を逐次表示
                        tmp_assistant_msg = part or ""
                        if tmp_assistant_msg == CrabBot.CLS:
                            assistant_msg=""
                        else:
                            assistant_msg += tmp_assistant_msg
                        assistant_response_area.write(assistant_msg)

if __name__ == "__main__":
    main()