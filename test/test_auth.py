import sys, os, time, traceback
import re
import unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")

from pyutils import isEmpty, to_md5
from testutils import test_title, Fail
import tiktoken
from crabDB import CrabTask, CrabDB, CrabUser, CrabSession, CrabThread, CrabBot, CrabMessage, CrabFileSegment

class TestA(unittest.TestCase):

    def check_user(self, user:CrabUser, uid, username, pw ):
        self.assertIsNotNone( user, msg=f'user {username} is None')
        self.assertEqual( user.xId, uid, msg=f'invalid user id {uid}')
        self.assertEqual( user.name, username, msg=f'invalid username {username}')
        if isEmpty(pw):
            self.assertEqual( user.passwd, '', msg="initial passwd is must blank")
        else:
            self.assertNotEqual( user.passwd, pw, msg="initial passwd is must blank")
            self.assertTrue( user.passwd.startswith('pw🦀'), msg="initial passwd data")

    def test_user_passwd(self):
        case:str = test_title()
        client:CrabDB = CrabDB( on_memory=True )
        session_root:CrabSession = client.login( 'root', '')
        self.assertIsNotNone( session_root, msg='can not initial root login')
        self.check_user( session_root.user, 3, 'root', '')
        # create new user
        username:str = 'testuser'
        userId:int = 1001
        session_root.upsert_user( username )
        user_testuser:CrabUser = client.get_user( username )
        self.check_user( user_testuser, userId, username, '')

        # password check loop
        current_passwd = ''
        for new_passwd in [ 'secret001', '002secret','', 'abc123' ]:

            # Must be able to log in with current password.
            session2:CrabSession = client.login( username, current_passwd )
            self.assertIsNotNone( session2, msg=f'can not login by before_pw {username}/{current_passwd}')
            self.check_user( session2.user, userId, username, current_passwd )
            user2:CrabUser = session2.user

            # chenge other
            memo = 'debug test ' + user2.description
            user2.description = memo
            session2.upsert_user(user2)
            user3:CrabUser = client.get_user( username )
            self.assertEqual( user3.description, memo, msg=f"can not update other? ")

            # When changing anything other, the password will not be changed.
            session4:CrabSession = client.login( username, current_passwd )
            self.assertIsNotNone( session4, msg=f'can not login by before_pw {username}/{new_passwd}')
            self.check_user( session4.user, userId, username, current_passwd )
            user4:CrabUser = session4.user

            # chenge passwd
            user4.passwd = new_passwd
            session4.upsert_user(user4)

            # check database
            # Plain passwords should not be recorded in the database.
            hash:str = to_md5(new_passwd)
            user5:CrabUser = client.get_user( username )
            if isEmpty(new_passwd):
                self.assertEqual( user5.passwd, new_passwd, msg=f"can not crypt passwd" )
                self.assertNotEqual( user5.passwd, hash, msg=f"can not crypt passwd" )
            else:
                self.assertNotEqual( user5.passwd, new_passwd, msg=f"can not crypt passwd" )
                self.assertNotEqual( user5.passwd, hash, msg=f"can not crypt passwd" )

            # Unable to login with the old password
            session6:CrabSession = client.login( username, current_passwd )
            self.assertIsNone( session6, msg=f'can login by before_pw {username}/{current_passwd}')

            # login with new passwd
            session7:CrabSession = client.login( username, new_passwd )
            self.assertIsNotNone( session7, msg=f'can not login by new passwd {username}/{new_passwd}')
            self.check_user( session7.user, userId, username, new_passwd )

            # Unable to login with the invalid passwd
            invalid_pw = f"abc ${new_passwd} def"
            session8:CrabSession = client.login( username, invalid_pw )
            self.assertIsNone( session8, msg=f'can login by invalid pw {username}/{invalid_pw}')

            current_passwd = new_passwd

    def test_user_oepnai_api_key(self):
        case:str = test_title()
        client:CrabDB = CrabDB( on_memory=True )

        root_session:CrabSession = client.login( 'root', '')
        self.assertIsNotNone( root_session, msg='can not initial root login')
        self.check_user( root_session.user, 3, 'root', '')
        self.assertEqual( root_session.user.share_key, False, msg="root default share_key")

        # create new user
        username:str = 'testuser'
        userId:int = 1001
        root_session.upsert_user( username )
        user_testuser:CrabUser = client.get_user( username )
        self.check_user( user_testuser, userId, username, '')
        user_session:CrabSession = client.login( username, '' )
        self.check_user( user_session.user, userId, username, '')
        self.assertEqual( user_session.user.share_key, True, msg=f"{username} default share_key")

        #
        ekey = 'sk-abcdefghijklmnopqrstuvwxyz0123456789abcdefghijkl'
        rkey = 'sk-0123456789abcdefghijklmnopqrstuvwxyzabcdefghijkl'
        ukey = 'sk-abcdefghijklmnopqrstuvwxyzabcdefghijkl0123456789'

        os.environ['OPENAI_API_KEY']=ekey[:10]
        key = root_session._get_openai_api_key()
        self.assertEqual( key, "", msg=f"")
        key = user_session._get_openai_api_key()
        self.assertEqual( key, "", msg=f"")

        os.environ['OPENAI_API_KEY']=ekey
        key = root_session._get_openai_api_key()
        self.assertEqual( key, ekey, msg=f"")
        key = user_session._get_openai_api_key()
        self.assertEqual( key, ekey, msg=f"")

        # update root key
        root_session.user.openai_api_key = rkey
        root_session.upsert_user( root_session.user )
        # 
        key = root_session._get_openai_api_key()
        self.assertEqual( key, rkey, msg=f"")
        key = user_session._get_openai_api_key()
        self.assertEqual( key, ekey, msg=f"")

        # update share_key to True
        root_session.user.share_key = True
        root_session.upsert_user( root_session.user )
        # 
        key = root_session._get_openai_api_key()
        self.assertEqual( key, rkey, msg=f"")
        key = user_session._get_openai_api_key()
        self.assertEqual( key, rkey, msg=f"")

        # update root key
        root_session.user.openai_api_key = rkey[:10]
        root_session.upsert_user( root_session.user )
        # 
        key = root_session._get_openai_api_key()
        self.assertEqual( key, ekey, msg=f"")
        key = user_session._get_openai_api_key()
        self.assertEqual( key, ekey, msg=f"")

        # update root key
        root_session.user.openai_api_key = rkey
        root_session.upsert_user( root_session.user )
        # 
        key = root_session._get_openai_api_key()
        self.assertEqual( key, rkey, msg=f"")
        key = user_session._get_openai_api_key()
        self.assertEqual( key, rkey, msg=f"")

        # update share_key to False
        user_session.user.share_key = False
        user_session.upsert_user( user_session.user )
        os.environ['OPENAI_API_KEY']=ekey[:10]
        key = user_session._get_openai_api_key()
        self.assertEqual( key, "", msg=f"")

        os.environ['OPENAI_API_KEY']=ekey
        key = user_session._get_openai_api_key()
        self.assertEqual( key, ekey, msg=f"")

        # update user key
        user_session.user.openai_api_key = ukey
        user_session.upsert_user( user_session.user )
        os.environ['OPENAI_API_KEY']=ekey[:10]
        key = user_session._get_openai_api_key()
        self.assertEqual( key, ukey, msg=f"")
        os.environ['OPENAI_API_KEY']=ekey
        key = user_session._get_openai_api_key()
        self.assertEqual( key, ukey, msg=f"")

if __name__ == '__main__':
    unittest.main()
