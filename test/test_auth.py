import sys, os, time
import re
import unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")

from pyutils import isEmpty, calculate_md5, to_md5
from testutils import test_title, Fail
import tiktoken
from crabDB import CrabTask, CrabDB, CrabUser, CrabSession, CrabThread, CrabBot, CrabMessage, CrabFileSegment

class TestA(unittest.TestCase):

    def check_user(self, user:CrabUser, uid, username, pw, ignore_pw=False ):
        self.assertIsNotNone( user, msg=f'can not create {username}')
        self.assertEqual( user.xId, uid, msg=f'invalid user id {uid}')
        self.assertEqual( user.name, username, msg=f'invalid username {username}')
        if not ignore_pw:
            self.assertEqual( user.passwd, pw, msg="initial passwd is must blank")

    def test_users(self):
        case:str = test_title()
        client:CrabDB = CrabDB( on_memory=True )
        session_root:CrabSession = client.login( 'root', '')
        self.assertIsNotNone( session_root, msg='can not initial root login')
        self.check_user( session_root.user, 3, 'root', '')
        #
        un:str = 'testuser'
        testuser_id:int = 1001
        session_root.upsert_user( un )
        user_testuser:CrabUser = client.get_user( un )
        self.check_user( user_testuser, testuser_id, un, '')
        # 
        session_testuser:CrabSession = client.login( un, '' )
        self.assertIsNotNone( session_testuser, msg=f'can not initial {un} login')
        self.check_user( session_testuser.user, testuser_id, un, '')
        # 
        session_x = client.login( un, 'abc123' )
        self.assertIsNone( session_x, msg=f"不正なパスワードでログインできた")
        #
        testuser_pw:str = 'secretpasswd'
        hash:str = to_md5(testuser_pw)
        user_testuser.passwd = testuser_pw
        session_testuser.upsert_user(user_testuser)
        user_testuser:CrabUser = client.get_user( un )
        self.assertNotEqual( user_testuser.passwd, testuser_pw, msg=f"plain passwd" )
        self.assertNotEqual( user_testuser.passwd, hash, msg=f"plain passwd" )
        #
        session_testuser:CrabSession = client.login( un, '' )
        self.assertIsNotNone( session_testuser, msg=f'can not initial {un} login')
        self.check_user( session_testuser.user, testuser_id, un, '', ignore_pw=True )

if __name__ == '__main__':
    unittest.main()