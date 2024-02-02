import sys, os, unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
from crabDB import CrabDB, CrabSession, CrabThread2, CrabBot, CrabMessage, CrabFileSegment

class TestA(unittest.TestCase):

    def test_sample(self):
        DB:CrabDB = CrabDB( on_memory=True)
        Session:CrabSession = DB.login( 'root' )
        Thre:CrabThread2 = Session.get_current_thread()

        data=[
            ['お好み焼の店は？', '金太とか千房とかかな'],
            ['お好み焼のトッピングは？', 'マヨネーズです'],
            ['お好み焼の具材は？', '豚肉、お餅、イカ、エビとか色々です'],
            ['お好み焼の値段は？', '300円から1500円まで色々です。'],
            ['お好み焼の大きさは？', '20cm程度です。'],
            ['お好み焼の重さは？', '200gから2kgまであります。'],
            ['ねこの名前は？', 'とらきちです'],
            ['ねこの性別は？', '男の子です'],
            ['ねこの体重は？', '2kgです'],
            ['ねこの色は？', 'しましまです'],
            ['ねこのお気に入りは？', 'コタツの中です'],
        ]
        for a in data:
            Thre.add_user_message( a[0] )
            tail_messages, query_embedding = Session._get_tail_messages( Thre.xId, 10 )
            print( f"{tail_messages[0].xId} - {tail_messages[-1].xId } ")
            Thre.add_assistant_message( a[1] )
        Thre.add_user_message( 'お好み焼の店は？' )
        tail_messages, query_embedding = Session._get_tail_messages( Thre.xId, 10 )
        excludeId = tail_messages[0].xId
        max_distance = 0.3
        hist_messges:list = Session._retrive_message( ThreId=Thre.xId, emb=query_embedding, excludeId=excludeId, max_distance=max_distance )
        for m in hist_messges:
            print( m.to_obj() )
if __name__ == '__main__':
    unittest.main()