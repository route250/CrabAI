import sys, os, unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")
from crabDB import CrabDB, CrabSession, CrabThread, CrabBot, CrabMessage, CrabFileSegment
import crabDB

class TestA(unittest.TestCase):

    def test_deocdeBool(self):
        print("test_decodeBool")
        self.assertEqual(crabDB.decodeBool(True), True, msg="decodeBool(True)")
        self.assertEqual(crabDB.decodeBool(False), False, msg="decodeBool(False)")
        self.assertEqual(crabDB.decodeBool(None), False, msg="decodeBool(None)")
        self.assertEqual(crabDB.decodeBool('True'), True, msg="decodeBool('True')")
        self.assertEqual(crabDB.decodeBool('False'), False, msg="decodeBool('False')")
        self.assertEqual(crabDB.decodeBool(''), False, msg="decodeBool('')")
        self.assertEqual(crabDB.decodeBool(1), True, msg="decodeBool(1)")
        self.assertEqual(crabDB.decodeBool(0), False, msg="decodeBool(0)")
        self.assertEqual(crabDB.decodeBool([1]), True, msg="decodeBool([0])")
        self.assertEqual(crabDB.decodeBool([]), False, msg="decodeBool([])")

    def test_get_asList(self):
        print(f"test get_asList")
        self.assertEqual(crabDB.get_asList({'key': 'value'}, 'key'), ['value'])
        self.assertEqual(crabDB.get_asList({'key': []}, 'key'), [])
        self.assertEqual(crabDB.get_asList({'key': ['value']}, 'key'), ['value'])
        self.assertEqual(crabDB.get_asList({'key': ['value1', 'value2']}, 'key'), ['value1', 'value2'])
        self.assertEqual(crabDB.get_asList({'key': [['value']]}, 'key'), ['value'])
        self.assertEqual(crabDB.get_asList({'key': [[['value']]]}, 'key'), [['value']])
        self.assertEqual(crabDB.get_asList({'key': 'value'}, 'nonexistent'), [])
        self.assertEqual(crabDB.get_asList({'key': None}, 'key'), [])
        self.assertEqual(crabDB.get_asList({}, 'key'), [])
        self.assertEqual(crabDB.get_asList(None, 'key'), [])
        self.assertEqual(crabDB.get_asList({}, None), [])

    def test_merge_metadatas(self):
        res = {
            'metadatas': [{'id': 1}, {'id': 2}],
        }
        expected_output = [
            {'id': 1},
            {'id': 2}
        ]
        self.assertEqual(crabDB.merge_metadatas(res), expected_output, msg="basic functionality 1")
        res = {
            'metadatas': [{'id': 1}, {'id': 2}],
            'documents': ['doc1', 'doc2'],
        }
        expected_output = [
            {'id': 1, 'content': 'doc1'},
            {'id': 2, 'content': 'doc2'}
        ]
        self.assertEqual(crabDB.merge_metadatas(res), expected_output, msg="basic functionality 2")
        res = {
            'metadatas': [{'id': 1}, {'id': 2}],
            'documents': ['doc1', 'doc2'],
            'distances': [0.1, 0.2]
        }
        expected_output = [
            {'id': 1, 'content': 'doc1', 'distance': 0.1},
            {'id': 2, 'content': 'doc2', 'distance': 0.2}
        ]
        self.assertEqual(crabDB.merge_metadatas(res), expected_output, msg="basic functionality 3")

        res = {
            'metadatas': [[{'id': 1}, {'id': 2}]],
        }
        expected_output = [
            {'id': 1},
            {'id': 2}
        ]
        self.assertEqual(crabDB.merge_metadatas(res), expected_output, msg="basic functionality 4")
        res = {
            'metadatas': [[{'id': 1}, {'id': 2}]],
            'documents': [['doc1', 'doc2']],
        }
        expected_output = [
            {'id': 1, 'content': 'doc1'},
            {'id': 2, 'content': 'doc2'}
        ]
        self.assertEqual(crabDB.merge_metadatas(res), expected_output, msg="basic functionality 5")
        res = {
            'metadatas': [[{'id': 1}, {'id': 2}]],
            'documents': [['doc1', 'doc2']],
            'distances': [[0.1, 0.2]]
        }
        expected_output = [
            {'id': 1, 'content': 'doc1', 'distance': 0.1},
            {'id': 2, 'content': 'doc2', 'distance': 0.2}
        ]
        self.assertEqual(crabDB.merge_metadatas(res), expected_output, msg="basic functionality 6")
        res = {
            'metadatas': [{'id': 1}, {'id': 2}, {'id': 3}],
            'documents': ['doc1', 'doc2'],
            'distances': [0.1, 0.2]
        }
        expected_output = [
            {'id': 1, 'content': 'doc1', 'distance': 0.1},
            {'id': 2, 'content': 'doc2', 'distance': 0.2},
            {'id': 3}
        ]
        self.assertEqual(crabDB.merge_metadatas(res), expected_output, msg="missing length")

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()