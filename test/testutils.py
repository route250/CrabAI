import sys, os, time, traceback, inspect
import unittest
sys.path.append( os.getcwd()+"/test")
sys.path.append( os.getcwd()+"/src")

def isEmpty(value) ->bool:
    if value is None:
        return True
    if isinstance(value,str):
        return value.strip()==''
    return value

def test_title() ->str:
    text:str = inspect.currentframe().f_back.f_code.co_name
    print("------------------------------------------------")
    print(f"  {text}")
    print("------------------------------------------------")
    return text

def Fail(log) ->bool:
    print(f"[Fail]{log}")
    return True