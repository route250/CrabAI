import os,sys
import hashlib

def isEmpty(value) ->bool:
    if value is None:
        return True
    if isinstance(value,str):
        return value.strip()==''
    return value

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def to_md5( text:str ) ->str:
    hash_md5 = hashlib.md5()
    hash_md5.update(text.encode())
    return hash_md5.hexdigest()

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