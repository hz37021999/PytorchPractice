import urllib.request
import urllib.parse
import time
import random
import hashlib
import json

while True:
    msg = input('输入内容（q!退出程序）：')
    if msg == 'q!':
        break
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
    ua = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    data = {}

    #def get_md(value):
    #    m = hashlib.md5()
    #    m.update(value.encode('utf-8'))
    #    return m.hexdigest()

    #msg = 'I love you.'
    #salt = str(int(time.time()*1000 + random.randint(0,10)))
    #sign = get_md('fanyideskweb'+ msg + salt + 'ebSeFb%=XZ%T[KZ)c(sy!')
    #lts = str(int(time.time()*1000))

    data['i'] = msg
    data['from'] = 'AUTO'
    data['to'] = 'AUTO'
    data['smartresult'] = 'dict'
    data['client'] = 'fanyideskweb'
    #data['salt'] = salt
    #data['sign'] = sign
    #data['lts'] = lts
    data['bv'] = 'f4d62a2579ebb44874d7ef93ba47e822'
    data['doctype'] = 'json'
    data['version'] = '2.1'
    data['keyfrom']= 'fanyi.web'
    data['action']= 'FY_BY_REALTlME'
    data = urllib.parse.urlencode(data).encode('utf-8')

    req = urllib.request.Request(url,data)
    req.add_header('User-Agent',ua)       #增加header

    response = urllib.request.urlopen(req)
    html = response.read().decode('utf-8')
    target = json.loads(html)
    print(target['translateResult'][0][0]['tgt'])
    time.sleep(2)