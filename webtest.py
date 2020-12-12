import socket
from jinja2 import Template

def f1(request):
    """
    处理用户请求 并返回内容
    :param request: 用户请求的信息
    :return: 
    """
    f = open('index.html','rb')
    data = f.read()
    f.close()
    return data

def f2(request):
    return b'you are entering f2'

routers = {
    ('/xxxx',f1),
    ('/oooo',f2),
}

def run():
    sock = socket.socket()
    sock.bind(('127.0.0.1', 8080))
    sock.listen(5)

    while True:
        conn, addr = sock.accept() #在这里等待住
    #有人来了
        data = conn.recv(8096)
        data = str(data,encoding='utf8')
        headers, bodies = data.split('\r\n\r\n')
        temp_list = headers.split('\r\n')
        method, url, protocol =temp_list[0].split(' ')
        conn.send(b"HTTP/1.1 200 OK\r\n\r\n")  # 响应头

        for i in routers:
            if i[0]== url:
                func_name = i[1]
                break

        if func_name:
            response =  func_name(data)  #函数指针
        else:
            response = b"404"

        conn.send(response)
        conn.close()

if __name__ == '__main__':
    run()