import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())

class Foobar:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print("hello" + str(args[0]))

def func(*args,**kwargs):
    print(args)  # (1,2,3,4)
    print(kwargs)  #{'x': 3, 'y': 5}
#func(1,2,3,4,x=3,y=5)

f = Foobar()
f(4,2,3)