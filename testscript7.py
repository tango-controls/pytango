import tango
from tango import DevState

dev = tango.DeviceProxy('pipeServer/tango/1')
blob = dev.read_pipe('TestPipe')
print 'PipeClient.py: ', blob
print '+++++++++++++++++++++++++++++'

blob = ('pipeWriteTest0', dict(x=5.9, y=15.1, anInt=-169, str="the test",
                               truth=False, state=DevState.FAULT))
print(blob)
#dev.write_pipe('TestPipe', blob)
print '+++++++++++++++++++++++++++++'
blob = ('pipeWriteTest0', {'x':5.9, 'y':15.1, 'anInt':-169, 'str':"the test",
                               'truth':False, 'state':DevState.FAULT})
print(blob)
#dev.write_pipe('TestPipe', blob)
print '+++++++++++++++++++++++++++++'
 
# float_list = [3.33, 3.34, 3.35, 3.36]
# inner_int_list = [11, 12, 13, 14, 15]
# inner_inner_data = [("InnerInnerFirstDE", 111),
#                     ("InnerInnerSecondDE", float_list),
#                     ("InnerInnerThirdDE", inner_int_list)]
# inner_inner_blob = ("InnerInner", dict(inner_inner_data))
# inner_data = [("InnerFirstDE", "Grenoble"),
#               ("InnerSecondDE", inner_inner_blob),
#               ("InnerThirdDE", True)]
# inner_blob = ("Inner", dict(inner_data))
# int_list = [3, 4, 5, 6]
# pipe_data = [("1DE", inner_blob), ("2DE", int_list)]
# blob = ("PipeWriteTest1", dict(pipe_data))
# dev.write_pipe('TestPipe', blob)

inner_data = [("pi", 3.142),("twopi",6.284)]
inner_blob = ("Inner", dict(inner_data))
pipe_data = [("1DE", inner_blob), ("2DE", 66), ("3DE","Grenoble")]
blob = ("PipeWriteTest1", dict(pipe_data))
print(blob)
dev.write_pipe('TestPipe', blob)

print '+++++++++++++++++++++++++++++'
inner_data = [("pi", 3.142),("twopi",6.284)]
inner_blob = ("Inner", dict(inner_data))
double_list = [3.4, 4.5, 5.6, 6.7]
long_list = [5, 22, 23, 24, 25, 26]
pipe_data = [("1DE", inner_blob), ("2DE", 66), ("3DE","Grenoble"), ("4DE", long_list), ("5DE", double_list)]
blob = ("PipeWriteTest1", dict(pipe_data))
print(blob)
dev.write_pipe('TestPipe', blob)
