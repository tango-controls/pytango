
import PyTango
import sys
import os
import time
import string
import threading
from twisted.plugins.cred_anonymous import theAnonymousCheckerFactory

class EventCallBack:
    def __init__(self):
        self.cb_executed = 0
        self.cb_err = 0
        self.old_sec = 0
        self.old_usec = 0 
        self.delta_msec = 0
        
    def push_event(self, event_data):
        if sys.platform == 'win32':
            gettime = time.clock
        else:
            gettime = time.time
        t = gettime()
        sec = int(t)
        usec = int((t - sec) * 1000000)
        print('date : tv_sec = ' + str(sec) + ', tv_usec = ' + str(usec))
        delta_msec = int(((sec - self.old_sec) * 1000000 + (usec - self.old_usec)) / 1000)
        self.old_sec = sec
        self.old_usec = usec
        print('delta_msec = ' + str(delta_msec))
        self.cb_executed += 1
        try:
            print('StateEventCallBack::push_event(): called attribute ' + event_data.attr_name + ' event ' + event_data.event)
            if event_data.err == False:
                print('CallBack vector value : ' + str(event_data.attr_value.value))
            else:
                print('Error send to callback')
                if event_data.errors[0].reason == 'aaa':
                    self.cb_err += 1
        except:
            print('EventCallBack::push_event(): could not extract data !')
        

class DevTest(PyTango.Device_4Impl):
    def __init__(self,cl,name):
        PyTango.Device_4Impl.__init__(self,cl,name)
        print('In DevTest __init__')
        DevTest.init_device(self)

    def init_device(self):
        print('In Python init_device method')
        self.set_state(PyTango.DevState.ON)

        self.Short_attr_except = False
        self.event_change_attr_except = False
        self.event_quality_attr_except = False
        self.event_throw_out_of_sync = False

        self.attr_long = 1246
        self.attr_short_rw = 66;
        self.attr_long_rw2 = -1
        self.PollLong_attr_num = 0;
        self.PollString_spec_attr_num = 0;

        self.attr_short_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.attr_long_array = [1,2,3,4,5,6,7,8,9,10]
        self.attr_db_array = [7.77,8.88,9.99]
        self.attr_str_array = ['a','b']

        self.attr_short_rw = 66

        self.attr_spec_float = [1,2,3]
        self.attr_spec_boolean = [1,2,3,4,5]
        self.attr_spec_ushort = [1,2]
        self.attr_spec_uchar = [1,2,3,4,5,6]
        self.Long_spec_attr_rw = [1,2,3,4]

        self.num_cmd1 = 0
        self.num_cmd2 = 0
        self.attr_event_size = 4
        self.attr_event = [10,20,30,40,50,60,70,80,90,100]
        self.attr_event64_size = 2
        self.attr_event64 = [0x800000000,44,55]
        self.attr_qua_event = [1.2,2.4]
        self.attr_event_qua = PyTango.AttrQuality.ATTR_VALID

        self.fast_actua = 0.0
        self.slow_actua = 0
        self.slow_actua_write = 0
        
        self.enc_attr_data = bytes(bytearray([97,98,99,100]))
        self.enc_attr = ['Which format?', self.enc_attr_data] #''.join(chr(i) for i in self.enc_attr_data).encode('utf-8')]
        self.enc_format_toggle = False
        self.encoded_cmd_ctr = 0
        self.remote_dev = None
        self.cb = EventCallBack()
        self.eve_id = 0
        
        self.attr_ulong_rw = 0xC0000000
        self.attr_ulong64_rw = 0xC000000000000000
        
        self.attr_state_rw = PyTango.DevState.FAULT
        self.attr_spec_ulong_rw = [2222,22222,222222]
        self.attr_spec_ulong64_rw = [8888,88888,888888]
        self.attr_spec_state_rw = [PyTango.DevState.ON,PyTango.DevState.OFF]
        self.attr_sub_device_tst = False
        self.attr_slow = 3.3
        self.att_conf = 10

#------------------------------------------------------------------

    def delete_device(self):
        print('[Device delete_device method] for device',self.get_name())

#------------------------------------------------------------------

    def signal_handler(self,in_sig):
        print('[Device signal handler] received signal number',in_sig,' for device',self.get_name())

#------------------------------------------------------------------

#    def State(self):
#        print "[Device] In my own State method"
#        sta = PyTango.Device_3Impl.dev_state(self)
#        print 'Original state =',sta
#        self.set_state(PyTango.DevState.STANDBY)
#        return self.get_state()

#------------------------------------------------------------------

#    def Status(self):
#        print "[Device] In my own Status method"
#        self.st = PyTango.Device_3Impl.dev_status(self)
#        print "Status =",self.st
#        self.st = self.st + " Si,si...."
#        return self.st

#------------------------------------------------------------------

    def is_IOVoid_allowed(self):
        print('In is_IOVoid_allowed method')
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOVoid(self):
        print('[IOVoid::execute] ')

#------------------------------------------------------------------

    def is_IOBool_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOBool(self,in_bool):
        print('[IOBool::execute] received number',in_bool,' type = ',type(in_bool))
        if (in_bool == True):
            in_bool = False
        else:
            in_bool = True
        print('[IOBool::execute] return number',in_bool)
        return in_bool;

#------------------------------------------------------------------

    def is_IOShort_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOShort(self,in_data):
        print('[IOShort::execute] received number',in_data,' type = ',type(in_data))
        in_data = in_data * 2;
        print('[IOShort::execute] return number',in_data)
        return in_data;

#------------------------------------------------------------------

    def is_IOLong_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOLong(self,in_data):
        print('[IOLong::execute] received number',in_data,' type = ',type(in_data))
        in_data = in_data * 2;
        print('[IOLong::execute] return number',in_data)
        return in_data;

#------------------------------------------------------------------

    def is_IOFloat_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOFloat(self,in_data):
        print('[IOFloat::execute] received number',in_data,' type = ',type(in_data))
        in_data = in_data * 2;
        print('[IOFloat::execute] return number',in_data)
        return in_data;

#------------------------------------------------------------------

    def is_IODouble_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IODouble(self,in_data):
        print('[IODouble::execute] received number',in_data,' type = ',type(in_data))
        in_data = in_data * 2;
        print('[IODouble::execute] return number',in_data)
        return in_data;

#------------------------------------------------------------------

    def is_IOUShort_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOUShort(self,in_data):
        print('[IOUShort::execute] received number',in_data,' type = ',type(in_data))
        in_data = in_data * 2;
        print('[IOUShort::execute] return number',in_data)
        return in_data;

#------------------------------------------------------------------

    def is_IOULong_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOULong(self,in_data):
        print('[IOULong::execute] received number',in_data,' type = ',type(in_data))
        in_data = in_data * 2;
        print('[IOULong::execute] return number',in_data)
        return in_data;

#------------------------------------------------------------------

    def is_IOString_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOString(self,in_data):
        print("[IOString::execute] received string",in_data," type = ",type(in_data))
        l = range(len(in_data) - 1,-1,-1)
        out_data=""
        for i in l:
            out_data = out_data + in_data[i]
        print("[IOString::execute] return string",out_data)
#        self.y = out_data
        return out_data;

#------------------------------------------------------------------

    def is_IOCharArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOCharArray(self,in_data):
        l = range(len(in_data)-1,-1,-1);
        out_index=0
        out_data=[]
        for i in l:
            print("[IOCharArray::execute] received char",in_data[out_index])
            out_data.append(in_data[i])
            print("[IOStringArray::execute] return char",out_data[out_index])
            out_index = out_index+1
        return out_data

#------------------------------------------------------------------

    def is_IOShortArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOShortArray(self,in_data):
        l = range(len(in_data));
        for i in l:
            print("[IOShortArray::execute] received number",in_data[i])
            in_data[i] = in_data[i] * 2
            print("[IOShortArray::execute] return number",in_data[i])
        return in_data


#------------------------------------------------------------------

    def is_IOLongArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOLongArray(self,in_data):
        l = range(len(in_data));
        for i in l:
            print("[IOLongArray::execute] received number",in_data[i])
            in_data[i] = in_data[i] * 2
            print("[IOLongArray::execute] return number",in_data[i])
        return in_data

#------------------------------------------------------------------

    def is_IOFloatArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOFloatArray(self,in_data):
        l = range(len(in_data));
        for i in l:
            print("[IOFloatArray::execute] received number",in_data[i])
            in_data[i] = in_data[i] * 2
            print("[IOFloatArray::execute] return number",in_data[i])
        return in_data

#------------------------------------------------------------------

    def is_IODoubleArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IODoubleArray(self,in_data):
        l = range(len(in_data));
        for i in l:
            print("[IODoubleArray::execute] received number",in_data[i])
            in_data[i] = in_data[i] * 2
            print("[IODoubleArray::execute] return number",in_data[i])
        return in_data

#------------------------------------------------------------------

    def is_IOUShortArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOUShortArray(self,in_data):
        l = range(len(in_data));
        for i in l:
            print("[IOUShortArray::execute] received number",in_data[i])
            in_data[i] = in_data[i] * 2
            print("[IOUShortArray::execute] return number",in_data[i])
        return in_data

#------------------------------------------------------------------

    def is_IOULongArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOULongArray(self,in_data):
        l = range(len(in_data));
        for i in l:
            print("[IOULongArray::execute] received number",in_data[i])
            in_data[i] = in_data[i] * 2
            print("[IOULongArray::execute] return number",in_data[i])
        return in_data

#------------------------------------------------------------------

    def is_IOStringArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOStringArray(self,in_data):
        l = range(len(in_data)-1,-1,-1);
        out_index=0
        out_data=[]
        for i in l:
            print("[IOStringArray::execute] received String",in_data[out_index])
            out_data.append(in_data[i])
            print("[IOStringArray::execute] return String",out_data[out_index])
            out_index = out_index+1
        return out_data

#------------------------------------------------------------------

    def is_IOLongString_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOLongString(self,in_data):
        l = range(len(in_data[0]))
        for i in l:
            print("[IOLongString::execute] received number",in_data[0][i])
            in_data[0][i] = in_data[0][i] * 2
            print("[IOLongString::execute] return number",in_data[0][i])

        l = range(len(in_data[1]))
        for i in l:
            print("[IOLongString::execute] received string",in_data[1][i])
            print("[IOLongString::execute] return string",in_data[1][i])
        return in_data

#------------------------------------------------------------------

    def is_IODoubleString_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IODoubleString(self,in_data):
        l = range(len(in_data[0]))
        for i in l:
            print("[IODoubleString::execute] received number",in_data[0][i])
            in_data[0][i] = in_data[0][i] * 2
            print("[IODoubleString::execute] return number",in_data[0][i])

        l = range(len(in_data[1]))
        for i in l:
            print("[IODoubleString::execute] received string",in_data[1][i])
            print("[IODoubleString::execute] return string",in_data[1][i])
        return in_data

#------------------------------------------------------------------

    def IOState(self,in_data):
        print "[IOState::execute] received state",in_data
        self.set_state(in_data)

#------------------------------------------------------------------

    def is_IOThrow_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOThrow(self,in_data):
        print "[IOThrow::execute] throwing severity exception",in_data[0][0]
        if (in_data[0][0] == PyTango.ErrSeverity.WARN):
            PyTango.Except.throw_exception(in_data[1][0],'This is a test','IOThrow::execute()',PyTango.ErrSeverity.WARN)
        elif (in_data[0][0] == PyTango.ErrSeverity.ERR):
            PyTango.Except.throw_exception(in_data[1][0],'This is a test','IOThrow::execute()',PyTango.ErrSeverity.ERR)
        if (in_data[0][0] == PyTango.ErrSeverity.PANIC):
            PyTango.Except.throw_exception(in_data[1][0],'This is a test','IOThrow::execute()',PyTango.ErrSeverity.PANIC)



#------------------------------------------------------------------

    def is_IOReThrow_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOReThrow(self,in_data):
        nb_except = len(in_data[0])
        print "[IOReThrow::execute] throwing",nb_except,"exception(s)"
        try:
            if (in_data[0][0] == PyTango.ErrSeverity.WARN):
                PyTango.Except.throw_exception(in_data[1][0],'This is a test','IOThrow::execute()',PyTango.ErrSeverity.WARN)
            elif (in_data[0][0] == PyTango.ErrSeverity.ERR):
                PyTango.Except.throw_exception(in_data[1][0],'This is a test','IOThrow::execute()',PyTango.ErrSeverity.ERR)
            elif (in_data[0][0] == PyTango.ErrSeverity.PANIC):
                PyTango.Except.throw_exception(in_data[1][0],'This is a test','IOThrow::execute()',PyTango.ErrSeverity.PANIC)
        except PyTango.DevFailed,e:
            if (nb_except > 2):
                ind = 1
                for i in range(nb_except - 2):
                    try:
                        sev = in_data[0][ind]
                        if (sev == PyTango.ErrSeverity.WARN):
                            PyTango.Except.re_throw_exception(e,in_data[1][ind],'This is a test','IOReThrow::execute()',PyTango.ErrSeverity.WARN)
                        elif (sev == PyTango.ErrSeverity.ERR):
                            PyTango.Except.re_throw_exception(e,in_data[1][ind],'This is a test','IOReThrow::execute()',PyTango.ErrSeverity.ERR)
                        elif (sev == PyTango.ErrSeverity.PANIC):
                            PyTango.Except.re_throw_exception(e,in_data[1][ind],'This is a test','IOReThrow::execute()',PyTango.ErrSeverity.PANIC)
                    except PyTango.DevFailed,ex:
                        ind = ind + 1
                        e = ex


            last_sever = in_data[0][nb_except - 1]
            if (last_sever == PyTango.ErrSeverity.WARN):
                PyTango.Except.re_throw_exception(e,in_data[1][nb_except-1],'This is a test','IOReThrow::execute()',PyTango.ErrSeverity.WARN)
            elif (last_sever == PyTango.ErrSeverity.ERR):
                PyTango.Except.re_throw_exception(e,in_data[1][nb_except-1],'This is a test','IOReThrow::execute()',PyTango.ErrSeverity.ERR)
            elif (last_sever == PyTango.ErrSeverity.PANIC):
                PyTango.Except.re_throw_exception(e,in_data[1][nb_except-1],'This is a test','IOReThrow::execute()',PyTango.ErrSeverity.PANIC)

#------------------------------------------------------------------

    def is_IODevByName_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IODevByName(self,in_data):
        try:
            print "[IODevByName::execute] received dev name",in_data
            U = PyTango.Util.instance()
            dev = U.get_device_by_name(in_data)
            self.name = dev.get_name()
            return self.name

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IODServDevice_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IODServDevice(self):
        try:
            print "[IODServDevice::execute]"
            U = PyTango.Util.instance()
            dev = U.get_dserver_device()
            self.name = dev.get_name()
            return self.name

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IODevListByClass_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IODevListByClass(self,in_data):
        try:
            print "[IODevListByClass::execute] received class name",in_data
            U = PyTango.Util.instance()
            dev_list = U.get_device_list_by_class(in_data)
            print dev_list.size()," devices in class", in_data
            name = dev_list.at(0).get_name()
            self.ret_name = name[:string.rfind(name,'/')]
            return self.ret_name

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def IOInitWAttr(self):
        self.dev_m_attr = self.get_device_attr()
        self.w_attr = self.dev_m_attr.get_w_attr_by_name("Short_attr_w")
        self.w_attr.set_write_value(10)

        self.w_attr = self.dev_m_attr.get_w_attr_by_name("Long_attr_w")
        self.w_attr.set_write_value(100);

        self.w_attr = self.dev_m_attr.get_w_attr_by_name("Double_attr_w")
        self.w_attr.set_write_value(1.1);

        self.w_attr = self.dev_m_attr.get_w_attr_by_name("String_attr_w")
        self.w_attr.set_write_value("Init");

#------------------------------------------------------------------

    def is_IOSetAttr_allowed(self):
        if ((self.get_state() == PyTango.DevState.ON) or (self.get_state() == PyTango.DevState.ALARM)):
            return True
        else:
            return False


    def IOSetAttr(self,in_data):
        try:
            print "[IOSetAttr::execute] received number ",in_data
            self.attr_long = in_data

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOAddAttribute_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOAddAttribute(self):
        try:
            print "[IOAddAttribute::execute] "
            at = PyTango.Attr("Added_short_attr", PyTango.ArgType.DevShort, PyTango.AttrWriteType.READ)
            def_prop = PyTango.UserDefaultAttrProp()
            def_prop.set_label("Test label");
            def_prop.set_description("Test description");
            def_prop.set_format("Illisible");
            at.set_default_properties(def_prop);
            self.add_attribute(at);

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IORegSig_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IORegSig(self,in_data):
        try:
            print "[IORegSig::execute] received signal number ",in_data
            self.register_signal(in_data)

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------


    def is_IOUnregSig_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOUnregSig(self,in_data):
        try:
            print "[IOUnregSig::execute] received signal number ",in_data
            self.unregister_signal(in_data)

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IORegClassSig_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IORegClassSig(self,in_data):
        try:
            print "[IORegClassSig::execute] received signal number ",in_data
            dc = self.get_device_class()
            dc.register_signal(in_data)

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOUnregClassSig_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOUnregClassSig(self,in_data):
        try:
            print "[IOUnregClassSig::execute] received signal number ",in_data
            ds = self.get_device_class()
            ds.unregister_signal(in_data)

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOStr1_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOStr1(self):
        try:
            out_data = "Hello from IOStr1"
            self.y = out_data
            return out_data;

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOStr2_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOStr2(self):
        try:
            out_data = "Hello from IOStr2"
            self.y = out_data
            return out_data;

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOExcept_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOExcept(self):
        PyTango.Except.throw_exception("API_ThrowException","This is a test ","IOExcept::execute()")

#------------------------------------------------------------------

    def IOStartPoll(self):
        return 11

#------------------------------------------------------------------

    def is_IOPollStr1_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOPollStr1(self):
        try:
            self.num_cmd1 += 1
            if ((self.num_cmd1 % 2) == 0):
                return "Even value from IOPollStr1"
            else:
                return "Odd value from IOPollStr1"

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOPollArray2_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOPollArray2(self):
        try:
            self.num_cmd2 += 1
            if ((self.num_cmd2 % 2) == 0):
                self.attr_short_array = [100,200]
            else:
                self.attr_short_array = [300,400]
                return self.attr_short_array

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise


#------------------------------------------------------------------

    def is_IOArray1_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOArray1(self):
        try:
            self.ioarray1 = [10,20,30,40]
            return self.ioarray1

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def IOTrigPoll(self,in_data):
        print "In IOTrigPoll command"
        U = PyTango.Util.instance()
        U.trigger_cmd_polling(self,in_data)

#------------------------------------------------------------------

    def IOAttrTrigPoll(self,in_data):
        print "In IOAttrTrigPoll command"
        U = PyTango.Util.instance()
        U.trigger_attr_polling(self,in_data)

#------------------------------------------------------------------

    def IOShortSleep(self,in_data):
        time.sleep(in_data[1])
        return in_data[0] * 2

#------------------------------------------------------------------

    def IOSleepExcept(self,in_data):
        time.sleep(in_data)
        PyTango.Except.throw_exception("aaa","This is a test ","IOSleepExcept::execute()")

#------------------------------------------------------------------

    def IOAttrThrowEx(self,in_data):
        if (in_data[0] == 0):
            if (in_data[1] == 0):
                self.Short_attr_except = False
            else:
                self.Short_attr_except = True
        elif (in_data[0] == 1):
            if (in_data[1] == 0):
                self.event_change_attr_except = False
            else:
                self.event_change_attr_except = True
        elif (in_data[0] == 2):
            if (in_data[1] == 0):
                self.event_quality_attr_except = False
            else:
                self.event_quality_attr_except = True
        elif (in_data[0] == 3):
            if (in_data[1] == 0):
                self.event_throw_out_of_sync = False
            else:
                self.event_throw_out_of_sync = True

#------------------------------------------------------------------

    def IOAddOneElt(self):
        self.attr_event_size += 1

#------------------------------------------------------------------

    def IORemoveOneElt(self):
        self.attr_event_size -= 1

#------------------------------------------------------------------

    def IOIncValue(self):
        self.attr_event[2] += 1

#------------------------------------------------------------------

    def IODecValue(self):
        self.attr_event[2] -= 1

#------------------------------------------------------------------

    def IOChangeQuality(self,in_data):
        if (in_data == 0):
            self.attr_event_qua = PyTango.AttrQuality.ATTR_VALID
        elif (in_data == 1):
            self.attr_event_qua = PyTango.AttrQuality.ATTR_INVALID
        elif (in_data == 2):
            self.attr_event_qua = PyTango.AttrQuality.ATTR_ALARM
        elif (in_data == 3):
            self.attr_event_qua = PyTango.AttrQuality.ATTR_CHANGING

#------------------------------------------------------------------

    def IOPushEvent(self):
        print "[DevTest::IOPushEvent] received"
#        ex = ({'reason':"Pourqoi pas",'desc':"N'importe quoi",'origin':'ici','severity':'ERROR'})
#        self.push_event("event_change_tst",ex)
        self.push_event("event_change_tst",self.attr_event,self.attr_event_size)

#------------------------------------------------------------------

    def is_IOLong64_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOLong64(self,in_data):
        print "[IOLong64::execute] received number",in_data," type = ",type(in_data)
        in_data = in_data * 2;
        print "[IOLong64::execute] return number",in_data
        return in_data;

#------------------------------------------------------------------

    def is_IOULong64_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOULong64(self,in_data):
        print "[IOULong64::execute] received number",in_data," type = ",type(in_data)
        in_data = in_data * 2;
        print "[IOULong64::execute] return number",in_data
        return in_data;

#------------------------------------------------------------------

    def is_IOLong64Array_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOLong64Array(self,in_data):
        l = range(len(in_data));
        for i in l:
            print "[IOLong64Array::execute] received number",in_data[i]
            in_data[i] = in_data[i] * 2
            print "[IOLong64Array::execute] return number",in_data[i]
        return in_data

#------------------------------------------------------------------

    def is_IOULong64Array_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOULong64Array(self,in_data):
        l = range(len(in_data));
        for i in l:
            print "[IOULong64Array::execute] received number",in_data[i]
            in_data[i] = in_data[i] * 2
            print "[IOULong64Array::execute] return number",in_data[i]
        return in_data

#------------------------------------------------------------------

#    def IOChangeStatus(self):
#        print "[DevTest::IOChangeStatus] received"
#        sta = self.get_status()
#        print "Status =",sta
#        sta = sta + "Hola, que tal ?"
#        self.set_status(sta)

#------------------------------------------------------------------

    def is_IOSleep_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSleep(self,in_data):
        print "[IOSleep::execute] sleeping time ",in_data
        time.sleep(in_data)

#------------------------------------------------------------------

    def is_IOArray2_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOArray2(self):
        try:
            self.attr_short_array = [100,200]
            return self.attr_short_array

        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOStrArray_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOStrArray(self):
        dev_string_array = ['First string from dev_string_array','Second string from dev_string_array']
        return dev_string_array

#------------------------------------------------------------------

    def is_IOStruct_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOStruct(self):
        dev_struct = [[1000,2000],['First string from dev_struct','Second string from dev_struct']]
        return dev_struct

#------------------------------------------------------------------

    def is_IORemoveAttribute_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IORemoveAttribute(self):
        try:
            print( '[IORemoveAttribute::execute] ')
            self.remove_attribute('Added_short_attr')
        
        except PyTango.DevFailed,e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def is_IOSeqVecChar_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecChar(self,in_data):
        print('[IOSeqVecChar::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def is_IOSeqVecShort_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecShort(self,in_data):
        print('[IOSeqVecShort::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def is_IOSeqVecLong_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecLong(self,in_data):
        print('[IOSeqVecLong::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def is_IOSeqVecFloat_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecFloat(self,in_data):
        print('[IOSeqVecFloat::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def is_IOSeqVecDouble_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecDouble(self,in_data):
        print('[IOSeqVecDouble::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def is_IOSeqVecUShort_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecUShort(self,in_data):
        print('[IOSeqVecUShort::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def is_IOSeqVecULong_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecULong(self,in_data):
        print('[IOSeqVecULong::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def is_IOSeqVecString_allowed(self):
        if (self.get_state() == PyTango.DevState.ON):
            return True
        else:
            return False

    def IOSeqVecString(self,in_data):
        print('[IOSeqVecString::execute] ' + str(len(in_data)) + ' elt(s) in temp vector')
        return in_data

#------------------------------------------------------------------

    def IOExit(self):
        os._exit(0)

#------------------------------------------------------------------

    def IOPushDevEncodedEvent(self):
        try:
            print('[DevTest::IOPushDevEncodedEvent] received ')
            self.push_event('Encoded_attr','','',*self.enc_attr)
            
        except PyTango.DevFailed as e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def IOSubscribeEvent(self):
        try:
            print('[DevTest::IOSubscribeEvent] received ')
            if self.remote_dev is None:
                devName = self.get_name()
                newName =  devName[:devName.rfind('/')] + '/20'
                self.remote_dev = PyTango.DeviceProxy(newName)
            attr_name = 'short_attr'
            self.cb.cb_executed = 0
            self.remote_dev.poll_attribute(attr_name, 1000)
            self.eve_id = self.remote_dev.subscribe_event(attr_name, PyTango.EventType.PERIODIC_EVENT,self.cb)
        
        except PyTango.DevFailed as e:
            PyTango.Except.print_exception(e)
            raise
#        U = PyTango.Util.instance()
#        devList = U.get_device_list_by_class('Starter')

#------------------------------------------------------------------

    def IOUnSubscribeEvent(self):
        try:
            print('[DevTest::IOUnSubscribeEvent] received ')
            if self.eve_id != 0:
                self.remote_dev.unsubscribe_event(self.eve_id)
                self.remote_dev.stop_poll_attribute('short_attr')
        
        except PyTango.DevFailed as e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def IOGetCbExecuted(self):
        print('[IOGetCbExecuted::execute] received number ')
        return self.cb.cb_executed

#------------------------------------------------------------------

    def is_OLong_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False
        
    def OLong(self):
        theNumber = 22
        print('[OLong::execute] return number ' + str(theNumber)) 
        return theNumber

#------------------------------------------------------------------

    def is_OULong_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False
        
    def OULong(self):
        theNumber = 333
        print('[OULong::execute] return number ' + str(theNumber)) 
        return theNumber

#------------------------------------------------------------------

    def is_OLongArray_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False
        
    def OLongArray(self):
        theReturnedArray = range(555,559)
        for i in theReturnedArray:
            print('[OLongArray::execute] return number ' + str(i)) 
        return theReturnedArray

#------------------------------------------------------------------

    def is_OULongArray_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False
        
    def OULongArray(self):
        theReturnedArray = range(777,780)
        for i in theReturnedArray:
            print('[OULongArray::execute] return number ' + str(i)) 
        return theReturnedArray

#------------------------------------------------------------------

    def is_OLongString_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False
        
    def OLongString(self):
        lvalue = range(999,1005)
        svalue = ['Hola todos'] * 1
        theReturned = (lvalue,svalue)
        for i in lvalue:
            print('[OLongString::execute] return number ' + str(i))
        for i in svalue:
            print('[OLongString::execute] return string ' + str(i))
        return theReturned

#------------------------------------------------------------------

    def GetLongSize(self):
        size = 32
        if PyTango.constants.TANGO_LONG64 == True:
            size = 64
        return size

#------------------------------------------------------------------

    def IOSetWAttrLimit(self,in_data):
        try:
            w_attr = self.get_device_attr().get_w_attr_by_name('Double_attr_w')
            limit = in_data[1]
            if in_data[0] == 0.0:
                w_attr.set_min_value(limit)
            else:
                w_attr.set_max_value(limit)
                
        except PyTango.DevFailed as e:
            PyTango.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def ChangeEncodedFormat(self):
        if self.enc_format_toggle == False:
            self.enc_attr[0] = 'This format'
            self.enc_format_toggle = True
        else:
            self.enc_attr[0] = 'Another format'
            self.enc_format_toggle = False
            
#------------------------------------------------------------------

    def ChangeEncodedData(self):
        data = bytearray(self.enc_attr[1])
        data[2] += 1
        self.enc_attr[1] = data
#        self.enc_attr[1][2] = chr(ord(self.enc_attr[1][2]) + 1).encode('utf-8')
            
#------------------------------------------------------------------

    def PushDataReady(self,in_data):
        try:
            print('Pushing Data Ready event for attribute ' + in_data[1][0])
            self.push_data_ready_event(in_data[1][0],in_data[0][0])
            
        except PyTango.DevFailed as e:
            PyTagno.Except.print_exception(e)    
            
#------------------------------------------------------------------

    def SubDeviceThread(self):
        try:
            tg = PyTango.Util.instance()
            deviceList = tg.get_device_list_by_class('DevTest')
            if len(deviceList) > 1:
                deviceList = sorted([device.get_name() for device in deviceList])
                # choose the 2nd device as a Sub Device
                print('Thread : Connect device = ' + deviceList[1])
                remote_device = PyTango.DeviceProxy(deviceList[1])
                
        except PyTango.DevFailed:
            pass
        

    def SubDeviceTst(self):
        try:
            t = threading.Thread(target=self.SubDeviceThread)
            t.start()
            
            connected = False
            tg = PyTango.Util.instance()
            deviceList = tg.get_device_list_by_class('DevTest')
            if len(deviceList) > 2:
                deviceList = sorted([device.get_name() for device in deviceList])
                # choose the 3rd device as a Sub Device
                remote_device = PyTango.DeviceProxy(deviceList[2])
                connected = True
            return connected
                
        except PyTango.DevFailed as e:
            PyTagno.Except.print_exception(e)
            raise
            
#------------------------------------------------------------------

    def is_IOEncoded_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False

    def IOEncoded(self,in_data):
        try:
            print('[IOEncoded::execute] received string ' + in_data[0])
            data = bytearray()
            for nb in bytearray(in_data[1]):
                print('[IOEncoded::execute] received number ' + str(nb))
                data += chr(nb*2)
                print('[IOEncoded::execute] returned number ' + str(data[-1]))
            return ['Returned string', data]
                
        except PyTango.DevFailed as e:
            PyTagno.Except.print_exception(e)
            raise
            
#------------------------------------------------------------------

    def is_OEncoded_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False

    def OEncoded(self):
        self.encoded_cmd_ctr += 1
        if self.encoded_cmd_ctr % 2 == 0:
            data = bytearray([11,21])
            theReturned = ['Odd - OEncoded format',data]
        else:
            data = bytearray([10,20,30,40])
            theReturned = ['Even - OEncoded format',data]
        return theReturned
            
#------------------------------------------------------------------

    def is_SetGetAlarms_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False
        
    def _setAlarms(self,name,min_alarm,min_warning,max_warning,max_alarm,result):
        attr = self.get_device_attr().get_attr_by_name(name)
        conf = attr.get_properties_3()
            
        try:
            attr.set_min_alarm(min_alarm)
            attr.set_min_warning(min_warning)
            attr.set_max_warning(max_warning)
            attr.set_max_alarm(max_alarm)
            
            result.append(attr.get_name())
            result.append(str(attr.get_min_alarm()))
            result.append(str(attr.get_min_warning()))
            result.append(str(attr.get_max_warning()))
            result.append(str(attr.get_max_alarm()))
            
            attr.set_upd_properties(conf)
        
        except PyTango.DevFailed as e:
            attr.set_upd_properties(conf)
            raise
    
    def SetGetAlarms(self):
        try:
            print('[SetGetAlarms::execute]')
            alarmsReturn = []
            self._setAlarms('Double_attr',-999.99,-888.88,888.88,999.99,alarmsReturn)
            self._setAlarms('Float_attr',-777.77,-666.66,666.66,777.77,alarmsReturn)
            self._setAlarms('Long_attr',1000,1100,1400,1500,alarmsReturn)
            self._setAlarms('Long64_attr',-90000,-80000,80000,90000,alarmsReturn)
            self._setAlarms('Short_attr',-5000,-4000,4000,5000,alarmsReturn)
            self._setAlarms('UChar_attr',1,2,230,240,alarmsReturn)
            self._setAlarms('ULong_attr',1,2,666666,777777,alarmsReturn)
            self._setAlarms('ULong64_attr',1,2,77777777,88888888,alarmsReturn)
            self._setAlarms('UShort_attr',1,2,20000,30000,alarmsReturn)
            return alarmsReturn
        
        except PyTango.DevFailed as e:
            print('Exception while setting alarms')
            PyTango.Except.print_exception(e)
            raise
            
#------------------------------------------------------------------

    def is_SetGetRanges_allowed(self):
        if self.get_state() == PyTango.DevState.ON:
            return True
        else:
            return False
        
    def _setRanges(self,name,min_value,max_value,result):
        wattr = self.get_device_attr().get_attr_by_name(name)
        conf = wattr.get_properties_3()
            
        try:
            wattr.set_min_value(min_value)
            wattr.set_max_value(max_value)
            
            result.append(wattr.get_name())
            result.append(str(wattr.get_min_value()))
            result.append(str(wattr.get_max_value()))
            
            wattr.set_upd_properties(conf)
        
        except PyTango.DevFailed as e:
            wattr.set_upd_properties(conf)
            raise
    
    def SetGetRanges(self):
        try:
            print('[SetGetRanges::execute]')
            rangesReturn = []
            self._setRanges('Double_attr_w',-1111.11,1111.11,rangesReturn)
            self._setRanges('Float_attr_w',-888.88,888.88,rangesReturn)
            self._setRanges('Long_attr_w',900,1600,rangesReturn)
            self._setRanges('Long64_attr_rw',-100000,100000,rangesReturn)
            self._setRanges('Short_attr_w',-6000,6000,rangesReturn)
            self._setRanges('UChar_attr_w',0,250,rangesReturn)
            self._setRanges('ULong_attr_rw',0,888888,rangesReturn)
            self._setRanges('ULong64_attr_rw',0,99999999,rangesReturn)
            self._setRanges('UShort_attr_w',0,40000,rangesReturn)
            return rangesReturn
        
        except PyTango.DevFailed as e:
            print('Exception while setting ranges')
            PyTango.Except.print_exception(e)
            raise
            
#------------------------------------------------------------------

    def IOPollingInDevice(self):
        try:
            ret = []
            att_name = 'Double_spec_attr'
            cmd_name = 'OULong'
            
            # is_xxx_polled()
            ss = 'Attribute ' + att_name + ' polled = '
            if self.is_attribute_polled(att_name) == True:
                ss += 'true'
            else:
                ss += 'false'
            ret.append(ss)
            ss = 'Command ' + cmd_name + ' polled = '
            if self.is_command_polled(cmd_name) == True:
                ss += 'true'
            else:
                ss += 'false'
            ret.append(ss)
            
            # get_xxx_poll_period()
            ss = 'Attribute ' + att_name + ' polling period = ' + str(self.get_attribute_poll_period(att_name))
            ret.append(ss)
            ss = 'Command ' + cmd_name + ' polling period = ' + str(self.get_command_poll_period(cmd_name))
            ret.append(ss)
            
            # poll_xxx()
            self.poll_attribute(att_name,250)
            self.poll_command(cmd_name,250)
            time.sleep(1)
            
            # is_xxx_polled()
            ss = 'Attribute ' + att_name + ' polled = '
            if self.is_attribute_polled(att_name) == True:
                ss += 'true'
            else:
                ss += 'false'
            ret.append(ss)
            ss = 'Command ' + cmd_name + ' polled = '
            if self.is_command_polled(cmd_name) == True:
                ss += 'true'
            else:
                ss += 'false'
            ret.append(ss)
            
            # get_xxx_poll_period()
            ss = 'Attribute ' + att_name + ' polling period = ' + str(self.get_attribute_poll_period(att_name))
            ret.append(ss)
            ss = 'Command ' + cmd_name + ' polling period = ' + str(self.get_command_poll_period(cmd_name))
            ret.append(ss)
            
            # stop_poll_xxx()
            self.stop_poll_attribute(att_name)
            self.stop_poll_command(cmd_name)
            time.sleep(1)
            
            # is_xxx_polled()
            ss = 'Attribute ' + att_name + ' polled = '
            if self.is_attribute_polled(att_name) == True:
                ss += 'true'
            else:
                ss += 'false'
            ret.append(ss)
            ss = 'Command ' + cmd_name + ' polled = '
            if self.is_command_polled(cmd_name) == True:
                ss += 'true'
            else:
                ss += 'false'
            ret.append(ss)
            
            # get_xxx_poll_period()
            ss = 'Attribute ' + att_name + ' polling period = ' + str(self.get_attribute_poll_period(att_name))
            ret.append(ss)
            ss = 'Command ' + cmd_name + ' polling period = ' + str(self.get_command_poll_period(cmd_name))
            ret.append(ss)
                
            return ret
        
        except PyTango.DevFailed as e:
            PyTango.Except.print_exception(e)
            raise
            
#------------------------------------------------------------------






#------------------------------------------------------------------
#------------------------------------------------------------------
#
#                READING THE ATTRIBUTES
#
#------------------------------------------------------------------
#------------------------------------------------------------------



    def read_attr_hardware(self,data):
        print 'In read_attr_hardware'

#------------------------------------------------------------------

    def read_Short_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Short_attr"
        self.attr_short = 12;
        if (self.Short_attr_except == False):
            the_att.set_value(self.attr_short)
        else:
            PyTango.Except.throw_exception("aaa","This is a test","DevTest::read_attr")

#------------------------------------------------------------------

    def read_Long_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Long_attr"
        the_att.set_value(self.attr_long)

#------------------------------------------------------------------

    def read_Double_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Double_attr"
        self.attr_double = 3.2;
        the_att.set_value(self.attr_double)

#------------------------------------------------------------------

    def read_String_attr(self,the_att):
        print "[DevTest::read_attr] attribute name String_attr"
        the_att.set_value("test_string");

#------------------------------------------------------------------

    def read_Short_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Short_spec_attr"
        self.attr_short_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        self.attr_short_array[0] = 10
        self.attr_short_array[1] = 20
        self.attr_short_array[2] = 30
        self.attr_short_array[3] = 40
        the_att.set_value(self.attr_short_array,4)

#------------------------------------------------------------------

    def read_Long_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Long_spec_attr"
        self.attr_long_array[0] = 0
        self.attr_long_array[1] = 1
        self.attr_long_array[2] = 2
        self.attr_long_array[3] = 3
        self.attr_long_array[4] = 4
        self.attr_long_array[5] = 5
        self.attr_long_array[6] = 6
        self.attr_long_array[7] = 7
        self.attr_long_array[8] = 8
        self.attr_long_array[9] = 9
        the_att.set_value(self.attr_long_array,10)

#------------------------------------------------------------------

    def read_Double_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Double_spec_attr"
        self.attr_db_array[0] = 1.11
        self.attr_db_array[1] = 2.22
        the_att.set_value(self.attr_db_array,2)

#------------------------------------------------------------------

    def read_String_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name String_spec_attr"
        self.attr_str_array = ['Hello world','Hello universe']
        the_att.set_value(self.attr_str_array,2)

#------------------------------------------------------------------

    def read_Short_ima_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Short_ima_attr"
        self.attr_short_array_1 = [40,60,80,100]
        the_att.set_value(self.attr_short_array_1,2,2)

#------------------------------------------------------------------

    def read_Long_ima_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Long_ima_attr"
        self.attr_long_array = [0,1,2,3,4,5]
        the_att.set_value(self.attr_long_array,3,2)

#------------------------------------------------------------------

    def read_Double_ima_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Double_ima_attr"
        self.attr_db_array = [5.55,6.66]
        the_att.set_value(self.attr_db_array,2,1)

#------------------------------------------------------------------

    def read_String_ima_attr(self,the_att):
        print "[DevTest::read_attr] attribute name String_ima_attr"
        self.attr_str_array = ["Hello milky way","Hello moon"]
        the_att.set_value(self.attr_str_array,1,2)

#------------------------------------------------------------------

    def read_attr_dq_sh(self,the_att):
        print "[DevTest::read_attr] attribute name attr_dq_sh"
        self.attr_dq_short = 77;
        the_att.set_value_date_quality(self.attr_dq_short,time.time(),PyTango.AttrQuality.ATTR_VALID);

#------------------------------------------------------------------

    def read_attr_dq_lo(self,the_att):
        print "[DevTest::read_attr] attribute name attr_dq_lo"
        self.attr_dq_long = 7777;
        the_att.set_value_date_quality(self.attr_dq_long,time.time(),PyTango.AttrQuality.ATTR_ALARM);

#------------------------------------------------------------------

    def read_attr_dq_db(self,the_att):
        print "[DevTest::read_attr] attribute name attr_dq_db"
        self.attr_dq_double = 8.888;
        the_att.set_value_date_quality(self.attr_dq_double,time.time(),PyTango.AttrQuality.ATTR_VALID);

#------------------------------------------------------------------

    def read_attr_dq_str(self,the_att):
        print "[DevTest::read_attr] attribute name attr_dq_str"
        self.attr_dq_str = 'Setting value date and quality'
        the_att.set_value_date_quality(self.attr_dq_str,time.time(),PyTango.AttrQuality.ATTR_ALARM);

#------------------------------------------------------------------

    def read_attr_no_data(self,the_att):
        print "[DevTest::read_attr] attribute name attr_no_data"
        the_att.set_quality(PyTango.AttrQuality.ATTR_VALID);

#------------------------------------------------------------------

    def read_attr_wrong_type(self,the_att):
        print "[DevTest::read_attr] attribute name attr_wrong_type"
        self.attr_long = 1246;
        the_att.set_value(self.attr_long);

#------------------------------------------------------------------

    def read_attr_wrong_size(self,the_att):
        print "[DevTest::read_attr] attribute name attr_wrong_size"
        the_att.set_value(self.attr_long_array,1000,1000);

#------------------------------------------------------------------

    def read_attr_no_alarm(self,the_att):
        print "[DevTest::read_attr] attribute name attr_no_alarm"
        self.attr_long = 1246;
        the_att.check_alarm();
        the_att.set_value(self.attr_long)

#------------------------------------------------------------------

    def read_Long_attr_with_w(self,the_att):
        print "[DevTest::read_attr] attribute name Long_attr_with_w"
        self.attr_long = 1246;
        the_att.set_value(self.attr_long)

#------------------------------------------------------------------

    def read_Long_attr_with_w2(self,the_att):
        print "[DevTest::read_attr] attribute name Long_attr_with_w2"
        the_att.set_value(self.attr_long_rw2)

#------------------------------------------------------------------

    def read_Short_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Short_attr_rw"
        the_att.set_value(self.attr_short_rw)

#------------------------------------------------------------------

    def read_Float_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Float_attr"
        the_att.set_value(4.5)

#------------------------------------------------------------------

    def read_Boolean_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Boolean_attr"
        the_att.set_value(True)

#------------------------------------------------------------------

    def read_UShort_attr(self,the_att):
        print "[DevTest::read_attr] attribute name UShort_attr"
        the_att.set_value(111)

#------------------------------------------------------------------

    def read_UChar_attr(self,the_att):
        print "[DevTest::read_attr] attribute name UChar_attr"
        the_att.set_value(88)

#------------------------------------------------------------------

    def read_Float_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Float_spec_attr"
        self.attr_spec_float = [4.5,8.5,16.5]
        the_att.set_value(self.attr_spec_float)

#------------------------------------------------------------------

    def read_Boolean_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Boolean_spec_attr"
        self.attr_spec_boolean = [True,True,False,True,True]
        the_att.set_value(self.attr_spec_boolean)

#------------------------------------------------------------------

    def read_UShort_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name UShort_spec_attr"
        self.attr_spec_ushort = [333,444]
        the_att.set_value(self.attr_spec_ushort)

#------------------------------------------------------------------

    def read_UChar_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name UChar_spec_attr"
        self.attr_spec_uchar = [28,45,156,34,200,12]
        the_att.set_value(self.attr_spec_uchar)

#------------------------------------------------------------------

    def read_PollLong_attr(self,the_att):
        print "[DevTest::read_attr] attribute name PollLong_attr"
        self.PollLong_attr_num += 1
        if ((self.PollLong_attr_num % 2) == 0):
            self.Poll_attr_long = 5555
        else:
            self.Poll_attr_long = 6666
        the_att.set_value(self.Poll_attr_long)

#------------------------------------------------------------------

    def read_PollString_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name PollString_spec_attr"
        self.PollString_spec_attr_num += 1
        if ((self.PollString_spec_attr_num % 2) == 0):
            self.attr_str_array = ["Hello world","Hello universe"]
        else:
            self.attr_str_array = ["Hello Grenoble","Hello Paris"]
        the_att.set_value(self.attr_str_array)

#------------------------------------------------------------------

    def read_Short_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Short_spec_attr_rw"
        self.short_spec_attr = [8,9]
        the_att.set_value(self.short_spec_attr)

#------------------------------------------------------------------

    def read_String_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Short_spec_attr_rw"
        self.string_spec_attr = ["Thank's god","It's friday"]
        the_att.set_value(self.string_spec_attr)

#------------------------------------------------------------------

    def read_Float_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Float_spec_attr_rw"
        self.attr_spec_float = [5.5,11.5]
        the_att.set_value(self.attr_spec_float)

#------------------------------------------------------------------

    def read_UChar_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name UChar_spec_attr_rw"
        self.attr_spec_uchar = [22,44,66]
        the_att.set_value(self.attr_spec_uchar)

#------------------------------------------------------------------

    def read_Short_ima_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Short_ima_attr_rw"
        self.short_ima_attr = [6,7,8,9]
        the_att.set_value(self.short_ima_attr,2,2)

#------------------------------------------------------------------

    def read_String_ima_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name String_ima_attr_rw"
        self.string_ima_attr = ["Alors la, pour une surprise","c'est une surprise"]
        the_att.set_value(self.string_ima_attr,2,1)

#------------------------------------------------------------------

    def read_Boolean_ima_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Boolean_ima_attr_rw"
        self.attr_spec_boolean = [True,False]
        the_att.set_value(self.attr_spec_boolean,2,1)

#------------------------------------------------------------------

    def read_UShort_ima_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name UShort_ima_attr_rw"
        self.attr_spec_ushort = [2,3]
        the_att.set_value(self.attr_spec_ushort,2,1)

#------------------------------------------------------------------

    def read_Poll_buff(self,the_att):
        print "[DevTest::read_attr] attribute name Poll_buff"

#------------------------------------------------------------------

    def read_Poll_buffRW(self,the_att):
        print "[DevTest::read_attr] attribute name Poll_buffRW"

#------------------------------------------------------------------

    def read_attr_asyn(self,the_att):
        print "[DevTest::read_attr] attribute name attr_asyn"
        time.sleep(2)
        the_att.set_value(5.55)
        print "Leaving reading attr_asyn attribute"

#------------------------------------------------------------------

    def read_attr_asyn_to(self,the_att):
        print "[DevTest::read_attr] attribute name attr_asyn_to"
        time.sleep(4)
        the_att.set_value(5.55)
        print "Leaving reading attr_asyn_to attribute"

#------------------------------------------------------------------

    def read_attr_asyn_except(self,the_att):
        print "[DevTest::read_attr] attribute name attr_asyn_except"
        time.sleep(2)
        print "Leaving reading attr_asyn_except attribute"
        PyTango.Except.throw_exception("aaa","This is a test ","DevTest::read_attr")

#------------------------------------------------------------------

    def read_event_change_tst(self,the_att):
        print "[DevTest::read_attr] attribute name event_change_tst"
        if (self.event_change_attr_except == False):
            if self.event_throw_out_of_sync == True:
                time.sleep(0.4)
            the_att.set_value(self.attr_event,self.attr_event_size)
        else:
            PyTango.Except.throw_exception("bbb","This is a test ","DevTest::read_attr")

#------------------------------------------------------------------

    def read_event64_change_tst(self,the_att):
        print "[DevTest::read_attr] attribute name event64_change_tst"
        if (self.event_change_attr_except == False):
            the_att.set_value(self.attr_event64,self.attr_event64_size)
        else:
            PyTango.Except.throw_exception("bbb64","This is a test ","DevTest::read_attr")

#------------------------------------------------------------------

    def read_event_quality_tst(self,the_att):
        print "[DevTest::read_attr] attribute name event_quality_tst"
        if (self.event_quality_attr_except == False):
            if (self.attr_event_qua != PyTango.AttrQuality.ATTR_INVALID):
                the_att.set_value(self.attr_qua_event,2)
                the_att.set_quality(self.attr_event_qua)
        else:
            PyTango.Except.throw_exception("ccc","This is a test ","DevTest::read_attr")


#------------------------------------------------------------------

    def read_slow_actuator(self,the_att):
        print "[DevTest::read_attr] attribute name slow_actuator"
        self.slow_actua += 1
        the_att.set_value(self.slow_actua)
        ti = time.time()
        if (self.slow_actua_write != 0):
            delta = ti - self.slow_actua_write
            if (delta >= 3):
                the_att.set_quality(PyTango.AttrQuality.ATTR_VALID)
                self.slow_actua_write = 0

#------------------------------------------------------------------

    def read_fast_actuator(self,the_att):
        print "[DevTest::read_attr] attribute name fast_actuator"
        the_att.set_value(self.fast_actua)

#------------------------------------------------------------------

    def read_Long64_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_attr"
        the_att.set_value(9223372036854775807)

#------------------------------------------------------------------

    def read_Long64_spec_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_spec_attr"
        attr_long64_array = [ 0, -9223372036854775808, 9223372036854775807, 0 ]
        the_att.set_value(attr_long64_array,4)

#------------------------------------------------------------------

    def read_Long64_ima_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_ima_attr"
        attr_long64_array = [0, -9223372036854775808, 9223372036854775807, 0, 0, -9223372036854775808, 9223372036854775807, 0]
        the_att.set_value(attr_long64_array,4,2)

#------------------------------------------------------------------

    def read_Long64_attr_w(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_attr_w"
        the_att.set_value(9223372036854775807)

#------------------------------------------------------------------

    def read_Long64_spec_attr_w(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_spec_attr_w"
        attr_long64_array = [ 0, -9223372036854775808, 9223372036854775807, 0 ]
        the_att.set_value(attr_long64_array,4)

#------------------------------------------------------------------

    def read_Long64_ima_attr_w(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_ima_attr_w"
        attr_long64_array = [0, -9223372036854775808, 9223372036854775807, 0, 0, -9223372036854775808, 9223372036854775807, 0]
        the_att.set_value(attr_long64_array,4,2)

#------------------------------------------------------------------

    def read_Long64_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_attr_rw"
        the_att.set_value(9223372036854775807)

#------------------------------------------------------------------

    def read_Long64_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_spec_attr_rw"
        attr_long64_array = [ 0, -9223372036854775808, 9223372036854775807, 0 ]
        the_att.set_value(attr_long64_array,4)

#------------------------------------------------------------------

    def read_Long64_ima_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Long64_ima_attr_rw"
        attr_long64_array = [0, -9223372036854775808, 9223372036854775807, 0, 0, -9223372036854775808, 9223372036854775807, 0]
        the_att.set_value(attr_long64_array,4,2)

#------------------------------------------------------------------

    def read_Long_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name Long_spec_attr_rw"
        the_att.set_value(self.Long_spec_attr_rw)

#------------------------------------------------------------------

    def read_Encoded_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Encoded_attr_rw"
        the_att.set_value(*self.enc_attr)

#------------------------------------------------------------------

    def read_ULong_attr(self,the_att):
        print "[DevTest::read_attr] attribute name ULong_attr"
        # TODO: 'L' suffix no more supported in Python3
        self.attr_ulong = 100L
        the_att.set_value(self.attr_ulong)

#------------------------------------------------------------------

    def read_ULong64_attr(self,the_att):
        print "[DevTest::read_attr] attribute name ULong64_attr"
        # TODO: 'L' suffix no more supported in Python3
        self.attr_ulong64 = 200L
        the_att.set_value(self.attr_ulong64)

#------------------------------------------------------------------

    def read_ULong_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name ULong_attr_rw"
        the_att.set_value(self.attr_ulong_rw)

#------------------------------------------------------------------

    def read_ULong64_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name ULong64_attr_rw"
        # TODO: long() no more supported in Python3
        the_att.set_value(long(self.attr_ulong64_rw))

#------------------------------------------------------------------

    def read_State_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name State_attr_rw"
        the_att.set_value(self.attr_state_rw)

#------------------------------------------------------------------

    def read_ULong_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name ULong_spec_attr_rw"
        the_att.set_value(self.attr_spec_ulong_rw,3)

#------------------------------------------------------------------

    def read_ULong64_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name ULong64_spec_attr_rw"
        # TODO: long() no more supported in Python3, as well as map (returns a pointer)
        the_att.set_value(map(long,self.attr_spec_ulong64_rw),3)

#------------------------------------------------------------------

    def read_State_spec_attr_rw(self,the_att):
        print "[DevTest::read_attr] attribute name State_spec_attr_rw"
        the_att.set_value(self.attr_spec_state_rw,2)

#------------------------------------------------------------------

    def read_Sub_device_tst(self,the_att):
        try:
            print('[DevTest::read_attr] attribute name Sub_device_tst')
            tg = PyTango.Util.instance()
            deviceList = tg.get_device_list_by_class('DevTest')
            if len(deviceList) > 1:
                deviceList = sorted([device.get_name() for device in deviceList])
                try:
                    remote_dev = PyTango.DeviceProxy(deviceList[1])
                    self.attr_sub_device_tst = True
                except:
                    self.attr_sub_device_tst = False
            else:
                self.attr_sub_device_tst = False
            the_att.set_value(self.attr_sub_device_tst)
        
        except PyTango.DevFailed as e:
            PyTagno.Except.print_exception(e)
            raise

#------------------------------------------------------------------

    def read_SlowAttr(self,the_att):
        time.sleep(0.5)
        the_att.set_value(self.attr_slow)

#------------------------------------------------------------------

    def read_Encoded_image(self,the_att):
        print('[DevTest::read_attr] attribute name Encoded_attr_image')
        self.jpeg = PyTango.EncodedAttribute()
        imageData = bytearray(256*256)
        for i in range(0,256):
            for j in range(0,256):
                imageData[i+j*256] = i
        imageData = bytes(imageData)
        self.jpeg.encode_jpeg_gray8(imageData,256,256,50.0)
        the_att.set_value(self.jpeg)

#------------------------------------------------------------------

    def read_DefAttr(self,the_att):
        print('[DevTest::read_attr] attribute name DefAttr')
        the_att.set_value(self.att_conf)

#------------------------------------------------------------------

    def read_DefUserAttr(self,the_att):
        print('[DevTest::read_attr] attribute name DefUserAttr')
        the_att.set_value(self.att_conf)

#------------------------------------------------------------------

    def read_DefClassAttr(self,the_att):
        print('[DevTest::read_attr] attribute name DefClassAttr')
        the_att.set_value(self.att_conf)

#------------------------------------------------------------------

    def read_DefClassUserAttr(self,the_att):
        print('[DevTest::read_attr] attribute name DefClassUserAttr')
        the_att.set_value(self.att_conf)

#------------------------------------------------------------------

    def read_DefUserAttr(self,the_att):
        print('[DevTest::read_attr] attribute name DefUserAttr')
        the_att.set_value(self.att_conf)

#------------------------------------------------------------------
#------------------------------------------------------------------
#
#                WRITING THE ATTRIBUTES
#
#------------------------------------------------------------------
#------------------------------------------------------------------





    def write_Short_attr_w(self,the_att):
        data = the_att.get_write_value()

#------------------------------------------------------------------

    def write_Long_attr_w(self,the_att):
        print "In write_Long_attr_w for attribute",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Float_attr_w(self,the_att):
        print "In write_Float_attr_w for attribute",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_UShort_attr_w(self,the_att):
        print "In write_UShort_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data;

#------------------------------------------------------------------

    def write_UChar_attr_w(self,the_att):
        print "In write_UChar_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Short_attr_rw(self,the_att):
        print "In write_Short_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        self.attr_short_rw = data[0]

#------------------------------------------------------------------

    def write_Long_attr_with_w(self,the_att):
        print "In write_Long_attr_with_w for attribute ",the_att.get_name()

#------------------------------------------------------------------

    def write_Long_attr_with_w2(self,the_att):
        print "In write_Long_attr_with_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        self.attr_long_rw2 = data[0]

#------------------------------------------------------------------

    def write_Short_attr_w2(self,the_att):
        print "In write_Short_attr_w2 for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Double_attr_w(self,the_att):
        print "In write_Double_attr_w for attribute ",the_att.get_name()
        data=[];
        the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_String_attr_w2(self,the_att):
        print "In write_String_attr_w2 for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_String_attr_w(self,the_att):
        print "In write_String_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Boolean_attr_w(self,the_att):
        print "In write_Boolean_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Short_spec_attr_w(self,the_att):
        print "In write_Short_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Short_ima_attr_w(self,the_att):
        print "In write_Short_ima_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Long_spec_attr_w(self,the_att):
        print "In write_Long_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Double_spec_attr_w(self,the_att):
        print "In write_Double_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_String_spec_attr_w(self,the_att):
        print "In write_String_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Float_spec_attr_w(self,the_att):
        print "In write_Float_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Boolean_spec_attr_w(self,the_att):
        print "In write_Boolean_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_UShort_spec_attr_w(self,the_att):
        print "In write_UShort_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_UChar_spec_attr_w(self,the_att):
        print "In write_UChar_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_String_ima_attr_w(self,the_att):
        print "In write_String_ima_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def is_write_Float_ima_attr_w(self,req_type):
        return True

    def write_Float_ima_attr_w(self,the_att):
        print "In write_Float_ima_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_UShort_ima_attr_w(self,the_att):
        print "In write_UShort_ima_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Short_spec_attr_rw(self,the_att):
        print "In write_Short_spec_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_String_spec_attr_rw(self,the_att):
        print "In write_String_spec_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Float_spec_attr_rw(self,the_att):
        print "In write_Float_spec_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_UChar_spec_attr_rw(self,the_att):
        print "In write_UChar_spec_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Short_ima_attr_rw(self,the_att):
        print "In write_Short_ima_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_String_ima_attr_rw(self,the_att):
        print "In write_String_ima_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Boolean_ima_attr_rw(self,the_att):
        print "In write_Boolean_ima_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_UShort_ima_attr_rw(self,the_att):
        print "In write_UShort_ima_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_attr_asyn_write(self,the_att):
        print "In write_attr_asyn_write for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data
        time.sleep(2)

#------------------------------------------------------------------

    def write_attr_asyn_write_to(self,the_att):
        print "In write_attr_asyn_write_to for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data
        time.sleep(4)

#------------------------------------------------------------------

    def write_attr_asyn_write_except(self,the_att):
        print "In write_attr_asyn_write_except for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data
        time.sleep(2)
        PyTango.Except.throw_exception("aaa","This is a test ","DevTest::write_attr_hardware")

#------------------------------------------------------------------

    def write_slow_actuator(self,the_att):
        print "In write_slow_actuator for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        self.slow_actua = data[0]
        self.slow_actua_write = time.time()
        the_att.set_value(self.slow_actua)
        the_att.set_quality(PyTango.AttrQuality.ATTR_CHANGING)

#------------------------------------------------------------------

    def write_fast_actuator(self,the_att):
        print "In write_fast_actuator for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        self.fast_actua = data[0]
        the_att.set_value(self.fast_actua)
        the_att.set_quality(PyTango.AttrQuality.ATTR_CHANGING)
        the_att.set_quality(PyTango.AttrQuality.ATTR_VALID)

#------------------------------------------------------------------

    def write_Long64_attr_w(self,the_att):
        print "In write_Long64_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Long64_spec_attr_w(self,the_att):
        print "In write_Long64_spec_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Long64_ima_attr_w(self,the_att):
        print "In write_Long64_ima_attr_w for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Long64_attr_rw(self,the_att):
        print "In write_Long64_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Long64_spec_attr_rw(self,the_att):
        print "In write_Long64_spec_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

#------------------------------------------------------------------

    def write_Long64_ima_attr_rw(self,the_att):
        print "In write_Long64_ima_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        print "Attribute value = ",data

    def write_Long_spec_attr_rw(self,the_att):
        print "In write_Long_spec_attr_rw for attribute ",the_att.get_name()
        data = the_att.get_write_value()
        self.Long_spec_attr_rw = data
        print "Attribute value = ",data


#------------------------------------------------------------------

    def write_Encoded_attr(self,the_att):
        print('In write_Encoded_attr_rw for attribute ' + the_att.get_name())
        self.enc_attr = the_att.get_write_value()
        print('\tReceived string = ' + self.enc_attr[0])
        print('\tReceived data nb = ' + str(len(self.enc_attr[1])))
        i = 0
        for c in self.enc_attr[1]:
            print('\t\tdata[' + str(i) + '] = ' + c)
            i += 1

#------------------------------------------------------------------

    def write_ULong_attr_rw(self,the_att):
        print('In write_ULong_attr_rw for attribute ' + the_att.get_name())
        self.attr_ulong_rw = the_att.get_write_value()
        if self.attr_ulong_rw > 1000:
            the_att.set_write_value(1111)

#------------------------------------------------------------------

    def write_ULong64_attr_rw(self,the_att):
        print('In write_ULong64_attr_rw for attribute ' + the_att.get_name())
        self.attr_ulong64_rw = the_att.get_write_value()

#------------------------------------------------------------------

    def write_Poll_buffRW(self,the_att):
        print('In write_Poll_buffRW for attribute ' + the_att.get_name())

#------------------------------------------------------------------

    def write_State_attr_rw(self,the_att):
        print('In write_State_attr_rw for attribute ' + the_att.get_name())
        self.attr_state_rw = the_att.get_write_value()

#------------------------------------------------------------------

    def write_ULong_spec_attr_rw(self,the_att):
        print('In write_ULong_spec_attr_rw for attribute ' + the_att.get_name())
        nb_data = the_att.get_write_value_length()
        ulg = the_att.get_write_value()
        print('Received ' + str(nb_data) + ' for attribute ' + the_att.get_name());

#------------------------------------------------------------------

    def write_ULong64_spec_attr_rw(self,the_att):
        print('In write_ULong64_spec_attr_rw for attribute ' + the_att.get_name())
        nb_data = the_att.get_write_value_length()
        ulg64 = the_att.get_write_value()
        print('Received ' + str(nb_data) + ' for attribute ' + the_att.get_name());

#------------------------------------------------------------------

    def write_State_spec_attr_rw(self,the_att):
        print('In write_State_spec_attr_rw for attribute ' + the_att.get_name())
        nb_data = the_att.get_write_value_length()
        sta = the_att.get_write_value()
        print('Received ' + str(nb_data) + ' for attribute ' + the_att.get_name());

#------------------------------------------------------------------

    def write_DefAttr(self,the_att):
        print('In write_DefAttr for attribute ' + the_att.get_name())

#------------------------------------------------------------------

    def write_DefUserAttr(self,the_att):
        print('In write_DefUserAttr for attribute ' + the_att.get_name())

#------------------------------------------------------------------

    def write_DefClassAttr(self,the_att):
        print('In write_DefClassAttr for attribute ' + the_att.get_name())

#------------------------------------------------------------------

    def write_DefClassUserAttr(self,the_att):
        print('In write_DefClassUserAttr for attribute ' + the_att.get_name())

#------------------------------------------------------------------
#------------------------------------------------------------------
#
#                FOR ATTRIBUTES ADDED ON THE FLY
#
#------------------------------------------------------------------
#------------------------------------------------------------------

    def read_Added_short_attr(self,the_att):
        print "[DevTest::read_attr] attribute name Added_short_attr"
        the_att.set_value(5)







class DevTestClass(PyTango.DeviceClass):
    def __init__(self,name):
        PyTango.DeviceClass.__init__(self,name)
        self.set_type("TestDevice")
        print 'In DevTestClass __init__'

        self.add_wiz_dev_prop('aaaa','bbbb','cccc')
        self.add_wiz_class_prop('aa','bb')


    def signal_handler(self,in_sig):
        print "[Class signal handler] received signal number",in_sig," for class",self.get_name()

    cmd_list = {'IOVoid':[[PyTango.ArgType.DevVoid],[PyTango.ArgType.DevVoid]],
                'IOBool':[[PyTango.ArgType.DevBoolean,"Number"],[PyTango.ArgType.DevBoolean,"Not number"]],
                'IOShort':[[PyTango.ArgType.DevShort,"Number"],[PyTango.ArgType.DevShort,"Number * 2"]],
                'IOLong':[[PyTango.ArgType.DevLong,"Number"],[PyTango.ArgType.DevLong,"Number * 2"]],
                'IOFloat':[[PyTango.ArgType.DevFloat,"Number"],[PyTango.ArgType.DevFloat,"Number * 2"]],
                'IODouble':[[PyTango.ArgType.DevDouble,"Number"],[PyTango.ArgType.DevDouble,"Number * 2"]],
                'IOUShort':[[PyTango.ArgType.DevUShort,"Number"],[PyTango.ArgType.DevUShort,"Number * 2"]],
                'IOULong':[[PyTango.ArgType.DevULong,"Number"],[PyTango.ArgType.DevULong,"Number * 2"]],
                'IOString':[[PyTango.ArgType.DevString,"Word"],[PyTango.ArgType.DevString,"the palindrome is"]],
                'IOLong64':[[PyTango.ArgType.DevLong64,"Number"],[PyTango.ArgType.DevLong64,"Number * 2"]],
                'IOULong64':[[PyTango.ArgType.DevULong64,"Number"],[PyTango.ArgType.DevULong64,"Number * 2"]],
                'IOCharArray':[[PyTango.ArgType.DevVarCharArray,"Array of char"],[PyTango.ArgType.DevVarCharArray,"This reversed array"]],
                'IOShortArray':[[PyTango.ArgType.DevVarShortArray,"Array of short"],[PyTango.ArgType.DevVarShortArray,"This array * 2"]],
                'IOLongArray':[[PyTango.ArgType.DevVarLongArray,"Array of long"],[PyTango.ArgType.DevVarLongArray,"This array * 2"]],
                'IOFloatArray':[[PyTango.ArgType.DevVarFloatArray,"Array of float"],[PyTango.ArgType.DevVarFloatArray,"This array * 2"]],
                'IODoubleArray':[[PyTango.ArgType.DevVarDoubleArray,"Array of double"],[PyTango.ArgType.DevVarDoubleArray,"This array * 2"]],
                'IOUShortArray':[[PyTango.ArgType.DevVarUShortArray,"Array of unsigned short"],[PyTango.ArgType.DevVarUShortArray,"This array * 2"]],
                'IOULongArray':[[PyTango.ArgType.DevVarULongArray,"Array of unsigned long"],[PyTango.ArgType.DevVarULongArray,"This array * 2"]],
                'IOStringArray':[[PyTango.ArgType.DevVarStringArray,"Array of string"],[PyTango.ArgType.DevVarStringArray,"This reversed array"]],
                'IOLongString':[[PyTango.ArgType.DevVarLongStringArray,"Array of long and string"],[PyTango.ArgType.DevVarLongStringArray,"This array of long * 2"]],
                'IODoubleString':[[PyTango.ArgType.DevVarDoubleStringArray,"Array of double and string"],[PyTango.ArgType.DevVarDoubleStringArray,"This array of double * 2"]],
                'IOLong64Array':[[PyTango.ArgType.DevVarLong64Array,"Array of long64"],[PyTango.ArgType.DevVarLong64Array,"This array * 2"]],
                'IOLong64Array':[[PyTango.ArgType.DevVarULong64Array,"Array of unsigned long 64"],[PyTango.ArgType.DevVarULong64Array,"This array * 2"]],
                'IOState':[[PyTango.ArgType.DevState,"New device state"],[PyTango.ArgType.DevVoid,"void"]],
                'IOThrow':[[PyTango.ArgType.DevVarLongStringArray,"Error facility"],[PyTango.ArgType.DevVoid,"An exception"]],
                'IOReThrow':[[PyTango.ArgType.DevVarLongStringArray,"Error facility"],[PyTango.ArgType.DevVoid,"An exception"]],
                'IODevByName':[[PyTango.ArgType.DevString,"device name"],[PyTango.ArgType.DevString,"device name (returned by name()"]],
                'IODServDevice':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevString,"dserver device name"]],
                'IODevListByClass':[[PyTango.ArgType.DevString,"class name"],[PyTango.ArgType.DevString,"class first device name"]],
                'IOInitWAttr':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOAddAttribute':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOSetAttr':[[PyTango.ArgType.DevLong,"New attr value"],[PyTango.ArgType.DevVoid,"void"]],
                'IORegSig':[[PyTango.ArgType.DevLong,"Signal number"],[PyTango.ArgType.DevVoid,"void"]],
                'IOUnregSig':[[PyTango.ArgType.DevLong,"Signal number"],[PyTango.ArgType.DevVoid,"void"]],
                'IORegClassSig':[[PyTango.ArgType.DevLong,"Signal number"],[PyTango.ArgType.DevVoid,"void"]],
                'IOUnregClassSig':[[PyTango.ArgType.DevLong,"Signal number"],[PyTango.ArgType.DevVoid,"void"]],
                'IOStr1':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevString,"Just a string dynamically allocated"]],
                'IOStr2':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevString,"A constant string"]],
                'IOExcept':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"An exception"]],
                'IOStartPoll':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevLong,"Constant number set to 11"]],
                'IOPollStr1':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevString,"Just a string changing at each call"]],
                'IOPollArray2':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarShortArray,"An array changing at each call"]],
                'IOArray1':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarLongArray,"An array allocated"]],
                'IOTrigPoll':[[PyTango.ArgType.DevString,"Command's name to trig polling"],[PyTango.ArgType.DevVoid,"void"]],
                'IOAttrTrigPoll':[[PyTango.ArgType.DevString,"Attribute's name to trig polling"],[PyTango.ArgType.DevVoid,"void"]],
                'IOShortSleep':[[PyTango.ArgType.DevVarShortArray,"Input short and sleeping time in sec"],[PyTango.ArgType.DevShort,"Output short (in * 2)"]],
                'IOSleepExcept':[[PyTango.ArgType.DevShort,"Sleep time (sec)"],[PyTango.ArgType.DevVoid,"An exception"]],
                'IOAttrThrowEx':[[PyTango.ArgType.DevVarShortArray,"2 elts : Attr code and throw except flag"],[PyTango.ArgType.DevVoid,"void"]],
                'IOAddOneElt':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IORemoveOneElt':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOIncValue':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IODecValue':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOChangeQuality':[[PyTango.ArgType.DevShort,"0->VALID, 1->INVALID, 2->ALARM, 3->CHANGING"],[PyTango.ArgType.DevVoid,"void"]],
                'IOPushEvent':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOSleep':[[PyTango.ArgType.DevUShort,"sleeping time"],[PyTango.ArgType.DevVoid,"void"]],
                'IOArray2':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarShortArray,"An array without copying"]],
                'IOStrArray':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarStringArray,"A string array"]],
                'IOStruct':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarLongStringArray,"A structure type"]],
                'IORemoveAttribute':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOSeqVecChar':[[PyTango.ArgType.DevVarCharArray,"Input char array"],[PyTango.ArgType.DevVarCharArray,"Output char array"]],
                'IOSeqVecShort':[[PyTango.ArgType.DevVarShortArray,"Input short array"],[PyTango.ArgType.DevVarShortArray,"Output short array"]],
                'IOSeqVecLong':[[PyTango.ArgType.DevVarLongArray,"Input long array"],[PyTango.ArgType.DevVarLongArray,"Output long array"]],
                'IOSeqVecFloat':[[PyTango.ArgType.DevVarFloatArray,"Input float array"],[PyTango.ArgType.DevVarFloatArray,"Output float array"]],
                'IOSeqVecDouble':[[PyTango.ArgType.DevVarDoubleArray,"Input double array"],[PyTango.ArgType.DevVarDoubleArray,"Output double array"]],
                'IOSeqVecUShort':[[PyTango.ArgType.DevVarUShortArray,"Input unsigned short array"],[PyTango.ArgType.DevVarUShortArray,"Output unsigned short array"]],
                'IOSeqVecULong':[[PyTango.ArgType.DevVarULongArray,"Input unsigned long array"],[PyTango.ArgType.DevVarULongArray,"Output unsigned long array"]],
                'IOSeqVecString':[[PyTango.ArgType.DevVarStringArray,"Input string array"],[PyTango.ArgType.DevVarStringArray,"Output string array"]],
                'IOExit':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOPushDevEncodedEvent':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOSubscribeEvent':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOUnSubscribeEvent':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'IOGetCbExecuted':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevLong,"Number of times the CB has been executed"]],
                'OLong':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevLong,"A long"]],
                'OULong':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevULong,"An unsigned long"]],
                'OLongArray':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarLongArray,"A long array"]],
                'OULongArray':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarULongArray,"A unsigned long array"]],
                'OLongString':[[PyTango.ArgType.DevVoid,"void"],[PyTango.CmdArgType.DevVarLongStringArray,"A unsigned long and string array"]],
                'GetLongSize':[[PyTango.ArgType.DevVoid,"void"],[PyTango.CmdArgType.DevShort,"Sizeof long (32 or 64 bits)"]],
                'IOSetWAttrLimit':[[PyTango.ArgType.DevVarDoubleArray,"arr[0]==0 -> min_value, arr[0]==1 -> max_value, arr[1]=limit"],[PyTango.CmdArgType.DevVoid,"void"]],
                'ChangeEncodedFormat':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'ChangeEncodedData':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]],
                'PushDataReady':[[PyTango.ArgType.DevVarLongStringArray,"s[0] = attribute name, l[0] = user counter"],[PyTango.ArgType.DevVoid,"void"]],
                'SubDeviceTst':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevBoolean,"true = sub device connected"]],
                'IOEncoded':[[PyTango.ArgType.DevEncoded,"DevEncoded structure"],[PyTango.ArgType.DevEncoded,"DevEncoded structure"]],
                'OEncoded':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevEncoded,"DevEncoded structure to test polling/history"]],
                'SetGetAlarms':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarStringArray,"DevVarStringArray containing alarm values"]],
                'SetGetRanges':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarStringArray,"DevVarStringArray containing ranges values"]],
                'IOPollingInDevice':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVarStringArray,"Polling in device test outputs"]]
               }

#                'IOChangeStatus':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevVoid,"void"]]
#                'String_attr':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ],
#                                 {'Polling period':250}],
#                'IOStartPoll':[[PyTango.ArgType.DevVoid,"void"],[PyTango.ArgType.DevLong,"Constant number set to 11"],
#                               {'Polling period':400}],

    attr_list = {'Short_attr':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'Long_attr':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ],
                              {'min alarm':1000,'max alarm':1500}],
                 'Double_attr':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'String_attr':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'Short_spec_attr':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,4]],
                 'Long_spec_attr':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,10]],
                 'Double_spec_attr':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,3]],
                 'String_spec_attr':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,2]],
                 'Short_ima_attr':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ,4,4]],
                 'Long_ima_attr':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ,10,2]],
                 'Double_ima_attr':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ,3,3]],
                 'String_ima_attr':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ,1,2]],
                 'attr_dq_sh':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_dq_lo':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_dq_db':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_dq_str':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_no_data':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_wrong_type':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_wrong_size':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,10]],
                 'attr_no_alarm':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'Long_attr_w':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE],
                                {'label':"Test label",'description':"Test description",'unit':"Kilogramme",'standard unit':100,
                                 'display unit':"Et ta soeur",'format':"Tres long",'min value':0,'max value':100000,
                                 'min alarm':1,'max alarm':99999,'min warning':2,'max warning':99998,'delta val':10000,
                                 'delta t':1,'event abs change':30000,'event rel change':20000,'event period':2000,
                                 'archive event abs change':33333,'archive event rel change':22222,'archive event period':3000}],

                 'Float_attr_w':[[PyTango.ArgType.DevFloat,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'UShort_attr_w':[[PyTango.ArgType.DevUShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'UChar_attr_w':[[PyTango.ArgType.DevUChar,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'Long_attr_with_w':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'Long_attr_with_w2':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'Short_attr_rw':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'Short_attr_w2':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'Double_attr_w':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'String_attr_w2':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'Short_attr_w':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE],
                                 {'memorized':'true'}],
                 'String_attr_w':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE],
                                  {'memorized':'true'}],
                 'Boolean_attr_w':[[PyTango.ArgType.DevBoolean,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE],
                                  {'memorized':'True'}],
                 'Float_attr':[[PyTango.ArgType.DevFloat,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'Boolean_attr':[[PyTango.ArgType.DevBoolean,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'UShort_attr':[[PyTango.ArgType.DevUShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'UChar_attr':[[PyTango.ArgType.DevUChar,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'Float_spec_attr':[[PyTango.ArgType.DevFloat,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,3]],
                 'Boolean_spec_attr':[[PyTango.ArgType.DevBoolean,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,5]],
                 'UShort_spec_attr':[[PyTango.ArgType.DevUShort,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,2]],
                 'UChar_spec_attr':[[PyTango.ArgType.DevUChar,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,6]],
                 'PollLong_attr':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'PollString_spec_attr':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,2]],
                 'Short_spec_attr_w':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,4],
                                      {'max value':100}],
                 'Short_ima_attr_w':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.WRITE,4,4]],
                 'Long_spec_attr_w':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,2]],
                 'Double_spec_attr_w':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,2]],
                 'String_spec_attr_w':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,2]],
                 'Float_spec_attr_w':[[PyTango.ArgType.DevFloat,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,4]],
                 'Boolean_spec_attr_w':[[PyTango.ArgType.DevBoolean,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,4]],
                 'UShort_spec_attr_w':[[PyTango.ArgType.DevUShort,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,4]],
                 'UChar_spec_attr_w':[[PyTango.ArgType.DevUChar,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,4]],
                 'String_ima_attr_w':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.WRITE,2,2]],
                 'Float_ima_attr_w':[[PyTango.ArgType.DevFloat,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.WRITE,8,8]],
                 'UShort_ima_attr_w':[[PyTango.ArgType.DevUShort,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.WRITE,8,2]],
                 'Short_spec_attr_rw':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,4]],
                 'Long_spec_attr_rw':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,4]],
                 'String_spec_attr_rw':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,2]],
                 'Float_spec_attr_rw':[[PyTango.ArgType.DevFloat,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,4]],
                 'UChar_spec_attr_rw':[[PyTango.ArgType.DevUChar,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,4]],
                 'Short_ima_attr_rw':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ_WRITE,4,4]],
                 'String_ima_attr_rw':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ_WRITE,2,2]],
                 'Boolean_ima_attr_rw':[[PyTango.ArgType.DevBoolean,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ_WRITE,4,4]],
                 'UShort_ima_attr_rw':[[PyTango.ArgType.DevUShort,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ_WRITE,2,4]],
                 'Poll_buff':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ,2,2]],
                 'Poll_buffRW':[[PyTango.ArgType.DevString,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ_WRITE,2,2]],
                 'attr_asyn':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_asyn_to':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_asyn_except':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'attr_asyn_write':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'attr_asyn_write_to':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'attr_asyn_write_except':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'event_change_tst':[[PyTango.ArgType.DevLong,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,10]],
                 'event64_change_tst':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,6]],
                 'event_quality_tst':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,2]],
                 'slow_actuator':[[PyTango.ArgType.DevShort,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'fast_actuator':[[PyTango.ArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'Long64_attr':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'Long64_spec_attr':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ,10]],
                 'Long64_ima_attr':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ,10,2]],
                 'Long64_attr_w':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.WRITE]],
                 'Long64_spec_attr_w':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.WRITE,4]],
                 'Long64_ima_attr_w':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.WRITE,4,4]],
                 'Long64_attr_rw':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'Long64_spec_attr_rw':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,4]],
                 'Long64_ima_attr_rw':[[PyTango.ArgType.DevLong64,PyTango.AttrDataFormat.IMAGE,PyTango.AttrWriteType.READ_WRITE,4,4]],
                 'Encoded_attr':[[PyTango.CmdArgType.DevEncoded,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'ULong_attr':[[PyTango.CmdArgType.DevULong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'ULong64_attr':[[PyTango.CmdArgType.DevULong64,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'ULong_attr_rw':[[PyTango.CmdArgType.DevULong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'ULong64_attr_rw':[[PyTango.CmdArgType.DevULong64,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'State_attr_rw':[[PyTango.CmdArgType.DevState,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'ULong_spec_attr_rw':[[PyTango.CmdArgType.DevULong,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,3]],
                 'ULong64_spec_attr_rw':[[PyTango.CmdArgType.DevULong64,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,3]],
                 'State_spec_attr_rw':[[PyTango.CmdArgType.DevState,PyTango.AttrDataFormat.SPECTRUM,PyTango.AttrWriteType.READ_WRITE,3]],
                 'Sub_device_tst':[[PyTango.CmdArgType.DevBoolean,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'SlowAttr':[[PyTango.CmdArgType.DevDouble,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'Encoded_image':[[PyTango.CmdArgType.DevEncoded,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ]],
                 'DefAttr':[[PyTango.CmdArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'DefUserAttr':[[PyTango.CmdArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE],
                                {'description':"User desc",'min value':30,'delta val':77, 'delta t':88,'event rel change':55,'event period':1500}],
                 'DefClassAttr':[[PyTango.CmdArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE]],
                 'DefClassUserAttr':[[PyTango.CmdArgType.DevLong,PyTango.AttrDataFormat.SCALAR,PyTango.AttrWriteType.READ_WRITE],
                                {'description':"User desc",'min value':30,'delta val':77, 'delta t':88,'event rel change':55,'event period':1500}]
              }


if __name__ == '__main__':
    try:
        py = PyTango.Util(sys.argv)
        py.add_TgClass(DevTestClass,DevTest,'DevTest')

        U = PyTango.Util.instance()
        U.server_init()
        U.server_run()

    except PyTango.DevFailed,e:
        print '-------> Received a DevFailed exception:',e
    except Exception,e:
        print '-------> An unforeseen exception occured....',e
