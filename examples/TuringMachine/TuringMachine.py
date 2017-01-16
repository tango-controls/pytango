import json
from PyTango import DevState
from PyTango.server import Device
from PyTango.server import attribute, command, device_property


class TuringMachine(Device):

    blank_symbol = device_property(dtype=str, default_value=" ")
    initial_state = device_property(dtype=str, default_value="init")

    def init_device(self):
        Device.init_device(self)
        self.__tape = {}
        self.__head = 0
        self.__state = self.initial_state
        self.__final_states = []
        self.__transition_function = None
        self.set_state(DevState.RUNNING)

    @attribute(dtype=(str,))
    def final_states(self):
        return self.__final_states

    @final_states.write
    def final_states(self, final_states):
        self.__final_states = final_states

    @attribute(dtype=str)
    def transition_function(self):
        return self.__transition_function

    @transition_function.write
    def transition_function(self, func_str):
        self.__transition_function = tf = {}
        for k, v in json.loads(func_str).items():
            tf[tuple(str(k).split(","))] = map(str, v)
        print(tf)

    @attribute(dtype=str)
    def tape(self):
        s, keys = "", self.__tape.keys()
        min_used, max_used = min(keys), max(keys)
        for i in range(min_used, max_used):
            s += self.__tape.get(i, self.__blank_symbol)
        return s

    @command
    def step(self):
        char_under_head = self.__tape.get(self.__head, self.blank_symbol)
        x = self.__state, char_under_head
        if x in self.__transition_function:
            y = self.__transition_function[x]
            self.__tape[self.__head] = y[1]
            if y[2] == "R":
                self.__head += 1
            elif y[2] == "L":
                self.__head -= 1
            self.__state = y[0]
        print(self.__state)

    def dev_state(self):
        if self.__state in self.__final_states:
            return DevState.ON
        else:
            return DevState.RUNNING


if __name__ == "__main__":
    TuringMachine.run_server()
