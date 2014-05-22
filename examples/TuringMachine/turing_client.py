from turing_machine import TuringMachine

transition_function = {("init","0"):("init", "1", "R"),
                       ("init","1"):("init", "0", "R"),
                       ("init"," "):("final"," ", "N"),
                       }

t = TuringMachine("010011 ", final_states=["final"],
                  transition_function=transition_function)

print("Input on Tape:")
print(t.get_tape_str())

while not t.final():
    t.step()

print("Result of the Turing machine calculation:")    
print(t.get_tape_str())
