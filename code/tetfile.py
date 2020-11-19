import numpy as np

class tfile:
    def __init__(self, fileID):
        self.fileID = fileID
        self.encode = {
            "none": 0,
            "rot_anticlock": 1,
            "rot_clock": 2,
            "move_left": 3,
            "move_right": 4    
        }
    
    def write(self, screenmat, instruction):                            #writes one labelled state
        f = open("trainingsession{0}.txt".format(self.fileID), "a")
        for i in range(4,24):
            for j in range(10):
                f.write(str(int(screenmat[i][j])))        
        f.write("\n{0}\n".format(self.encode[instruction]))
        f.close()
    
    def read(self):                                                     #reads ALL labelled states until BREAK
        labelled_states = []
        f = open("trainingsession{0}.txt".format(self.fileID), "r")
        datastrings = f.readlines()
        is_state = True
        for line in datastrings:
            if line == "BREAK\n":
                break
            state = np.zeros((25,10))
            instruction = 0
            if is_state:
                charcount = 0
                for i in range(4,24):
                    for j in range(10):
                        state[i][j] = int(line[charcount])
                        charcount += 1
                is_state = False
            else:
                instruction = int(line)
                is_state = True
            labelled_states.append((state,instruction))
        f.close()
        return labelled_states
    
    def getID(self):
        return self.fileID
    
    def cutoff(self):
        f = open("trainingsession{0}.txt".format(self.fileID), "a")
        f.write("BREAK\n")
        f.close()
