import numpy as np
import pygame as pg
import random as rng
import tetfile as tfile
WHITE = (255,255,255)
BLACK = (0,0,0)
GREY = (100,100,100)
GREEN = (0,255,0)
INITIAL_SCREENMAT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


decode = {
        0: "none",
        1: "move_left",
        2: "move_right"   
        }

class Tetromino:
    
    def __init__(self, name, x,y):
        self.name = name
        self.x = x
        self.y = y
        self.shape = np.array([[1,1],[1,1]])                                    ##
                                                                                ##

def freezetet(screenmat):                                                              #converts the tetromino squares into white remnant squares and destroys the active tet, ready to create a new one
    for i in range(4,24):
        for j in range(10):
            if screenmat[i,j] == 2:
                screenmat[i,j] = 1
    return screenmat

def drawscreen(screen, screenmat):                                                                   #visually renders the screenmat matrix
    screen.fill(GREY)                                                               #grey border
    pg.draw.rect(screen, BLACK, [50,0,200,400])                                     #black background
    for i in range(4,24):
        for j in range(10):
            if screenmat[i,j] == 1:
                pg.draw.rect(screen, WHITE, [(50+(20*j)),(20*(i-4)),19,19])         #tets that have already fallen are white
            elif screenmat[i,j] == 2:
                pg.draw.rect(screen, GREEN, [(50+(20*j)),(20*(i-4)),19,19])         #currently controlled active tet is green
    pg.display.flip()


def tetdrop(tet, screenmat):                                                               #inserts the active tetromino into the screen matrix
    tetsize = 0
    tet.y += 1                                                                                 #drop tetromino by one layer (only occurs when checking new position)
    tetsize = 2 
    for j in range(tet.x,(tet.x + tetsize)):
        for i in range(tet.y, (tet.y + tetsize)):
            if tet.shape[(i - tet.y),(j - tet.x)] == 1:
                if (not (0 <= i <= 23)) or (not (0 <= j <= 9) or (screenmat[i,j] == 1)):        
                    tet.y += (-1)
                    # print("fail")
                    failmat = np.full_like(screenmat, 2)
                    return failmat                                                                             
    
    for i in range(0,25):                                                           #If function is still running then the tetromino projected position is safe so screenmat can be updated with its new position
        for j in range(10):
            if screenmat[i,j] == 2:                                                 #clear old tetromino position
                screenmat[i,j] = 0
            # if i > 23:
            #     screenmat[i,j] = 0                                                  #ensure everything below the bottom of the screen is clear
    for j in range(tet.x,(tet.x + tetsize)):                                    
        for i in range(tet.y, (tet.y + tetsize)):               
            if tet.shape[(i - tet.y),(j - tet.x)] == 1:                             #add new position
                screenmat[i,j] = 2                                                
    return screenmat
                    
def scorecheck(screenmat):
    increment = 0
    for i in range(4,24):
        fullrow = True
        for j in range(10):
            if screenmat[i,j] == 0:
                fullrow = False                                                                     
        if fullrow == True:
            for l in range(10):                                                     #if no break the row must be complete
                k = 4                                                               #find the top of each column and remove, to get rid of completed row
                while screenmat[k,l] == 0:     
                    k += 1
                screenmat[k,l] = 0      
            increment += 1
    return increment, screenmat

                #####GRID ORIGIN IS IN THE TOP LEFT CORNER BECAUSE MATRIX INDICES COUNT FROM THERE#####

class TetrisGame:

    def __init__(self,sessionID,manualinput,demo):
        pg.init()
        self.screenmat = INITIAL_SCREENMAT.copy()       #a 10x24 pixel matrix. top 4 rows should not be rendered as they are used for spawning tetrominoes. 
                                                 #25th row is under the displayed area, and is there to avoid running off the end of the array during hitfloor check
        size = [300, 400]
        self.screen = pg.display.set_mode(size)
        pg.display.set_caption("tetrisgame")
        self.score = 0
        self.manualinput = manualinput                          #SWITCH BETWEEN HUMAN INPUT AND FILE INPUT
        self.demo = demo                                        #DO WE SHOW MACHINE PLAY?
        self.frameadv = True
        self.realtime = True
        self.sessionID = sessionID
        self.stepcount = 0
        # if self.manualinput == True:
            # self.sessionfile = tfile.tfile(sessionID)
        rng.seed()                                                                     
        self.instruction = "none"
        init_x = rng.randint(4,6)
        self.tet = Tetromino("Bx", init_x,0)                                                        #initialise                                                                    
        tetdrop(self.tet, self.screenmat)

    def game_reset(self):
        if self.demo == True:
            print("YOU LOST!")
        self.screenmat = INITIAL_SCREENMAT.copy()
        self.score = 0
        self.stepcount = 0
        self.tet = Tetromino("Bx", 4,0)
        tetdrop(self.tet, self.screenmat)

    def update(self):
        for event in pg.event.get():                                                    # read input
            if event.type == pg.QUIT:                                                   # if user clicked close
                # if self.manualinput == True:
                    # self.sessionfile.cutoff()                                           # close file if recording
                return "quit"                                                            # kill main loop
            if event.type == pg.KEYDOWN:    
                if self.manualinput == True:                                             # read other keys to find next instruction
                    if event.key == pg.K_a:
                        self.instruction = "move_left"
                    if event.key == pg.K_d:
                        self.instruction = "move_right"
                    if event.key == pg.K_v:                                                     # manual frame advance
                        if self.realtime == False:
                            self.frameadv = True     
                    if event.key == pg.K_p:                                                     # pause/unpause
                        self.realtime = not self.realtime
                        self.frameadv = self.realtime
                    if event.key == pg.K_x:                                                     # print screenmat for debug
                        if self.frameadv == False:
                            print(self.screenmat)                                                   
                else:
                    if event.key == pg.K_s:
                        self.demo = not self.demo
        status = "wait"
        if self.frameadv == True:                                                #stuff to do when frame advances
            if self.realtime == False:
                self.frameadv = False
            # if self.manualinput == True:
                # self.sessionfile.write(self.screenmat, self.instruction)                                             #begin by writing current state
            if self.instruction == "move_left":
                self.tet.x += (-1)
            elif self.instruction == "move_right":
                self.tet.x += 1

            failmat = np.full_like(self.screenmat, 2)    
            check = tetdrop(self.tet, self.screenmat)
            if np.array_equal(check, failmat):                                            #instructions that move the tetromino offscreen are invalid       
                if self.instruction == "move_left":                                     #find out which instruction was given and undo
                    self.tet.x += 1
                elif self.instruction == "move_right":
                    self.tet.x += (-1)
                # print("undone")
                check = tetdrop(self.tet, self.screenmat)
            if np.array_equal(check, failmat):                                               # if moving the tetromino straight down without carrying out the instruction
                self.screenmat = freezetet(self.screenmat)                    # still fails, then it must have landed on the floor or the stack
                scoreret = scorecheck(self.screenmat)
                inc = scoreret[0]
                if inc != 0:
                    self.score += inc
                    self.screenmat = scoreret[1]
                    if self.demo == True:
                        print("POINT(S) SCORED! Current score is {0}".format(self.score))
                init_x = rng.randint(4,6)
                self.tet = Tetromino("Bx", init_x,0)
                tetdrop(self.tet, self.screenmat)
            else:
                self.screenmat = check                                                               
            
            self.stepcount += 1
            self.instruction = "none"                                            #instruction handled, can now reset
            status = "advance"
            for i in range(10):                                             #has the stack reached the top of the screen?
                if self.screenmat[4][i] == 1:
                    status = "end_episode"
                    self.game_reset()
            
        if (self.manualinput == True) or (self.demo == True):
            drawscreen(self.screen, self.screenmat)
        return status

    def dqn_quitcheck(self):
        for event in pg.event.get():                                                    #read input
            if event.type == pg.QUIT:                                                   #If user clicked close
                return True                                                            #kill main loop
    
    def dqn_update(self):
        clock = pg.time.Clock()
        if self.instruction == "move_left":
            self.tet.x += (-1)
        elif self.instruction == "move_right":
            self.tet.x += 1

        failmat = np.full_like(self.screenmat, 2)    
        check = tetdrop(self.tet, self.screenmat)
        if np.array_equal(check, failmat):                                            #instructions that move the tetromino offscreen are invalid       
            if self.instruction == "move_left":                                     #find out which instruction was given and undo
                self.tet.x += 1
            elif self.instruction == "move_right":
                self.tet.x += (-1)
            # print("undone")
            check = tetdrop(self.tet, self.screenmat)
        if np.array_equal(check, failmat):                                               # if moving the tetromino straight down without carrying out the instruction
            self.screenmat = freezetet(self.screenmat)                    # still fails, then it must have landed on the floor or the stack
            scoreret = scorecheck(self.screenmat)
            inc = scoreret[0]
            if inc != 0:
                self.score += inc
                self.screenmat = scoreret[1]
            init_x = rng.randint(4,6)
            self.tet = Tetromino("Bx", init_x,0)
            tetdrop(self.tet, self.screenmat)
        else:
            self.screenmat = check                                                               
        
        self.stepcount += 1
        self.instruction = "none"                                            #instruction handled, can now reset
        
        status = "advance"
        for i in range(10):                                             #has the stack reached the top of the screen?
            if self.screenmat[4][i] == 1:
                status = "end_episode"
        
        if self.demo == True:
            drawscreen(self.screen, self.screenmat)
            clock.tick(5)
        return status

    def get_max_height(self):
        height = 0
        for i in range(25):
            for j in range(10):
                if self.screenmat[i][j] == 1:
                    height = 24-i
                    return height
        return height
    
    def flush_metric(self):                         # part of reward: test how close shape fits with space below
        metric = 0
        width = []
        distcounts = []
        for j in range(10):                             # check distances from shape to stack: sum differences of distances w.r.t. minimum: this gives number of holes below shape
            distcount = 0
            countcolumn = False
            for i in range(25):                
                if self.screenmat[i][j] == 2:
                    distcount = 0
                    countcolumn = True
                elif self.screenmat[i][j] == 1:
                    break
                else:
                    if countcolumn == True:
                        distcount += 1
            if countcolumn == True:
                distcounts.append(distcount)
                width.append(j)
        if width == []:
            return 0                                        # 1 frame where tetromino isn't present during reset step: return zero to avoid crash if tet not found (i.e. screenmat[i][j] != 2 for all i,j)
        for d in distcounts:
            # print("min: {0} d: {1}".format(min(distcounts), d))
            metric += (min(distcounts) - d)        
        if metric == 0:                                             # don't bother to check edges if gaps are left below
            if min(width) == 0:                             # check sides: first minimum (left) then maximum (right), 1 point per height of side (1 points if edge of screen)
                metric += 1
            else: 
                j1 = min(width) - 1
                j2 = min(width)
                h1 = 0
                h2 = 0
                for i in range(25):                        # horribly inefficient implementation, need to think of a better way to do this
                    if self.screenmat[i][j1] == 1:
                        h1 = i
                        break
                for i in range(25):
                    if self.screenmat[i][j2] == 1:
                        h2 = i
                        break
                metric += (h2 - h1)
            if max(width) == 9:                             
                metric += 1
            else: 
                j1 = max(width) + 1
                j2 = min(width)
                h1 = 0
                h2 = 0
                for i in range(25):                        
                    if self.screenmat[i][j1] == 1:
                        h1 = i
                        break
                for i in range(25):
                    if self.screenmat[i][j2] == 1:
                        h2 = i
                        break
                metric += (h2 - h1)
        metric += 10
        if metric <= 0:
            metric = 0
        return metric