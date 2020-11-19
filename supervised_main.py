import sys
sys.path.append("code")
import gamelib as gl
import pygame as pg
import supervised_model as ml
import numpy as np

manualinput = True                           #SWITCH BETWEEN HUMAN INPUT AND FILE INPUT
sessionID = 6
model = 0                                      
demo = True                                     #DO WE SHOW MACHINE PLAY?

if manualinput == False:
    model = ml.BuildAndTrain(sessionID)

clock = pg.time.Clock()
done = False
game = gl.TetrisGame(sessionID,manualinput,demo)
while not done:
    status = game.update()
    if status == "quit":
        done = True
    if (manualinput == True) or (demo == True):
        clock.tick(5)
        height = game.get_max_height()
        print(str(height))
    if manualinput == False:
        screen = game.screenmat.reshape(1,250).astype("float32")
        predictions = model.predict(screen)
        predictions[0,0] = predictions[0,0]**3        #root the odds of doing nothing, hacky fix but might show improved behaviour
        game.instruction = gl.decode[np.argmax(predictions)]
        if demo == True:
            print("instruction: {0} confidence: {1}".format(gl.decode[np.argmax(predictions)], np.amax(predictions)))
pg.quit()

# TO DO: model actually looks good for first couple of moves: try alternating between automatic and manual control every 50 moves or so? remember to record manual moves and add to file, should lead to big
# file that can be used to train improved model.

# research suggests that supervised tetris is basically impossible because the state space is too big to explore in a reasonable time