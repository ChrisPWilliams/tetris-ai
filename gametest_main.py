import sys
sys.path.append("code")
import gamelib as gl
import pygame as pg
import numpy as np

sessionID = 6

clock = pg.time.Clock()
done = False
game = gl.TetrisGame(sessionID,True,True)
game.realtime = False
game.frameadv = False
while not done:
    status = game.update()
    if status == "quit":
        done = True
    if status == "advance":
        # height = game.get_max_height()        
        # print("height is {0}".format(height))
        flush = game.flush_metric()
        print(str(flush))
    clock.tick(5)
pg.quit()