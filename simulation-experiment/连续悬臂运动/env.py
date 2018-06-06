# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/5/9 14:45
'''

import pyglet

class ArmEnv(object):
    viewer=None
    def __init__(self):
        pass
    def step(self,action):
        pass
    def reset(self):
        pass
    def render(self):
        if self.viewer is None:
            self.viewer=Viewer()
        self.viewer.render()

class Viewer(pyglet.window.Window):
    bar_thc=5
    def __init__(self):
        super(Viewer,self).__init__(width=1000,height=400,resizable=False,caption='Arm',vsync=False)
        pyglet.gl.glClearColor(1,1,1,1)
        self.batch=pyglet.graphics.Batch()

        self.point=self.batch.add(
            4,pyglet.gl.GL_QUADS,None,
            ('v2f',[50,50,
                    50,100,
                    100,100,
                    100,50]),
            ('c3B',(86,109,249)*4))

        self.arm1=self.batch.add(
            4,pyglet.gl.GL_QUADS,None,
            ('v2f',[250,250,250,300,260,300,260,250]),
            ('c3B',(249,86,86)*4))

        self.arm2=self.batch.add(
            4,pyglet.gl.GL_QUADS,None,
            ('v2f', [200, 200, 300, 200, 300, 210, 200, 210]),
            ('c3B', (249, 86, 86) * 4))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')

    def on_draw(self):
        self.clear()
        self.batch.draw()
    def _update_arm(self):
        pass

if __name__=='__main__':
    env=ArmEnv()
    while True:
        env.render()