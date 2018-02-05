import pygame
import tkinter as tk
import threading
import time

from ddpg import *
from sensor import *
from plotting import *
from save_data import *
from joystick import *

flag=0

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global flag
        sen=CONTROL()
        agent = DDPG()
        sen.start()
        save_data = SAVE_DATA()
        hand = JOYSTICK()

        for step in range(2200):
            if step==799:
                start_time = time.clock()
                print("开始控制")
            # 系统状态采集
            angle1, angle_velocity1 = sen.read_ahrs()
            angle2, angle_velocity2, toqure = sen.read_daq()
            #对状态进行预处理和整合
            state = sen.state_pre([angle1, angle2, angle_velocity1, angle_velocity2])
            value = 10/2.73*agent.action(state)[0]
            value = hand.control
            if step>=799:
                if flag==0:
                    value=0
            else:
                value=0
            sen.write_daq(value)
            # 保存数据
            if step>=799:
                data = [angle1, angle2, angle_velocity1, angle_velocity2, value, toqure]
                save_data.recorde(data)
        print('消耗时间：%s s'%(time.clock() - start_time))
        sen.stop()
        save_data.save_to_file()

def start_me():
    global flag
    flag=1
    thread=MyThread()
    thread.start()

def stop_me():
    global flag
    flag=0

def plot_me():
    save_data = SAVE_DATA()
    plotting = PLOTTING()
    plotting.plot(save_data.load())

if __name__ == '__main__':
    joy=0
    if joy==1:
        pygame.init()
        j = pygame.joystick.Joystick(0)
        j.init()
        action = 0
        done = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 3: #Y键开始
                        start_me()
                        print('开始')
                    elif event.button == 2:#B键停止
                        stop_me()
                        print('停止')
                    elif event.button == 8:  # back按键是退出键
                        done = 1
            if done:
                break
            time.sleep(0.02)
        plot_me()
    else:
        window = tk.Tk()
        window.title('control')
        window.geometry('300x150')

        start = tk.Button(window,
                          text='开始',  # 显示在按钮上的文字
                          width=15, height=2,
                          command=start_me)  # 点击按钮式执行的命令
        start.pack()

        stop = tk.Button(window,
                         text='停止',  # 显示在按钮上的文字
                         width=15, height=2,
                         command=stop_me)  # 点击按钮式执行的命令
        stop.pack()

        start = tk.Button(window,
                          text='画图',  # 显示在按钮上的文字
                          width=15, height=2,
                          command=plot_me)  # 点击按钮式执行的命令
        start.pack()

        window.mainloop()
