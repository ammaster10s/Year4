# coding:utf-8
"""
迷路プログラム（実験Ⅱ第2週）
"""
import numpy as np
import random
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from colour import Color
import csv
import matplotlib.animation as animation
import os

MAIN_LOOP_MAX=10000 # 最大エピソード数
MAX_OF_TRIAL=512 # 最大ステップ数
HEIGHT, WIDTH = 8,8 # 迷路の大きさ, 高さ：HEIGHT, 幅：WIDTH
NUM_OF_STATES=HEIGHT*WIDTH
NUM_OF_ACTION=4
ALPHA=0.2
GAMMA=0.9
EPSILON=0.3
REWARD=10
#ε-decayを入れる場合（例）の定数
# EPSILON=1.0 #εの初期値
# EPSILON_DECAY = 0.9995 #εの減衰率
# EPSILON_MIN = 0.0 #εの最小値

#迷路定義
initial_node = 9
terminal_node = 54
#initial_node = 21
#terminal_node = 378
next_s_value=np.array([-WIDTH,WIDTH,-1,1])
UP,DOWN,LEFT,RIGHT= 0,1,2,3
cntn = False # 前回のQ_tableを継続して使用する場合：True
#狼セル（禁止領域・壁）
# wallcells = [34,35,36,37]
wallcells = []

# environment class
class Env:
    #def __init__(self):        
    def step(self, current_State, action): 
        """
        1step actionに従って状態を進める
        """
        next_state = current_State + next_s_value[action]
        return next_state
    
    def admissible_action_check(self, state, action):
        """
        与えらえたstateとactionに対して，移動可能か否かを判定する
        """
        if (state < WIDTH  and action == UP) or \
            (state >= WIDTH*(HEIGHT-1) and action == DOWN) or \
                (state%WIDTH == 0 and action == LEFT) or\
                    (state%WIDTH == WIDTH-1 and action == RIGHT): 
            return False
        else:
            return True

    def get_reward(self, next_state):
        done=False
        reward=0

        if next_state in wallcells:
            reward=-10
            done=True
        elif next_state == terminal_node:
            reward=100
            done=True
        else:
            done=False  
        return reward, done 

class Agent:
    def __init__(self):
        self.env = Env()  

    def select_action(self, state, qvalue, epsilon): # 行動選択
        """
        ε-greedy法による行動選択
        """
        if random.random() < epsilon:
            while True:
                action = random.randrange(NUM_OF_ACTION)
                if self.env.admissible_action_check(state,action):
                    break
            return action
        else:
            action_sorted = np.argsort(-qvalue)
            for action  in action_sorted[state]:
                if self.env.admissible_action_check(state,action):
                    break        
            return action

    def Q_value_Update(self, s, snext, a, qvalue,done,reward): 
        """
        Q値の更新
        """
        if done == True:# 報酬が付与される場合
            qvalue[s][a] = qvalue[s][a] + ALPHA * (reward-qvalue[s][a])       
        else: # {  /*報酬なし*/
            qvalue[s][a] = qvalue[s][a] + ALPHA * (GAMMA*qvalue[snext].max() - qvalue[s][a])
        return qvalue  

def main():    
    env=Env()
    agent = Agent()    
    # Q_tableの初期化
    q_value = np.random.rand(NUM_OF_STATES,NUM_OF_ACTION)
    #Q_tableをファイルから入力
    if os.path.isfile('q_table.dat') and cntn == True:
        q_value = np.loadtxt('q_table.dat', np.float32)
    epsilon = EPSILON  
    #描画セットアップ
    fig = plt.figure(figsize=(10,15))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(223) 
    ax3 = fig.add_subplot(224)    
    images_hm = []
    images_r = []
    route = []    
    data = []    
    # ルート探索
    for i in range(MAIN_LOOP_MAX):# 最大エピソード数まで繰り返す
        current_state = initial_node #　スタートノード
        step = 0
        # epsilon = max(epsilon*EPSILON_DECAY,EPSILON_MIN) 

        while step <= MAX_OF_TRIAL: # 最大ステップ数まで繰り返す
            action = agent.select_action(current_state, q_value, epsilon)
            next_state = env.step(current_state, action)
            reward, done = env.get_reward(next_state)
            q_value  = agent.Q_value_Update(current_state,next_state,action,q_value,done,reward)
            current_state = next_state
            route.append(next_state)
            if done == True:
                break
            step += 1

        #データの保存
        if i%100 == 0:
            im_hm = heet_map(q_value,i,epsilon)
            images_hm.append(im_hm)
        if i%(MAIN_LOOP_MAX/5) == 0:
            for j in range(len(route)):
                im_r = route_map(route[j],i,epsilon)            
                images_r.append(im_r)
        route.clear()
        if True: #if i % 10 == 0:   
            data.append([i, step])

    #結果をファイルに保存    
    data=np.array(data).reshape(len(data),-1)
    with open('result.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    #ヒートマップをファイルに保存
    images_hm[0].save('heet_map.gif',save_all=True, append_images=images_hm[1:], optimize=False, duration=40)
    images_r[0].save('route.gif',save_all=True, append_images=images_r[1:], optimize=False, duration=200)

    #結果を画面に描画  
    ax1.plot(data[:,0][::10],data[:,1][::10],"-",linewidth=1)
    ax1.set_title('Fig.1: step number')
    ani1 = animation.FuncAnimation(fig, animation_hm_update, len(images_hm), fargs=(images_hm, ax2), interval=40, repeat=False)
    ani2 = animation.FuncAnimation(fig, animation_r_update, len(images_r), fargs=(images_r, ax3), interval=1, repeat=False)
    plt.show()

    #Q_tableをファイルに保存    
    np.savetxt('q_table.dat',  q_value)

    # end of main

def heet_map(q_value, epispde, epsilon): 
    """
    q_table のヒートマップの作成，青→赤で値が大きくなる
    """
    env=Env()
    red = Color("red")
    blue = Color('blue') 
    im = Image.new('RGB', (20*WIDTH, 20*HEIGHT), (128, 128, 128))
    draw = ImageDraw.Draw(im) 
    color_step = 100
    blue_red = list(blue.range_to(red, color_step))
    max_q_value = q_value.max(axis=1)/q_value.max()
    color_level = np.round(max_q_value*(color_step-1)).astype(int)
    for num in range(len(q_value)):
        i, j = divmod(num, WIDTH)
        rgb_view = np.round(255*np.array(blue_red[color_level[num]].get_rgb())).astype(int)
        draw.rectangle((20*j, 20*i, 20*(j+1) , 20*(i+1)), fill=(rgb_view[0],rgb_view[1],rgb_view[2]))
    for num in wallcells:
        i, j = divmod(num, WIDTH)
        draw.rectangle((20*j, 20*i, 20*(j+1) , 20*(i+1)), fill=(255,255,255))  
    draw.text((5, 5), f'epispde = {epispde}, epsilon= {epsilon:.2f}', (255,255,2555))
    return im

def route_map(route, epispde, epsilon):
    """
    生成されたルートの表示
    """
    env=Env()
    im = Image.new('RGB', (20*WIDTH, 20*HEIGHT), (50, 50, 50))
    draw = ImageDraw.Draw(im) 
    i, j = divmod(route, WIDTH)
    draw.ellipse((20*j+5, 20*i+5, 20*j+15 , 20*i+15), fill=(255,0,0))    
    i, j = divmod(initial_node, WIDTH)   
    draw.ellipse((20*j+5, 20*i+5, 20*j+15 , 20*i+15), fill=(0,255,255))
    i, j = divmod(terminal_node, WIDTH)
    draw.ellipse((20*j+5, 20*i+5, 20*j+15 , 20*i+15), fill=(255,255,0))
    for num in wallcells:
        i, j = divmod(num, WIDTH)
        draw.rectangle((20*j, 20*i, 20*(j+1) , 20*(i+1)), fill=(255,255,255))  
    if route  == terminal_node:
        i, j = divmod(route, WIDTH)
        draw.ellipse((20*j+5, 20*i+5, 20*j+15 , 20*i+15), fill=(255,255,255))   
    draw.text((5, 5), f'epispde = {epispde}, epsilon = {epsilon:.2f}', (255,255,255))
    return im

def animation_hm_update(i, im_hms, ax2):
    """
    ヒートマップのアニメーション
    """
    #plt.cla()
    anim_im = im_hms[i]
    ax2.imshow(anim_im)
    ax2.set_title('Fig.2: Heatmap of q_value')

def animation_r_update(i, images_r, ax3):
    """
    生成されたルートのアニメーションの表示
    """
    plt.cla()
    anim_im = images_r[i]
    ax3.imshow(anim_im)
    ax3.set_title('Fig.3: route map')
    
if __name__ == '__main__':
    main()