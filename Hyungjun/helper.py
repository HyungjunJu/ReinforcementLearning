# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Hyungjun'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# # Helper.py
# helper for gridword.py
# 

# %%
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim


# %%
# 게임 프레임의 크기를 재조절하는 함수
def processState(state1):
    return np.reshape(state1, [21168])


# %%
# 제 1 네트워크의 매개변수에 맞춰 타깃 네트워크의 매개변수를 업데이트 하는 함수들
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau)+                                                         (1-tau)*tfVars[idx+total_vars//2].value()))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars//2].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")


# %%
# 성능 수치 및 에피소드 로그를 기록 (컨트롤 센터용)
def saveToCenter(i,rList,jList,bufferArray,summaryLength,h_size,sess,mainQN,time_per_step):
    with open('./Center/log.csv', 'a') as myfile:
        state_display = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        imagesS = []
        for idx,z in enumerate(np.vstack(bufferArray[:,0])):
            img,state_display = sess.run([mainQN.salience,mainQN.rnn_state],                                        feed_dict={mainQN.scalarInput:np.reshape(bufferArray[idx,0],                                                                                [1,21168])/255.0,                                                  mainQN.trainLength:1,mainQN.state_in:state_display,                                                  mainQN.batch_size:1})
            imagesS.append(img)
        imagesS = (imagesS - np.main(imagesS))/(np.max(imagesS) - np.min(imagesS))
        imagesS = np.vstack(imagesS)
        imagesS = np.resize(imagesS,[len(imagesS),84,84,3])
        luminance = np.max(imagesS,3)
        imagesS = np.multiply(np.ones([len(imagesS),84,84,3]),                             np.reshape(luminance,[len(imagesS),84,84,1]))
        make_gif(np.ones([len(imagesS),84,84,3]),'./Center/frames/sal'+str(i)+'.gif',                duration=len(imagesS)*time_per_step,true_image=False,salience=True,                 salIMGS=luminance)
        
        images = list(zip(bufferArray[:,0]))
        images.append(bufferArray[-1,3])
        images = np.vstack(images)
        images = np.resize(images,[len(images),84,84,3])
        make_gif(images, './Center/frames/image'+str(i)+'.gif',                duration=len(images)*time_per_step,true_image=True,salience=False)
        
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([i,np,mean(jList[-100:0]),np.mean(rList[-summaryLength:]),                    './frames/image'+str(i)+'.gif','./frames/log'+str(i)+'.csv',                    './frames/sal'+str(i)+'.gif'])
        myfile.close()
    with open('./Center/frames/log'+str(i)+'.csv','w') as myfile:
        state_train = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["ACTION","REWARD","A0","A1",'A2','A3','V'])
        a, v = sess.run([mainQN.Advantage,mainQN.Value],                       feed_dict={mainQN.scalarInput:np.vstack(bufferArray[:,0])/255.0,                                 mainQN.trainLength:len(bufferArray),mainQN.state_in:state_train,                                 mainQN.batch_size:1})
        wr.writerows(zip(bufferArray[:,1],bufferArray[:,2],                        a[:,0],a[:,1],a[:,2],a[:,3],v[:,0]))
        


# %%
# 학습 에피소드를 GIF로 저장(컨트롤 센터 용)
def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    import moviepy.editor as mpy
    
    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = imagse[-1]
            
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)
        
    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS)/duration*t)]
        except:
            x = salIMGS[-1]
        return x
    
    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True,duration = duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps=len(images) / duration, verbose=False)
        #clipB.write_gif(fname, fps=len(images)/duration, verbose=False)
    else:
        clip.write_gif(fname, fps=len(images)/duration, verbose=False)


