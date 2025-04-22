# train_dqn.py
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from gameplay import update_board, find_legal, check_for_win, choose_action

class ReplayBuffer:
    def __init__(self, cap=10000): self.cap=cap; self.buf=[]
    def add(self, t):
        if len(self.buf)>=self.cap: self.buf.pop(0)
        self.buf.append(t)
    def sample(self, bs): return random.sample(self.buf, min(len(self.buf),bs))

def build_dqn_model(input_shape=(6,7,2), output_size=7):
    m = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32,3,activation='relu',padding='same'),
        Conv2D(64,3,activation='relu',padding='same'),
        Flatten(), Dense(128,activation='relu'),
        Dense(output_size,activation='linear')
    ])
    m.compile(Adam(1e-3),'mse')
    return m

def play_game_dqn(model, opp, epsilon=0.1, gamma=0.99):
    board=np.zeros((6,7,2),int); trans=[]; cur="plus"
    while True:
        legal=find_legal(board)
        if not legal:
            trans.append((board.copy(),None,0,board.copy(),True)); break
        if cur=="plus":
            if random.random()<epsilon: a=random.choice(legal)
            else:
                q=model.predict(np.expand_dims(board,0),verbose=0)[0]
                mask=np.full_like(q,-np.inf); mask[legal]=q[legal]; a=int(np.argmax(mask))
            prev=board.copy(); board=update_board(board,"plus",a)
            res=check_for_win(board,a)
            rew=1 if res=="plus" else -1 if res=="minus" else 0
            done = res!="nobody"
            trans.append((prev,a,rew,board.copy(),done))
            if done: break
        else:
            a=choose_action(opp,board)
            board=update_board(board,"minus",a)
            res=check_for_win(board,a)
            if res!="nobody":
                prev,ac,r,ns,d=trans[-1]; trans[-1]=(prev,ac,-1,ns,True)
                return trans
        cur="minus" if cur=="plus" else "plus"
    return trans

def train_dqn(model, opp, episodes=500, bs=32, gamma=0.99,
              eps_start=1.0, eps_min=0.1, eps_decay=0.995):
    buf=ReplayBuffer(); opt=Adam(1e-3); losses=[]; eps=eps_start
    for ep in range(episodes):
        game=play_game_dqn(model, opp, eps, gamma)
        for t in game: buf.add(t)
        batch=buf.sample(bs)
        if not batch: continue
        S=np.array([s for s,a,r,s2,d in batch]);
        A=np.array([a or 0 for s,a,r,s2,d in batch]);
        R=np.array([r for s,a,r,s2,d in batch]);
        S2=np.array([s2 for s,a,r,s2,d in batch]);
        D=np.array([d for s,a,r,s2,d in batch])
        target_q=model.predict(S2,verbose=0)
        M=np.max(target_q,axis=1)
        T=R + gamma*M*(1-D.astype(int))
        with tf.GradientTape() as tape:
            qv=model(S,training=True)
            idx=tf.stack([tf.range(len(A)),A],axis=1)
            preds=tf.gather_nd(qv,idx)
            loss=tf.reduce_mean((T-preds)**2)
        grads=tape.gradient(loss,model.trainable_variables)
        opt.apply_gradients(zip(grads,model.trainable_variables))
        losses.append(loss.numpy())
        eps=max(eps_min,eps*eps_decay)
        if (ep+1)%100==0: print(f"DQN Ep {ep+1}/{episodes} Loss {loss.numpy():.4f} Eps {eps:.3f}")
    return losses

if __name__ == "__main__":
    m2=load_model("M2.h5")
    dqn=build_dqn_model()
    dqn_losses=train_dqn(dqn,m2,episodes=500)
    dqn.save("DQN_trained.h5")
    np.save("dqn_losses.npy",dqn_losses)
    print("Saved DQN_trained.h5 and dqn_losses.npy")
