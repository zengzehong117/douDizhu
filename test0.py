# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from operator import itemgetter

class model(nn.Module):
    # initializers
    def __init__(self, in_dim = 48,embed_in = 15,embed_out = 128,num_act = [6,14,14]):
        super(model, self).__init__()
        self.embed = nn.Embedding(embed_in,embed_out)
        self.conv1 = nn.Conv1d(embed_out,256,kernel_size=3,stride = 2,padding = 1)
        self.conv2 = nn.Conv1d(256,256,kernel_size=3,stride = 1,padding = 1)
        self.conv3 = nn.Conv1d(256,512,kernel_size=3,stride = 2,padding = 1)
        self.conv4 = nn.Conv1d(512,512,kernel_size=3,stride = 1,padding = 1)
        self.conv5 = nn.Conv1d(512,1024,kernel_size=3,stride = 2,padding = 1)
        self.conv6 = nn.Conv1d(1024,1024,kernel_size=3,stride = 2,padding = 1)
#        self.conv5 = nn.Conv1d(256,256,kernel_size=3,stride = 1,padding = 1)
#        self.conv6 = nn.Conv2d(128, 128, kernel_size=3,stride=2)
        
        self.fc1 = nn.Linear(1024*3, num_act[0])
        self.fc2 = nn.Linear(1024*3, num_act[1])
        self.fc3 = nn.Linear(1024*3, num_act[2])

    # weight_init
    # forward method
    def forward(self, input):
        x    = self.embed(input).permute(0,2,1)
        x    = torch.relu(self.conv1(x))
        x    = torch.relu(self.conv2(x))
        x    = torch.relu(self.conv3(x))
        x    = torch.relu(self.conv4(x))
        x    = torch.relu(self.conv5(x))
        x    = torch.relu(self.conv6(x))
        x    = x.view(-1,1024*3)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)


        return out1,out2,out3 #F.softmax(out1),out2,out3
Q_net        = torch.load(f'model/Qnet_model{20000}.model') #model()
Q_net_oppo   = model()
optimizer    =  torch.optim.SGD(Q_net.parameters(), lr=0.01, momentum=0.9) #torch.optim.Adam(Q_net.parameters(), lr=0.001) 
target_Q_net = model()
test_player  = 0
print_mode   = 1
class Trainer():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 7
        self.board_height = 6
        self.n_in_row = 4

        self.game = Game(15)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 100  # mini-batch size for training
#        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        
#        if init_model: self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,model_file=init_model)
#        else: self.policy_value_net = PolicyValueNet(self.board_width,self.board_height)
#        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,c_puct=self.c_puct, n_playout=self.n_playout,is_selfplay=1)



    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play()
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
#            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)



    def run(self):
        """run the training pipeline"""
        

        self.collect_selfplay_data(self.play_batch_size)






class Game(object):
    """game server"""

    def __init__(self, num):
#        self.State_A = np.zeros((4,13))
#        self.State_B = np.zeros((4,13))
#        self.State_discarded = np.zeros((4,13))
#        self.State_hold = np.zeros((4,13))
        self.card_num = num
        self.state_curent = []
    def graphic(self, board, player1, player2):
        print(1)
    def get_onehot(self):
        print(1)
    def init_state(self):
        self.discard_cards = np.random.choice(np.arange(52),size=self.card_num*2,replace=False,p=None)
        card_a =  np.sort(self.discard_cards[:self.card_num])
        card_b =  np.sort(self.discard_cards[self.card_num:])
        card_a_onehot = np.eye(52,52)[card_a]
        card_b_onehot = np.eye(52,52)[card_b]
        state_a_413 = card_a_onehot.sum(0).reshape(4,13)
        state_b_413 = card_b_onehot.sum(0).reshape(4,13)
#        print('init cards 0 and 1:',state_a_413.sum(0),state_b_413.sum(0))
        state_a_413.sort(0)
        state_b_413.sort(0)
        self.state_curent = [state_a_413,state_b_413]
        if print_mode:
            state_a_for_print = np.argwhere(state_a_413==1)[:,1]
            state_b_for_print = np.argwhere(state_b_413==1)[:,1]
            state_a_for_print.sort()
            state_b_for_print.sort()
            self.cards_to_print = {0:'3',1:'4',2:'5',3:'6',4:'7',5:'8',6:'9',7:'10',8:'J',9:'Q',10:'K',11:'A',12:'2'}
            print('init player 0:',itemgetter(*state_a_for_print)(self.cards_to_print))
            print('init player 1:',itemgetter(*state_b_for_print)(self.cards_to_print))
        self.State_input = np.zeros((16,16))
        self.last_episode_end = True
        self.player_current   = 0
        self.action_space     = [6,14,14]

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        print(3)
    def put_cards(self, cards,state_playerself,putcards_type):
        """start a game between two players"""
        game_end = False
        episode_end = False
        cards_for_print = np.argwhere(cards==1)[:,1]
        cards_for_print.sort()
        if print_mode:
            if cards.sum() == 0.:
                print(f'player {self.player_current}  过.',)
            else:
                print(f'player {self.player_current}  put card:',itemgetter(*cards_for_print)(self.cards_to_print))
#        print('not put:',self.state_curent)
        if putcards_type ==7:
            self.last_episode_end = True
        else:
            self.last_episode_end = False
        state_playerself_puted = state_playerself - cards
        
        if (state_playerself_puted<0.).sum():
#            print(state_playerself,cards)
            print('error:  cards cannt be puted!')
        state_playerself_puted.sort(0)
        self.state_curent[self.player_current] = state_playerself_puted
        self.last_puted_cards = cards
        self.last_putcard_type= putcards_type
#        print('player %d put card:'%self.player_current,cards.sum(0))
        if state_playerself_puted.sum() <1.:
            game_end = True
            episode_end = True
            print(f'Winer is player {self.player_current}.',)
            
        self.player_current = 1 - self.player_current
#        print('put after:',self.state_curent)
#        print(self.state_curent)
#        self.last_episode_end = False
        return episode_end,game_end
        
        
        print(3)
    def play_step(self,use_target_Q_net = False):
        
        state_playerself = self.state_curent[0] if self.player_current == 0 else self.state_curent[1]
        state_playeroppo = self.state_curent[1] if self.player_current == 0 else self.state_curent[0]
        cards_toput = []
        padded_num = 16
        yaobuqi    = False
        if self.last_episode_end:
            self.State_input[:4,1:14]   = state_playerself
            self.State_input[4:8,1:14]  = state_playeroppo
            self.State_input[8:12,1:14] = np.zeros((4,13))
            
            state_self = np.sort(np.argwhere(self.State_input[:4,:]==1)[:,1])
            state_opp  = np.sort(np.argwhere(self.State_input[4:8,:]==1)[:,1])
            state_puted_last = np.sort(np.argwhere(self.State_input[8:12,:]==1)[:,1])
            state_self_pad = np.pad(state_self,(0,padded_num-len(state_self)),'constant',constant_values = 14)
            state_opp_pad  = np.pad(state_opp,(0,padded_num-len(state_opp)),'constant',constant_values = 14)
            state_puted_last_pad = np.pad(state_puted_last,(0,padded_num-len(state_puted_last)),'constant',constant_values = 14)
            input      =Variable(torch.LongTensor(np.concatenate([state_self_pad,state_opp_pad ,state_puted_last_pad]))).unsqueeze(0)
            output1,output2,output3 = Q_net(input) if self.player_current == test_player else Q_net_oppo(input)
            legal_act1 = np.zeros(self.action_space[0])
            for act1_id in range(self.action_space[0]):
                if act1_id==0:  #one card
                    if state_playerself.sum()>0:
                        legal_act1[0] =1
                if act1_id==1:                  #double cards
                    if (state_playerself.sum(0)>1).sum()>0:
                        legal_act1[1] =1
                if act1_id==2:                  #three cards
                    if (state_playerself.sum(0)>2).sum()>0:
                        legal_act1[2] =1
                if act1_id==3:                  #4 cards or 4+1 cards
                    if (state_playerself.sum(0)>3).sum()>0:
                        legal_act1[3] =1
                if act1_id==4:                  #more than 4 cards
                    tmp = np.ones(5)
                    for tmp_id in range(13-5+1):
                        if ((state_playerself.sum(0)>0)[tmp_id:tmp_id+5] * tmp).sum() == 5:
                            legal_act1[4] =1
                            break
                legal_act1[5] = 0
            output1_log = F.softmax(output1)
            output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
            action1 = output1_log_mask.view(-1).max(0)[1].item()
            
            legal_act2 = np.zeros(self.action_space[1])
#            for act2_id in range(self.action_space[1]):
            if action1 == 0:
                legal_act2[:-1][state_playerself.sum(0)>0] =1.
                legal_act2[-1] = 0.
            if action1 == 1:
                legal_act2[:-1][state_playerself.sum(0)>1] =1.
                legal_act2[-1] = 0.
            if action1 == 2:
                legal_act2[:-1][state_playerself.sum(0)>2] =1.
                legal_act2[-1] = 0.
            if action1 == 3:
                legal_act2[:-1][state_playerself.sum(0)>3] =1.
                legal_act2[-1] = 0.
            if action1 == 4:            #0 is 5 
                legal_act2[0] =1.
                legal_act2[-1] = 0.
            output2_log = F.softmax(output2)
            output2_log_mask = output2_log*torch.FloatTensor(legal_act2)
            action2 = output2_log_mask.view(-1).max(0)[1].item()
            
            legal_act3 = np.zeros(self.action_space[2])
            if action1 <=1:
                legal_act3[-1] = 1.
            if action1 == 2:
                legal_act3[:-1][state_playerself.sum(0)>0] =1.
                legal_act3[action2] =0
                legal_act3[-1] = 1.
            if action1 ==3:
                legal_act3[:-1][state_playerself.sum(0)>0] =1.
                legal_act3[action2] =0.
                legal_act3[-1] = 1.
            if action1 ==4:
                conv_tmp =np.ones(5+action2)
                for tmp_id in range(9-action2):
                    if ((state_playerself.sum(0)>0)[tmp_id:tmp_id+5] * conv_tmp).sum() == 5:
                        legal_act3[tmp_id] = 1.
            output3_log = F.softmax(output3)
            output3_log_mask = output3_log*torch.FloatTensor(legal_act3)
            action3 = output3_log_mask.view(-1).max(0)[1].item()
#            if action1 ==5:
            
            
            #get putCards
            cards_toput = np.zeros((4,13))
            if action1 == 0:
                cards_toput[-1,action2] =1.
                putcards_type = 0
            if action1 == 1:
                cards_toput[-1,action2] =1.
                cards_toput[-2,action2] =1.
                putcards_type = 1
            if action1 == 2:
                cards_toput[-1,action2] =1.
                cards_toput[-2,action2] =1.
                cards_toput[-3,action2] =1.
                putcards_type = 3
                if action2==action3:
                    print('error: action2==action3')
                if action3!=13:
                    cards_toput[-1,action3] = 1.  #3+1
                    putcards_type = 2
            if action1 == 3:
                cards_toput[-1,action2] =1.
                cards_toput[-2,action2] =1.
                cards_toput[-3,action2] =1.
                cards_toput[-4,action2] =1.
                putcards_type = 5
                if action2==action3:
                    print('error: action2==action3')
                if action3!=13:
                    cards_toput[-1,action3] = 1.   #4+1
                    putcards_type = 4
            if action1 == 4:            #0 is 5 
                for toput_id in range(13):
                    if toput_id>=action3 and toput_id < action3 + action2 + 5:
                        cards_toput[-1,toput_id] = 1.
                putcards_type = 6
            episode_end,game_end = self.put_cards(cards_toput,state_playerself,putcards_type)
            
        else:
            self.State_input[:4,1:14] = state_playerself
            self.State_input[4:8,1:14] = state_playeroppo
            self.State_input[8:12,1:14] = self.last_puted_cards
            state_self = np.sort(np.argwhere(self.State_input[:4,:]==1)[:,1])
            state_opp  = np.sort(np.argwhere(self.State_input[4:8,:]==1)[:,1])
            state_puted_last = np.sort(np.argwhere(self.State_input[8:12,:]==1)[:,1])
            state_self_pad = np.pad(state_self,(0,padded_num-len(state_self)),'constant',constant_values = 14)
            state_opp_pad  = np.pad(state_opp,(0,padded_num-len(state_opp)),'constant',constant_values = 14)
            state_puted_last_pad = np.pad(state_puted_last,(0,padded_num-len(state_puted_last)),'constant',constant_values = 14)
            input      =Variable(torch.LongTensor(np.concatenate([state_self_pad,state_opp_pad ,state_puted_last_pad]))).unsqueeze(0)
            output1,output2,output3 = Q_net(input) if self.player_current == test_player else Q_net_oppo(input)
            def check_bigger_boom():
                if (state_playerself.sum(0)>3).sum() == 0:
                    return False
                else:
                    if self.last_putcard_type != 5:
                        return True
                    else:
                        puted_card = np.argwhere(self.last_puted_cards.sum(0)==4.)[0][0]
                        if (np.argwhere(state_playerself.sum(0)>3)>puted_card).sum():
                            return True
                        else:
                            return False
                
            def check_cards():
                
#                print('self.last_putcard_type',self.last_episode_end,self.last_putcard_type)
                
                if self.last_putcard_type == 0:
                    puted_card = np.argwhere(self.last_puted_cards.sum(0)==1.)[0][0]
                    if (np.argwhere(state_playerself.sum(0)>0)>puted_card).sum():
                        return True
                    else:
                        return False
                if self.last_putcard_type == 1:
                    puted_card = np.argwhere(self.last_puted_cards.sum(0)==2.)[0][0]
                    if (np.argwhere(state_playerself.sum(0)>1)>puted_card).sum():
                        return True
                    else:
                        return False
                if self.last_putcard_type == 3:
                    puted_card = np.argwhere(self.last_puted_cards.sum(0)==3.)[0][0]
                    if (np.argwhere(state_playerself.sum(0)>2)>puted_card).sum():
                        return True
                    else:
                        return False
                if self.last_putcard_type == 2:
                    puted_card = np.argwhere(self.last_puted_cards.sum(0)==3.)[0][0]
                    if (np.argwhere(state_playerself.sum(0)>2)>puted_card).sum() and state_playerself.sum()>3:
                        return True
                    else:
                        return False
                if self.last_putcard_type == 4:
                    puted_card = np.argwhere(self.last_puted_cards.sum(0)==4.)[0][0]
                    if (np.argwhere(state_playerself.sum(0)>3)>puted_card).sum() and state_playerself.sum()>4:
                        return True
                    else:
                        return False
                if self.last_putcard_type == 5:
                    return False
                if self.last_putcard_type == 6:
                    puted_card = np.argwhere(self.last_puted_cards.sum(0)==1.)[0][0]
                    numcards   = int(self.last_puted_cards.sum())
                    conv_tmp   =np.ones(numcards)
                    flag_t     = False
                    for tmp_id in range(puted_card,13-numcards+1):
                        if ((state_playerself.sum(0)>0)[tmp_id:tmp_id+5] * conv_tmp).sum() == self.last_puted_cards.sum():
#                            legal_act3[tmp_id] = 1.
                            flag_t =  True
                            break
                    return flag_t
                    
            checkbiggerboom = check_bigger_boom()
            checkcards = check_cards()
            legal_act1 = np.zeros(self.action_space[0])
            legal_act2 = np.zeros(self.action_space[1])
            legal_act3 = np.zeros(self.action_space[2])
            legal_act1[-1] = 1.
            
            if checkbiggerboom==False and checkcards==False:
                action1 = 5
                action2 = 13
                action3 = 13
                yaobuqi = True
                
            
            
            if checkbiggerboom and checkcards==False:
                legal_act1[3] = 1.
                output1_log = F.softmax(output1)
                output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
                action1 = output1_log_mask.view(-1).max(0)[1].item()
                puted_card = -1
                if action1 == 5:
                    action2 = 13
                    action3 = 13
                else:
                    if self.last_putcard_type == 5:
                        puted_card = np.argwhere(self.last_puted_cards.sum(0)==4.)[0][0]
                    booms = np.argwhere(state_playerself.sum(0)>3)
                    action2 = booms[booms>puted_card].min()
                    action3 = 13
                    self.last_putcard_type =5
            if checkcards:
                if self.last_putcard_type < 2:
#                    action1 = 0
                    legal_act1[self.last_putcard_type] = 1
                    output1_log = F.softmax(output1)
                    output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
                    action1 = output1_log_mask.view(-1).max(0)[1].item()
                    if action1 != 5:
                        puted_card = np.argwhere(self.last_puted_cards.sum(0)==self.last_putcard_type+1)[0][0]
                        legal_location_tmp = np.argwhere(state_playerself.sum(0) > self.last_putcard_type)
                        legal_location    = legal_location_tmp[legal_location_tmp>puted_card]
                        legal_act2[legal_location] = 1.
                        output2_log = F.softmax(output2)
                        output2_log_mask = output2_log*torch.FloatTensor(legal_act2)
                        action2 = output2_log_mask.view(-1).max(0)[1].item()
                    
                    else:
                        action2 = 13
                    action3 = 13
                if self.last_putcard_type == 3:
#                    action1 = 0
                    legal_act1[2] = 1
                    output1_log = F.softmax(output1)
                    output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
                    action1 = output1_log_mask.view(-1).max(0)[1].item()
                    if action1 != 5:
                        puted_card = np.argwhere(self.last_puted_cards.sum(0)==3.)[0][0]
                        legal_location_tmp = np.argwhere(state_playerself.sum(0) >2)
                        legal_location    = legal_location_tmp[legal_location_tmp>puted_card]
                        legal_act2[legal_location] = 1.
                        output2_log = F.softmax(output2)
                        output2_log_mask = output2_log*torch.FloatTensor(legal_act2)
                        action2 = output2_log_mask.view(-1).max(0)[1].item()
                    
                    else:
                        action2 = 13
                    action3 = 13
                if self.last_putcard_type == 2:
#                    action1 = 0
                    legal_act1[2] = 1
                    output1_log = F.softmax(output1)
                    output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
                    action1 = output1_log_mask.view(-1).max(0)[1].item()
                    if action1 != 5:
                        puted_card = np.argwhere(self.last_puted_cards.sum(0)==3.)[0][0]
                        legal_location_tmp = np.argwhere(state_playerself.sum(0) >2)
                        legal_location    = legal_location_tmp[legal_location_tmp>puted_card]
                        legal_act2[legal_location] = 1.
                        output2_log = F.softmax(output2)
                        output2_log_mask = output2_log*torch.FloatTensor(legal_act2)
                        action2 = output2_log_mask.view(-1).max(0)[1].item()
                    
                    else:
                        action2 = 13
                    legal_act3 = np.zeros(self.action_space[2])
                    legal_act3[-1] = 0
                    legal_location_tmp = np.argwhere(state_playerself.sum(0) >0)
                    legal_act3[legal_location_tmp] = 1.
                    legal_act3[action2]  = 0
                    output3_log = F.softmax(output3)
                    output3_log_mask = output3_log*torch.FloatTensor(legal_act3)
                    action3 = output3_log_mask.view(-1).max(0)[1].item()
                if self.last_putcard_type == 4:
#                    action1 = 0
                    legal_act1[3] = 1
                    output1_log = F.softmax(output1)
                    output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
                    action1 = output1_log_mask.view(-1).max(0)[1].item()
                    if action1 != 5:
                        puted_card = np.argwhere(self.last_puted_cards.sum(0)==4.)[0][0]
                        legal_location_tmp = np.argwhere(state_playerself.sum(0) >3)
                        legal_location    = legal_location_tmp[legal_location_tmp>puted_card]
                        legal_act2[legal_location] = 1.
                        output2_log = F.softmax(output2)
                        output2_log_mask = output2_log*torch.FloatTensor(legal_act2)
                        action2 = output2_log_mask.view(-1).max(0)[1].item()
                    
                    else:
                        action2 = 13
                    legal_act3 = np.zeros(self.action_space[2])
                    legal_act3[-1] = 0
                    legal_location_tmp = np.argwhere(state_playerself.sum(0) >0)
                    legal_act3[legal_location_tmp] = 1.
                    legal_act3[action2]  = 0
                    output3_log = F.softmax(output3)
                    output3_log_mask = output3_log*torch.FloatTensor(legal_act3)
                    action3 = output3_log_mask.view(-1).max(0)[1].item()
                if self.last_putcard_type == 6:
                    legal_act1[4] = 1
                    output1_log = F.softmax(output1)
                    output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
                    action1 = output1_log_mask.view(-1).max(0)[1].item()
                    if action1 != 5:
                        puted_card = np.argwhere(self.last_puted_cards.sum(0)==1.)[0][0]
                        numcards   = int(self.last_puted_cards.sum())
                        
                        action2 = numcards-5

                        
                        conv_tmp   =np.ones(numcards)
                        flag_t     = False
                        for tmp_id in range(puted_card,13-numcards+1):
                            if ((state_playerself.sum(0)>0)[tmp_id:tmp_id+5] * conv_tmp).sum() == self.last_puted_cards.sum():
                                legal_act3[tmp_id] = 1.
                        output3_log = F.softmax(output3)
                        output3_log_mask = output3_log*torch.FloatTensor(legal_act3)
                        action3 = output3_log_mask.view(-1).max(0)[1].item()
                    else:
                        action2 = 13
                    
                        action3 = 13
                if self.last_putcard_type == 5:
                    if checkbiggerboom:
                        legal_act1[3] = 1
                        output1_log = F.softmax(output1)
                        output1_log_mask = output1_log*torch.FloatTensor(legal_act1)
                        action1 = output1_log_mask.view(-1).max(0)[1].item()
                        action2 = 13
                    
                        action3 = 13
                    else:
                        action1 = 5
                        action2 = 13
                    
                        action3 = 13
#            print('checks:',checkbiggerboom , checkcards)
#            if 1: #self.last_putcard_type!= 0:
#                print(checkcards,checkbiggerboom,self.last_putcard_type)   
            cards_toput = np.zeros((4,13))
            if action1 == 0:
                cards_toput[-1,action2] =1.
                putcards_type = 0
            if action1 == 1:
                cards_toput[-1,action2] =1.
                cards_toput[-2,action2] =1.
                putcards_type = 1
            if action1 == 2:
                cards_toput[-1,action2] =1.
                cards_toput[-2,action2] =1.
                cards_toput[-3,action2] =1.
                putcards_type = 3
                if action2==action3:
                    print('error: action2==action3')
                if action3!=13:
                    cards_toput[-1,action3] = 1.  #3+1
                    putcards_type = 2
            if action1 == 3:
                cards_toput[-1,action2] =1.
                cards_toput[-2,action2] =1.
                cards_toput[-3,action2] =1.
                cards_toput[-4,action2] =1.
                putcards_type = 5
                if action2==action3:
                    print('error: action2==action3')
                if action3!=13:
                    cards_toput[-1,action3] = 1.   #4+1
                    putcards_type = 4
            if action1 == 4:            #0 is 5 
                for toput_id in range(13):
                    if toput_id>=action3 and toput_id < action3 + action2 + 5:
                        cards_toput[-1,toput_id] = 1.
                putcards_type = 6
            if action1 == 5:   
                putcards_type = 7
#            print(cards_toput,state_playerself,putcards_type)
            episode_end,game_end = self.put_cards(cards_toput,state_playerself,putcards_type)
            
#                    if (np.argwhere(state_playerself.sum(0) > 0) > puted_card).sum():
#                        return True
                    
                
                        
            
        return self.State_input.copy(),[output1,output2,output3],[action1,action2,action3],episode_end,game_end,yaobuqi
            
        
        
        
        
    def start_self_play(self,player = 'A'):
        
#        p1, p2 = self.board.players
        step = 0
        loss0 = 0.
        win0sum = 0.
        while True:
            self.init_state()
            game_end = False
            states_ = []
            actions_= []
            rewards_= []
            padded_num = 16
            entroy_loss=nn.CrossEntropyLoss()
            actions_test = []
            
            while not game_end:
                state,Q_net_outputs,actions,episode_end,game_end,yaobuqi = self.play_step()
                actions_test.append(actions)
#                if yaobuqi:
#                    rewards_.append(0)
#                    continue
                state_self = np.sort(np.argwhere(state[:4,:]==1)[:,1])
                state_opp  = np.sort(np.argwhere(state[4:8,:]==1)[:,1])
                state_puted_last = np.sort(np.argwhere(state[8:12,:]==1)[:,1])
                state_self_pad = np.pad(state_self,(0,padded_num-len(state_self)),'constant',constant_values = 14)
                state_opp_pad  = np.pad(state_opp,(0,padded_num-len(state_opp)),'constant',constant_values = 14)
                state_puted_last_pad = np.pad(state_puted_last,(0,padded_num-len(state_puted_last)),'constant',constant_values = 14)

                states_.append(np.concatenate([state_self_pad,state_opp_pad ,state_puted_last_pad]))
                actions_.append(actions)
                if self.player_current == 1:
                    rewards_.append(1)
                else:
                    rewards_.append(-1)
            data_len = len(states_)
            
            if 1-data_len%2 == test_player:
                win0sum += 1
#            if win0sum/(step+0.001) >0.6:
            print(111111111111,win0sum/(step+1))
            rewards_ = rewards_[::-1]
            rewards_ = [x for x in rewards_ if x != 0]
            if len(rewards_) < 10:
                print('len < 10:',states_,actions_)
#            rewards_ =rewards_[::-1]
            input    =  np.stack(states_)
            acts     =  np.stack(actions_)
            target   =  np.stack(rewards_)
            input    = Variable(torch.LongTensor(input))
            acts     = torch.LongTensor(acts)
            target    = Variable(torch.FloatTensor(target))
            
            
#            while True:
                
            step+=1
#                for i in range(data_len):
            optimizer.zero_grad()
            output = Q_net(input)
            Q_value0 = torch.gather(output[0],1, acts[:,0].unsqueeze(1)).view(-1) #(output[0])[np.arange(data_len),acts[:,0]] # torch.gather(output[0],1, acts[:,0].unsqueeze(1)).view(-1)
            Q_value1 = output[1].gather(1, acts[:,1].unsqueeze(1))
            Q_value2 = output[2].gather(1, acts[:,2].unsqueeze(1))
            Qvalue   = (Q_value0.view(-1) + Q_value1.view(-1) + Q_value2.view(-1))/3
            loss = F.mse_loss(Qvalue,target) #torch.abs(Q_value0-target).mean() #F.mse_loss(Q_value0,target)
            loss0 += loss.item()
            if step%5000==0:
                torch.save(Q_net,f'model/Qnet_model{step}.model')
            if step%100==0:
                print('step:' , step,'loss: ',loss.item())
                print(output[0].gather(1, acts[:,0].unsqueeze(1)).view(-1),target,'loss:',loss0/100)
                loss0 = 0.

                
training_pipeline = Trainer()
training_pipeline.run()
