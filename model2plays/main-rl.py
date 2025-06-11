import itertools
import datetime
import sys
import string
import time
import pickle

from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.rl import SimpleAgent
from spirecomm.spire.character import PlayerClass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


if __name__ == "__main__":
    agent = SimpleAgent()
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

    crit_crit = nn.L1Loss()
    opt_crit = optim.Adam(agent.critic.parameters(), lr=0.0001, eps=0.1)
    opt_act = optim.Adam(agent.actor.parameters(), lr=0.0001, eps=0.1)

    # Play games forever, only as Ironclad
    while True:
        agent.change_class(PlayerClass.IRONCLAD)
        agent.reset()

        coordinator.play_one_game(PlayerClass.IRONCLAD)

        #train critic
        for i, batch in enumerate(agent.rb):
            if i > len(agent.rb)/agent.rb_batch_size*0.1:
                break

            opt_crit.zero_grad()
            inputs = torch.cat((batch['state'], batch['action']), 1).to(agent.device)
            outputs = agent.critic(inputs)
            loss = crit_crit(outputs, batch['reward'].float().to(agent.device))
            loss.backward()
            opt_crit.step()

        #train actor
        for i, batch in enumerate(agent.rb):
            if i > len(agent.rb)/agent.rb_batch_size*0.1:
                break

            inputs = batch['state'].to(agent.device)
            outputs = agent.actor(inputs)

            #get critic's evaluation of playing the card with ID `i`
            def i2val(i):
                cards = F.one_hot(torch.tensor(i), agent.len_playable).to(agent.device)
                cards = cards.repeat(agent.rb_batch_size,1)
                inputs2 = torch.cat((inputs, cards), 1)
                outputs2 = agent.critic(inputs2)
                return outputs2

            #get critic's evaluation of each possible play
            values = [i2val(i) for i in range(agent.len_playable)]
            values = torch.cat(values, 1)

            #estimate value of the average play for the given turn
            expected = outputs * values
            expected = expected.sum(1, True)

            #see if actual play was better or worse than the average play
            observation = batch['reward'].to(agent.device)
            advantage = observation - expected
            advantage = advantage.sum()

            #encourage better moves and discourage worse moves
            opt_act.zero_grad()
            advantage.backward()
            opt_act.step()

        #save neural net weights
        torch.save(agent.critic.state_dict(), 'shadorbs/critic')
        torch.save(agent.actor.state_dict(), 'shadorbs/actor')

