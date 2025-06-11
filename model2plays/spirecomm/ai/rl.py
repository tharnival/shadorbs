import time
import random

from spirecomm.spire.game import Game
from spirecomm.spire.character import Intent, PlayerClass
import spirecomm.spire.card
from spirecomm.spire.screen import RestOption
from spirecomm.communication.action import *
from spirecomm.ai.priorities import *

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage
from tensordict.tensordict import TensorDict


ICards = {'Bash':True,
         'Defend_R':False,
         'Strike_R':True,
         'Anger':True,
         'Armaments':False,
         'Body Slam':True,
         'Clash':True,
         'Cleave':False,
         'Clothesline':True,
         'Flex':False,
         'Havoc':False,
         'Headbutt':True,
         'Heavy Blade':True,
         'Iron Wave':True,
         'Perfected Strike':True,
         'Pommel Strike':True,
         'Shrug It Off':False,
         'Sword Boomerang':False,
         'Thunderclap':False,
         'True Grit':False,
         'Twin Strike':True,
         'Warcry':False,
         'Wild Strike':True,
         'Battle Trance':False,
         'Blood for Blood':True,
         'Bloodletting':False,
         'Burning Pact':False,
         'Carnage':True,
         'Combust':False,
         'Dark Embrace':False,
         'Disarm':True,
         'Dropkick':True,
         'Dual Wield':False,
         'Entrench':False,
         'Evolve':False,
         'Feel No Pain':False,
         'Fire Breathing':False,
         'Flame Barrier':False,
         'Ghostly Armor':False,
         'Hemokinesis':True,
         'Infernal Blade':False,
         'Inflame':False,
         'Intimidate':False,
         'Metallicize':False,
         'Power Through':False,
         'Pummel':True,
         'Rage':False,
         'Rampage':True,
         'Reckless Charge':True,
         'Rupture':False,
         'Searing Blow':True,
         'Second Wind':False,
         'Seeing Red':False,
         'Sentinel':False,
         'Sever Soul':True,
         'Shockwave':False,
         'Spot Weakness':True,
         'Uppercut':True,
         'Whirlwind':False,
         'Barricade':False,
         'Berserk':False,
         'Bludgeon':True,
         'Brutality':False,
         'Corruption':False,
         'Demon Form':False,
         'Double Tap':False,
         'Exhume':False,
         'Feed':True,
         'Fiend Fire':False,
         'Immolate':False,
         'Impervious':False,
         'Juggernaut':False,
         'Limit Break':False,
         'Offering':False,
         'Reaper':False
         }

GCards = {'Bandage Up':False,
         'Blind':True,
         'Dark Shackles':True,
         'Deep Breath':False,
         'Discovery':False,
         'Dramatic Entrance':False,
         'Enlightenment':False,
         'Finesse':False,
         'Flash of Steel':True,
         'Forethought':False,
         'Good Instincts':False,
         'Impatience':False,
         'Jack Of All Trades':False,
         'Madness':False,
         'Mind Blast':True,
         'Panacea':False,
         'PanicButton':False,
         'Purity':False,
         'Swift Strike':True,
         'Trip':True,
         'Apotheosis':False,
         'Chrysalis':False,
         'HandOfGreed':True,
         'Magnetism':False,
         'Master of Strategy':False,
         'Mayhem':False,
         'Metamorphosis':False,
         'Panache':False,
         'Sadistic Nature':False,
         'Secret Technique':False,
         'Secret Weapon':False,
         'The Bomb':False,
         'Thinking Ahead':False,
         'Transmutation':False,
         'Violence':False,
         'Apparition':False,
         #'Beta':False,
         'Bite':True,
         'Expunger':True,
         'Insight':False,
         'J.A.X.':False,
         #'Miracle':False,
         #'Omega':False,
         'RitualDagger':True,
         'Safety':False,
         #'Shiv':True,
         'Smite':True,
         'Through Violence':True,
         'Slimed':False}
BCards = ['Burn',
          'Dazed',
          'Wound',
          'Void',
          "AscendersBane",
          'Clumsy',
          'Curse of the Bell',
          'Decay',
          'Doubt',
          'Injury',
          'Necronomicurse',
          'Normality',
          'Pain',
          'Parasite',
          'Pride',
          'Regret',
          'Shame',
          'Writhe']

playable = ICards | GCards
cardNames = list(playable.keys()) + BCards
cardIDs = dict(zip(cardNames, range(len(cardNames))))

#the neural net for the actor
class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.fc1 = nn.Linear((len(cardNames)+1) * 10 + 6, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, len(playable))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x):
        canplay = torch.tensor([-1000]).repeat((x.size()[0], len(playable))).float().to(self.device)
        for c in range(10):
            canplay += x[:,c*(len(cardNames)+1):c*(len(cardNames)+1)+len(playable)]*2000
        canplay = self.sigmoid(canplay)

        x = torch.concat((x, torch.zeros((x.size()[0], 1)).float().to(self.device)), 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = x * canplay
        x = self.softmax(x)
        return x

#the neural net for the critic
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear((len(cardNames)+1) * 10 + 5 + len(playable), 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 1000)
        self.fc4 = nn.Linear(1000, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SimpleAgent:

    def __init__(self, chosen_class=PlayerClass.THE_SILENT):
        self.game = Game()
        self.errors = 0
        self.choose_good_card = False
        self.skipped_cards = False
        self.visited_shop = False
        self.map_route = []
        self.chosen_class = chosen_class
        self.priorities = Priority()
        self.change_class(chosen_class)

        self.len_playable = len(playable)
        self.len_cards = len(cardNames)

        #buffer used to store transitions
        # before the discounted cumulative reward can be calculated
        self.turn_buffer = []

        #the replay buffer contains the transitions with discounted cumulative rewards
        #it is used to train the model between each run
        self.rb_batch_size = 20
        self.rb = ReplayBuffer(
            storage=ListStorage(max_size=20_000),
            batch_size=self.rb_batch_size
            )

        self.gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.gpu else "cpu")

        self.actor = Net(self.device)
        self.critic = Net2()
        if(self.gpu):
            self.actor.to(self.device)
            self.critic.to(self.device)

        #uncomment if you have pretrained models
        #self.actor.load_state_dict(torch.load('shadorbs/actor', map_location=self.device))
        #self.critic.load_state_dict(torch.load('shadorbs/critic', map_location=self.device))

        #probability that the actor will ignore the model and make a random move
        self.explore = 0.1

        self.reset()

    def reset(self):
        self.first_turn = True
        self.prev_state = 0
        self.prev_action = 0
        self.prev_hp_diff = 0
        self.was_in_battle = False

    def change_class(self, new_class):
        self.chosen_class = new_class
        if self.chosen_class == PlayerClass.THE_SILENT:
            self.priorities = SilentPriority()
        elif self.chosen_class == PlayerClass.IRONCLAD:
            self.priorities = IroncladPriority()
        elif self.chosen_class == PlayerClass.DEFECT:
            self.priorities = DefectPowerPriority()
        else:
            self.priorities = random.choice(list(PlayerClass))

    def handle_error(self, error):
        raise Exception(error)

    def get_next_action_in_game(self, game_state):
        self.game = game_state
        #time.sleep(0.07)
        if self.game.choice_available:
            if self.was_in_battle:
                self.log_transition(end=True)
                self.was_in_battle = False
            return self.handle_screen()
        if self.game.proceed_available:
            return ProceedAction()
        if self.game.play_available:
            if self.game.room_type == "MonsterRoomBoss" and len(self.game.get_real_potions()) > 0:
                potion_action = self.use_next_potion()
                if potion_action is not None:
                    return potion_action
            return self.get_play_card_action()
        if self.game.end_available:
            self.log_transition(end=False)
            return EndTurnAction()
        if self.game.cancel_available:
            return CancelAction()

    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class)

    def is_monster_attacking(self):
        for monster in self.game.monsters:
            if monster.intent.is_attack() or monster.intent == Intent.NONE:
                return True
        return False

    def get_incoming_damage(self):
        incoming_damage = 0
        for monster in self.game.monsters:
            if not monster.is_gone and not monster.half_dead:
                if monster.move_adjusted_damage is not None:
                    incoming_damage += monster.move_adjusted_damage * monster.move_hits
                elif monster.intent == Intent.NONE:
                    incoming_damage += 5 * self.game.act
        return incoming_damage

    def get_low_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = min(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def get_high_hp_target(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        best_monster = max(available_monsters, key=lambda x: x.current_hp)
        return best_monster

    def many_monsters_alive(self):
        available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
        return len(available_monsters) > 1

    #convert a card to a tensor that can be passed to a neural net
    def card2tensor(self, card, upgraded, played=False):
        card = cardIDs[card]
        if played:
            card = F.one_hot(torch.tensor(card), len(playable)+1)
        else:
            card = F.one_hot(torch.tensor(card), len(cardNames)+1)
        if upgraded > 0.5:
            card[-1] = 1
        return card

    #log a transition from one game state to another one into the replay buffer
    # so it can be used later for training
    def log_transition(self, end=False):
        #get a tensor representation of the current game state
        hand = [self.card2tensor(c.card_id, c.upgrades) for c in self.game.hand]
        if len(hand) > 0:
            hand = torch.cat(hand, 0)
        padding = torch.zeros(((10-len(self.game.hand))*(len(cardNames)+1)))
        misc = None
        if self.game.in_combat:
            misc = torch.tensor([self.game.player.current_hp/100, self.game.player.max_hp/100, self.game.player.block/100, self.game.player.energy/10, self.game.floor/100])
        else:
            misc = torch.tensor([self.game.current_hp/100, self.game.max_hp/100, 0, 0, self.game.floor/100])
        state = torch.cat((padding, misc), 0)
        if len(hand) > 0:
            state = torch.cat((hand, state), 0)

        #evaluate the current game state based on the difference between the player's hp and the monsters' hp
        hp_diff = 0
        if self.game.in_combat:
            expected_hp = self.game.player.current_hp - min(0, self.get_incoming_damage() - self.game.player.block)
            monsters_hp = 0
            available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
            for monster in available_monsters:
                monsters_hp += monster.current_hp
            hp_diff = expected_hp*2 - monsters_hp
        else:
            hp_diff = self.game.current_hp*2

        if self.first_turn:
            self.first_turn = False
        else:
            reward = hp_diff - self.prev_hp_diff

            done = 1 if end else 0

            #add transition data to turn buffer
            transition = TensorDict(
                {
                    "state": self.prev_state,
                    "action": self.card2tensor(self.prev_action.card_id, self.prev_action.upgrades, played=True)[:-1],
                    "reward": torch.tensor([reward]).float(),
                    "new_state": state,
                    "done": torch.tensor([done]),
                },
                batch_size=[],
            )
            self.turn_buffer.append(transition)

        self.prev_state = state
        self.prev_hp_diff = hp_diff

        #if end of battle
        if end:
            self.first_turn = True

            discounted = 0
            gamma = 0.95
            #calculate discounted cumulative rewards for each transition
            for tdict in reversed(self.turn_buffer):
                prev_sum = discounted
                discounted += tdict['reward']
                discounted = discounted * gamma
                tdict['reward'] += prev_sum

            #add updated transitions to replay buffer
            for tdict in self.turn_buffer:
                self.rb.add(tdict)

            self.turn_buffer = []

    def ask_actor(self):
        playable_cards = [card for card in self.game.hand if card.is_playable and card.card_id in playable]
        if len(playable_cards) == 0:
            return (False, EndTurnAction())

        #maybe make a random play for the sake of exploration
        if random.random() < self.explore:
            cardi = random.randint(0, len(playable_cards)-1)
            return (True, playable_cards[cardi])

        #convert current hand to a form that the neural net can take as input
        hand = [self.card2tensor(c.card_id, c.upgrades) for c in self.game.hand]
        hand = torch.cat(hand, 0)
        padding = torch.zeros(((10-len(self.game.hand))*(len(cardNames)+1)))
        misc = torch.tensor([self.game.player.current_hp/100, self.game.player.max_hp/100, self.game.player.block/100, self.game.player.energy/10, self.game.floor/100])
        state = torch.cat((hand, padding, misc), 0)

        if self.gpu:
            state = state.to(self.device)

        #get predictions from the neural net
        pred = self.actor(state.view(1,-1))
        pred = pred[0].cpu().detach().numpy()

        #the actor will assign a probability to each playable card,
        # giving better plays higher probabilities
        plays = np.zeros_like(pred)
        playable_ids = np.unique([cardIDs[c.card_id] for c in playable_cards])
        probs = []
        for i in playable_ids:
            probs.append(pred[i])
        probs = nn.Softmax(dim=0)(torch.tensor(probs))
        for i,idx in enumerate(playable_ids):
            plays[idx] = probs[i]

        #pick a random card based on the probabilities from the actor
        plays = plays.cumsum(0)
        id_to_play = torch.searchsorted(torch.tensor(plays), torch.rand(1))
        playable_cards = [c for c in playable_cards if cardIDs[c.card_id] == id_to_play]

        #decide on whether to play an upgraded or non-upgraded version of the card
        upgr = False
        nonupgr = False
        for card in playable_cards:
            if card.upgrades == 1:
                upgr = True
            else:
                nonupgr = True

        if upgr and nonupgr:
            upgr_pref = 1
            playable_cards = [c for c in playable_cards if c.upgrades == upgr_pref]

        return (True, playable_cards[0])

    def get_play_card_action(self):

        cont, card_to_play = self.ask_actor()
        if not cont:
            return EndTurnAction()

        self.log_transition()
        self.prev_action = card_to_play
        self.was_in_battle = True

        if card_to_play.has_target:
            available_monsters = [monster for monster in self.game.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
            if len(available_monsters) == 0:
                return EndTurnAction()

            if card_to_play.type == spirecomm.spire.card.CardType.ATTACK:
                target = self.get_low_hp_target()
            else:
                target = self.get_high_hp_target()

            return PlayCardAction(card=card_to_play, target_monster=target)
        else:
            return PlayCardAction(card=card_to_play)

    def use_next_potion(self):
        for potion in self.game.get_real_potions():
            if potion.can_use:
                if potion.requires_target:
                    return PotionAction(True, potion=potion, target_monster=self.get_low_hp_target())
                else:
                    return PotionAction(True, potion=potion)

    def handle_screen(self):
        if self.game.screen_type == ScreenType.EVENT:
            if self.game.screen.event_id in ["Vampires", "Masked Bandits", "Knowing Skull", "Ghosts", "Liars Game", "Golden Idol", "Drug Dealer", "The Library"]:
                return ChooseAction(len(self.game.screen.options) - 1)
            else:
                return ChooseAction(0)
        elif self.game.screen_type == ScreenType.CHEST:
            return OpenChestAction()
        elif self.game.screen_type == ScreenType.SHOP_ROOM:
            if not self.visited_shop:
                self.visited_shop = True
                return ChooseShopkeeperAction()
            else:
                self.visited_shop = False
                return ProceedAction()
        elif self.game.screen_type == ScreenType.REST:
            return self.choose_rest_option()
        elif self.game.screen_type == ScreenType.CARD_REWARD:
            return self.choose_card_reward()
        elif self.game.screen_type == ScreenType.COMBAT_REWARD:
            for reward_item in self.game.screen.rewards:
                if reward_item.reward_type == RewardType.POTION and self.game.are_potions_full():
                    continue
                elif reward_item.reward_type == RewardType.CARD and self.skipped_cards:
                    continue
                else:
                    return CombatRewardAction(reward_item)
            self.skipped_cards = False
            return ProceedAction()
        elif self.game.screen_type == ScreenType.MAP:
            return self.make_map_choice()
        elif self.game.screen_type == ScreenType.BOSS_REWARD:
            relics = self.game.screen.relics
            best_boss_relic = self.priorities.get_best_boss_relic(relics)
            return BossRewardAction(best_boss_relic)
        elif self.game.screen_type == ScreenType.SHOP_SCREEN:
            if self.game.screen.purge_available and self.game.gold >= self.game.screen.purge_cost:
                return ChooseAction(name="purge")
            for card in self.game.screen.cards:
                if self.game.gold >= card.price and not self.priorities.should_skip(card):
                    return BuyCardAction(card)
            for relic in self.game.screen.relics:
                if self.game.gold >= relic.price:
                    return BuyRelicAction(relic)
            return CancelAction()
        elif self.game.screen_type == ScreenType.GRID:
            if not self.game.choice_available:
                return ProceedAction()
            if self.game.screen.for_upgrade or self.choose_good_card:
                available_cards = self.priorities.get_sorted_cards(self.game.screen.cards)
            else:
                available_cards = self.priorities.get_sorted_cards(self.game.screen.cards, reverse=True)
            num_cards = self.game.screen.num_cards
            return CardSelectAction(available_cards[:num_cards])
        elif self.game.screen_type == ScreenType.HAND_SELECT:
            if not self.game.choice_available:
                return ProceedAction()
            # Usually, we don't want to choose the whole hand for a hand select. 3 seems like a good compromise.
            num_cards = min(self.game.screen.num_cards, 3)
            return CardSelectAction(self.priorities.get_cards_for_action(self.game.current_action, self.game.screen.cards, num_cards))
        else:
            return ProceedAction()

    def choose_rest_option(self):
        rest_options = self.game.screen.rest_options
        if len(rest_options) > 0 and not self.game.screen.has_rested:
            if RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp / 2:
                return RestAction(RestOption.REST)
            elif RestOption.REST in rest_options and self.game.act != 1 and self.game.floor % 17 == 15 and self.game.current_hp < self.game.max_hp * 0.9:
                return RestAction(RestOption.REST)
            elif RestOption.SMITH in rest_options:
                return RestAction(RestOption.SMITH)
            elif RestOption.LIFT in rest_options:
                return RestAction(RestOption.LIFT)
            elif RestOption.DIG in rest_options:
                return RestAction(RestOption.DIG)
            elif RestOption.REST in rest_options and self.game.current_hp < self.game.max_hp:
                return RestAction(RestOption.REST)
            else:
                return ChooseAction(0)
        else:
            return ProceedAction()

    def count_copies_in_deck(self, card):
        count = 0
        for deck_card in self.game.deck:
            if deck_card.card_id == card.card_id:
                count += 1
        return count

    def choose_card_reward(self):
        reward_cards = self.game.screen.cards
        if self.game.screen.can_skip and not self.game.in_combat:
            pickable_cards = [card for card in reward_cards if self.priorities.needs_more_copies(card, self.count_copies_in_deck(card))]
        else:
            pickable_cards = reward_cards
        if len(pickable_cards) > 0:
            potential_pick = self.priorities.get_best_card(pickable_cards)
            return CardRewardAction(potential_pick)
        elif self.game.screen.can_bowl:
            return CardRewardAction(bowl=True)
        else:
            self.skipped_cards = True
            return CancelAction()

    def generate_map_route(self):
        node_rewards = self.priorities.MAP_NODE_PRIORITIES.get(self.game.act)
        best_rewards = {0: {node.x: node_rewards[node.symbol] for node in self.game.map.nodes[0].values()}}
        best_parents = {0: {node.x: 0 for node in self.game.map.nodes[0].values()}}
        min_reward = min(node_rewards.values())
        map_height = max(self.game.map.nodes.keys())
        for y in range(0, map_height):
            best_rewards[y+1] = {node.x: min_reward * 20 for node in self.game.map.nodes[y+1].values()}
            best_parents[y+1] = {node.x: -1 for node in self.game.map.nodes[y+1].values()}
            for x in best_rewards[y]:
                node = self.game.map.get_node(x, y)
                best_node_reward = best_rewards[y][x]
                for child in node.children:
                    test_child_reward = best_node_reward + node_rewards[child.symbol]
                    if test_child_reward > best_rewards[y+1][child.x]:
                        best_rewards[y+1][child.x] = test_child_reward
                        best_parents[y+1][child.x] = node.x
        best_path = [0] * (map_height + 1)
        best_path[map_height] = max(best_rewards[map_height].keys(), key=lambda x: best_rewards[map_height][x])
        for y in range(map_height, 0, -1):
            best_path[y - 1] = best_parents[y][best_path[y]]
        self.map_route = best_path

    def make_map_choice(self):
        if len(self.game.screen.next_nodes) > 0 and self.game.screen.next_nodes[0].y == 0:
            self.generate_map_route()
            self.game.screen.current_node.y = -1
        if self.game.screen.boss_available:
            return ChooseMapBossAction()
        chosen_x = self.map_route[self.game.screen.current_node.y + 1]
        for choice in self.game.screen.next_nodes:
            if choice.x == chosen_x:
                return ChooseMapNodeAction(choice)
        # This should never happen
        return ChooseAction(0)

