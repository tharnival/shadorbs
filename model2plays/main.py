import itertools
import datetime
import sys

from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass

from spirecomm.ai.shadorbs import SimpleAgent



if __name__ == "__main__":
    agent = SimpleAgent()
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

    # Play games forever, only as Ironclad
    while True:
        agent.change_class(PlayerClass.IRONCLAD)
        result = coordinator.play_one_game(chosen_class)
