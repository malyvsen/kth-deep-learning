import numpy as np
from .rnn import RNN
from .text import Text


def synthesize(network: RNN, training_text: Text, length: int):
    state = [network.initial_state]
    character_id = training_text.character_ids["."]
    result = []
    for step in range(length):
        state, output = network.step(
            state, [character_id], new_state_gradient=None, output_gradient=None
        )
        character_id = np.random.choice(
            list(training_text.character_ids.values()), p=output[0]
        )
        result.append(training_text.id_characters[character_id])
    return "".join(result)
