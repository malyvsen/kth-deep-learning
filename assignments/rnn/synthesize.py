import numpy as np
from tqdm.auto import trange
from crater.premade import RNN
from .text import Text


def synthesize(network: RNN, training_text: Text, length: int):
    state = [network.initial_state]
    character_id = training_text.character_ids["."]
    result = []
    for step in trange(length):
        state, output = network.step(state, [character_id])
        state = state.no_backward
        character_id = np.random.choice(
            list(training_text.character_ids.values()), p=output.numpy[0]
        )
        result.append(training_text.id_characters[character_id])
    return "".join(result)
