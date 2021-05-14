import numpy as np
from .rnn import RNN
from .tweets import Tweets


def synthesize(network: RNN, training_tweets: Tweets, length: int):
    state = [network.initial_state]
    character_id = training_tweets.character_ids["."]
    result = []
    for char_count in range(0, length):
        state, output = network.step(
            state=state,
            input=[character_id],
            char_count=[char_count],
            new_state_gradient=None,
            output_gradient=None,
        )
        character_id = np.random.choice(
            list(training_tweets.character_ids.values()), p=output[0]
        )
        result.append(training_tweets.id_characters[character_id])
    return "".join(result)
