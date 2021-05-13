import numpy as np
from .rnn import RNN
from .tweets import Tweets


def synthesize(network: RNN, training_tweets: Tweets, length: int):
    state = [network.initial_state]
    character_id = training_tweets.character_ids["."]
    result = []
    for ttl in range(length, 0, -1):
        state, output = network.step(
            state=state,
            input=[character_id],
            ttl=[ttl],
            new_state_gradient=None,
            output_gradient=None,
        )
        character_id = np.random.choice(
            list(training_tweets.character_ids.values()), p=output[0]
        )
        result.append(training_tweets.id_characters[character_id])
    return "".join(result)
