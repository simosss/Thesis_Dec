from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
import torch
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import RGCN
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop

n_tokeep = 999
minimum = 999

tf = TriplesFactory.from_path(f'data/rare/rare_{minimum}_{n_tokeep}.csv')
training, testing = tf.split([.9, .1])


np_array = np.array([[1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1],
                     [1,2,1]])

entity_initializer = lambda t: torch.as_tensor(np_array, dtype=torch.float32)


model = RGCN(triples_factory=training, entity_representations=entity_initializer)
optimizer = Adam(params=model.get_grad_params())
training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training,
    optimizer=optimizer,
)

# train
training_loop.train(
    triples_factory=training,
    num_epochs=500,
    batch_size=256,
)

# evaluate
evaluator = RankBasedEvaluator(ks=[50])
mapped_triples = testing.mapped_triples

foo = evaluator.evaluate(
    model=model,
    mapped_triples=mapped_triples,
    batch_size=1024
)
