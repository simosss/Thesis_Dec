from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import TransE
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
import pandas as pd

n_tokeep = 300
minimum = 500

tf = TriplesFactory.from_path(f'data/rare/rare_{minimum}_{n_tokeep}.csv')
training, testing, validation = tf.split([.8, .1, .1], random_state=2253012122)
training_triples = pd.DataFrame(training.triples)
training_triples.to_csv('rare_training.csv', index=False)
testing_triples = pd.DataFrame(testing.triples)
testing_triples.to_csv('rare_testing.csv', index=False)


# Pick model, optimizer, training approach
model = TransE(triples_factory=training)
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
