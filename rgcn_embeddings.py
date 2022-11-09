"""I have some problem to feed an RGCN model with embedings"""

from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import RGCN
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
from pykeen.pipeline import pipeline
import json

tf = TriplesFactory.from_path(f'data/rare/rare_500_50.csv')
training, testing = tf.split([.8, .2])

# I can do sth like that and use randomly initialised feature vectors
result_pipeline = pipeline(
    training=training,
    testing=testing,
    model='RGCN',
    model_kwargs=dict(embedding_dim=13,
                      num_layers=2),
    training_kwargs=dict(sampler="schlichtkrull",
                         num_epochs=30),
    evaluator=RankBasedEvaluator,
    evaluator_kwargs=dict(ks=[50])
)

# Or like this
# Pick model, optimizer, training approach
model = RGCN(triples_factory=training)
optimizer = Adam(params=model.get_grad_params())
training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training,
    optimizer=optimizer
)

training_loop.train(
    triples_factory=training,
    num_epochs=35,
    batch_size=10024,
    sampler="schlichtkrull"
)

evaluator = RankBasedEvaluator(ks=[50])
mapped_triples = testing.mapped_triples

foo = evaluator.evaluate(
    model=model,
    mapped_triples=mapped_triples,
    batch_size=1024
)



# you can load a toy version of my feature vectors for each drug
with open(f'results/feature_vectors_mono_mini.json', 'r') as f:
    vectors = json.load(f)

# But reading the documentation and source code of the Pykeen RGCN implementation
# https://pykeen.readthedocs.io/en/stable/_modules/pykeen/models/unimodal/rgcn.html#RGCN
# I cannot understand how I could feed them into the model
