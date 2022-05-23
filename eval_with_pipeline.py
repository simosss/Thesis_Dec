from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
import json

n_tokeep = 300
minimum = 500

tf = TriplesFactory.from_path(f'data/rare/rare_{minimum}_{n_tokeep}.csv')
training, testing, validation = tf.split([.8, .1, .1])


# read the feature vectors
with open(f'results/feature_vectors_mono_mini.json', 'r') as f:
    vectors = json.load(f)

result_pipeline = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='rescal',
    model_kwargs=dict(embedding_dim=300),
    training_kwargs=dict(#sampler="schlichtkrull",
                         checkpoint_name='rescale_checkpoint_300_.pt',
                         checkpoint_frequency=5,
                         num_epochs=60#,
                         #batch_size=128
),
    evaluator=RankBasedEvaluator,
    evaluator_kwargs=dict(ks=[10, 50]),
    stopper='early',
    stopper_kwargs=dict(
        metric='hits_at_k',
        frequency=4, patience=3, relative_delta=0.002)
)

result_pipeline.plot_losses()




from pykeen.models import RGCN
from torch.optim import Adam

# Pick model, optimizer, training approach
model = RGCN(triples_factory=training, embedding_dim=13, num_layers=2)
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
