from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
import json

n_tokeep = 300
minimum = 500

tf = TriplesFactory.from_path(f'data/rare/rare_{minimum}_{n_tokeep}.csv')
training, testing = tf.split([.8, .2])


result_pipeline = pipeline(
    training=training,
    testing=testing,
    model='RESCAL',
    model_kwargs=dict(embedding_dim=300),
    training_kwargs=dict(#sampler="schlichtkrull",
                         # checkpoint_name='RGCN_checkpointt.pt',
                         # checkpoint_frequency=5,
                         num_epochs=200#,
                         #batch_size=128
),
    evaluator=RankBasedEvaluator,
    evaluator_kwargs=dict(ks=[50])
)
result_pipeline.plot_losses()


result_pipeline.plot()
