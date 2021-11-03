from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline

n_tokeep = 200
minimum = 500

tf = TriplesFactory.from_path(f'data/rare/rare_{minimum}_{n_tokeep}.csv')
training, testing, validation = tf.split([.8, .1, .1])


result_pipeline = pipeline(
    training=training,
    testing=testing,
    model='RESCAL',
    epochs=100,
    evaluator=RankBasedEvaluator,
    evaluator_kwargs=dict(ks=[50])
)

result_pipeline.plot()
