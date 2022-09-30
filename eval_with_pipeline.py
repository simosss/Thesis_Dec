from pykeen.triples import TriplesFactory
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
import json
import pandas as pd

n_tokeep = 300
minimum = 500

tf = TriplesFactory.from_path(f'data/rare/rare_{minimum}_{n_tokeep}.csv')
training, testing, validation = tf.split([.8, .1, .1])


relations = pd.read_csv('per_se_transe_rare.csv')
relations = list(relations.columns)
relations.pop()

# read the feature vectors
#with open(f'results/feature_vectors_mono_mini.json', 'r') as f:
#    vectors = json.load(f)



# result_pipeline.plot_losses()
per_se_10 = dict()
per_se_50 = dict()


for idx, rel in enumerate(relations):
    if idx < 3:
        evaluation_relation_whitelist = {rel}

        result_pipeline = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model='RotatE',
            evaluation_relation_whitelist=evaluation_relation_whitelist,
            model_kwargs=dict(embedding_dim=500),
            training_kwargs=dict(checkpoint_name='rotate_checkpoin_rare.pt',
                                 checkpoint_frequency=5,
                                 num_epochs=60  # ,
                                 # batch_size=128
                                 ),
            evaluator=RankBasedEvaluator,
            evaluator_kwargs=dict(ks=[10, 50], batch_size=256)  # ,
            # stopper='early',
            # stopper_kwargs=dict(
            #    metric='hits_at_k',
            #    frequency=10, patience=3, relative_delta=0.002)
        )
        per_se_10[rel] = result_pipeline.metric_results.hits_at_k['both']['realistic'][10]
        per_se_50[rel] = result_pipeline.metric_results.hits_at_k['both']['realistic'][50]

#with open("per_se_50.json", "w") as outfile:
#    json.dump(per_se_50, outfile)
import csv
metrics = [per_se_10, per_se_50]
with open('per_se_rotate.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=relations)
    writer.writeheader()
    writer.writerows(metrics)

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
