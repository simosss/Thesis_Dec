from pykeen.models import ERModel, RGCN
from pykeen.nn import Embedding
import torch
from pykeen.nn.modules import DistMultInteraction
from pykeen.datasets import Nations
from pykeen.training import SLCWATrainingLoop
from torch.optim import Adam
import pandas as pd
import json
from pykeen.triples import TriplesFactory


# Pykeen class
class PretrainedInitializer:
    """
    Initialize tensor with pretrained weights.

    Example usage:

    .. code-block::

        import torch
        from pykeen.pipeline import pipeline
        from pykeen.nn.init import create_init_from_pretrained

        # this is usually loaded from somewhere else
        # the shape must match, as well as the entity-to-id mapping
        pretrained_embedding_tensor = torch.rand(14, 128)

        result = pipeline(
            dataset="nations",
            model="transe",
            model_kwargs=dict(
                embedding_dim=pretrained_embedding_tensor.shape[-1],
                entity_initializer=PretrainedInitializer(tensor=pretrained_embedding_tensor),
            ),
        )
    """

    def __init__(self, tensor: torch.FloatTensor) -> None:
        """
        Initialize the initializer.

        :param tensor:
            the tensor of pretrained embeddings.
        """
        self.tensor = tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the tensor with the given tensor."""
        if x.shape != self.tensor.shape:
            raise ValueError(
                f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
        return self.tensor.to(device=x.device, dtype=x.dtype)


class DistMult(ERModel):
    def __init__(
            self,
            ent_emb,
            rel_emb,
            random_seed=42,
            **kwargs
    ) -> None:
        super().__init__(
            interaction=DistMultInteraction(),
            entity_representations=Embedding(
                num_embeddings=ent_emb.shape[0],
                embedding_dim=3,
                initializer=PretrainedInitializer(tensor=ent_emb),
                trainable=True,
            ),

            relation_representations=Embedding(
                num_embeddings=rel_emb.shape[0],
                embedding_dim=3,
                initializer=PretrainedInitializer(tensor=rel_emb),
                trainable=True,
            ),

            random_seed=random_seed,
            **kwargs
        )


class RGCN_sim(RGCN):
    def __init__(
            self,
            ent_emb,
            num_layers=2,
            random_seed=42,
            **kwargs
    ) -> None:
        super().__init__(
            interaction=DistMultInteraction(),
            num_layers=num_layers,
            base_entity_initializer=Embedding(
                num_embeddings=ent_emb.shape[0],
                embedding_dim=13,
                initializer=PretrainedInitializer(tensor=ent_emb),
                trainable=True,
            ),
            random_seed=random_seed,
            **kwargs
        )


#dataset = Nations()
# entity_embeddings = {0: [1, 2, 3],
#                      1: [4, 5, 6],
#                      2: [1, 2, 3],
#                      3: [4, 5, 6],
#                      4:[1, 2, 3],
#                      5:[4, 5, 6],
#                      6:[1, 2, 3],
#                      7:[4, 5, 6],
#                      8:[1, 2, 3],
#                      9:[4, 5, 6],
#                      10:[1, 2, 3],
#                      11:[4, 5, 6],
#                      12:[1, 2, 3],
#                      13:[4, 5, 6]
#                      }
# relation_embedding = {0: [1, 2, 3],
#                      1: [4, 5, 6],
#                      2: [1, 2, 3],
#                      3: [4, 5, 6],
#                      4:[1, 2, 3],
#                      5:[4, 5, 6],
#                      6:[1, 2, 3],
#                      7:[4, 5, 6],
#                      8:[1, 2, 3],
#                      9:[4, 5, 6],
#                      10:[1, 2, 3],
#                      11:[4, 5, 6],
#                      12:[1, 2, 3],
#                      13:[4, 5, 6],
#                      14: [4, 5, 6],
#                      15: [1, 2, 3],
#                      16: [4, 5, 6],
#                               17:[1, 2, 3],
#                      18:[4, 5, 6],
#                      19:[1, 2, 3],
#                      20:[4, 5, 6],
#                      21:[1, 2, 3],
#                               22:[1, 2, 3],
#                      23:[4, 5, 6],
#                      24:[1, 2, 3],
#                      25:[4, 5, 6],
#                               26:[1, 2, 3],
#                      27:[4, 5, 6],
#                      28: [4, 5, 6],
#                      29: [1, 2, 3],
#                      30: [4, 5, 6],
#                               31:[1, 2, 3],
#                      32:[4, 5, 6],
#                      33:[1, 2, 3],
#                      34:[4, 5, 6],
#                               35:[1, 2, 3],
#                      36:[4, 5, 6],
#                      37:[1, 2, 3],
#                      38:[4, 5, 6],
#                               39:[1, 2, 3],
#                      40:[4, 5, 6],
#                      41: [4, 5, 6],
#                      42: [1, 2, 3],
#                      43: [4, 5, 6],
#                               44:[1, 2, 3],
#                      45:[4, 5, 6],
#                      46:[1, 2, 3],
#                      47:[4, 5, 6],
#                               48:[1, 2, 3],
#                      49:[4, 5, 6],
#                      50:[1, 2, 3],
#                      51:[4, 5, 6],
#                               52:[1, 2, 3],
#                      53:[4, 5, 6],
#                      54:[4, 5, 6]}

with open(f'results/feature_vectors_mono_mini.json', 'r') as f:
    entity_embeddings = json.load(f)

# TODO need to adjust the order of these currently not ordered
df = pd.DataFrame(entity_embeddings)
array = df.values.T
pe_ent = torch.Tensor(array).to(torch.long)


# df2 = pd.DataFrame(relation_embedding)
# array2 = df2.values.T
# pe_rel = torch.Tensor(array2)

tf = TriplesFactory.from_path(f'data/rare/rare_500_50.csv')
training, testing, validation = tf.split([.8, .1, .1])

rgcn_model = RGCN_sim(ent_emb=pe_ent.to(torch.int64),
                      triples_factory=training)

distmult_model = DistMult(ent_emb=pe_ent,
                          rel_emb=pe_rel,
                          triples_factory=dataset.training)


# Pick an optimizer from Torch
optimizer = Adam(params=distmult_model.parameters())

# Pick a training approach (sLCWA or LCWA)
training_loop = SLCWATrainingLoop(
    model=distmult_model,
    triples_factory=dataset.training,
    optimizer=optimizer,
)

# Train like Cristiano Ronaldo
_ = training_loop.train(
    triples_factory=dataset.training,
    num_epochs=5,
    batch_size=256,
)
