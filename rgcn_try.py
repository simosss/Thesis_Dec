from pykeen.models import ERModel, EntityRelationEmbeddingModel
from pykeen.nn import EmbeddingSpecification, Embedding
import torch
from pykeen.nn.modules import DistMultInteraction
from pykeen.datasets import Nations
import numpy as np
import pandas as pd
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
            raise ValueError(f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
        return self.tensor.to(device=x.device, dtype=x.dtype)


class Pretrained_Emb(Embedding):

    def __init__(self, ent_emb):
        self.ent_emb = ent_emb
        super().__init__(
            num_embeddings=ent_emb.shape[0],
            embedding_dim=ent_emb.shape[1],
            trainable=False)

    def forward(self, indices):
        return self.ent_emb[indices]


class DistMult(ERModel):
    def _init_(
            self,
            ent_emb,
            random_seed=42,
            **kwargs,
    ) -> None:
        super().__init__(
            interaction=DistMultInteraction(),
            entity_representations=Embedding(
                num_embeddings=ent_emb.shape[0],
                shape=ent_emb.shape[1],
                initializer=PretrainedInitializer(tensor=ent_emb),
                trainable=False,
            ),

            random_seed=random_seed,
            **kwargs
        )


dataset = Nations()
entity_embeddings = {0: [1, 2, 3],
                     1: [4, 5, 6], 2: [1, 2, 3], 3: [4, 5, 6],
                              4:[1, 2, 3], 5:[4, 5, 6], 6:[1, 2, 3], 7:[4, 5, 6],
                              8:[1, 2, 3], 9:[4, 5, 6], 10:[1, 2, 3], 11:[4, 5, 6],
                              12:[1, 2, 3], 13:[4, 5, 6]}


# Creates a dataframe with keys as index and values as cell values.
df = pd.DataFrame(entity_embeddings)
array = df.values.T
pe_ent = Pretrained_Emb(torch.Tensor(array))

# till here everything is right
distmult_model = DistMult(interaction=DistMultInteraction(),
                          entity_representations=pe_ent,
                          triples_factory=dataset.training)






# Get a training dataset
from pykeen.datasets import Nations
dataset = Nations()
training_triples_factory = dataset.training

# Pick a model
from pykeen.models import TransE
model = TransE(triples_factory=training_triples_factory)

# Pick an optimizer from Torch
from torch.optim import Adam
optimizer = Adam(params=model.get_grad_params())

# Pick a training approach (sLCWA or LCWA)
from pykeen.training import SLCWATrainingLoop
training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=training_triples_factory,
    optimizer=optimizer,
)

# Train like Cristiano Ronaldo
_ = training_loop.train(
    triples_factory=training_triples_factory,
    num_epochs=5,
    batch_size=256,
)