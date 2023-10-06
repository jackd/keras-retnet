from keras_nlp.samplers import RandomSampler

from keras_retnet.backend import ops
from keras_retnet.models.retnet import RetnetCausalLM

preset = "retnet-base"

# NOTE: no weights currently available, so this will be gibberish.
lm = RetnetCausalLM.from_preset(preset, load_weights=False)
lm.backbone.summary()
tokenizer = lm.preprocessor.tokenizer
sampler = RandomSampler()
# sampler = "greedy"
lm.compile(sampler=sampler)

ctx = """\nIn a shocking finding, scientist discovered a herd of dragons living in a \
remote, previously unexplored valley, in Tibet. Even more surprising to the \
researchers was the fact that the dragons spoke perfect Chinese."""
output = lm.generate([ctx])
print(ops.convert_to_numpy(output[0]).item())
