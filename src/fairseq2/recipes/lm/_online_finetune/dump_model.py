from fairseq2.setup import setup_fairseq2
from fairseq2.recipes.common import (
    create_checkpoint_manager,
    setup_gangs,
    setup_model,
)
from fairseq2.recipes.lm import InstructionFinetuneConfig
from fairseq2.models.decoder import DecoderModel
from fairseq2.context import get_runtime_context
from pathlib import Path

setup_fairseq2()

context = get_runtime_context()

config = InstructionFinetuneConfig()

config.gang.tensor_parallel_size=8
model_name = "llama3_3_70b_instruct"
config.model.name = model_name

gangs = setup_gangs(context, config)
output_dir = Path(f"/checkpoint/ram/kulikov/dump_{model_name}")
checkpoint_manager = create_checkpoint_manager(context, gangs, output_dir)

model = setup_model(
        DecoderModel, context, config, output_dir, gangs, checkpoint_manager
    )

checkpoint_manager.begin_checkpoint(step_nr=0)
print("saving the model")
checkpoint_manager.save_model(model=model)
checkpoint_manager.commit_checkpoint()
