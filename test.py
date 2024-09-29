import torch
import modeling_mega
from timm.models import create_model
import argparse




def get_args():
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--model', default='model_mega')
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        init_values=12,
        vocab_size=8192
    )

    return model

model = get_model(get_args())

# model = modeling_mega.NeuralTransformerForMaskedEEGModeling()
model.train()
x = torch.rand(64, 64, 4, 200)
input_chans = torch.arange(0, 65)

output = model(x, input_chans, 0.75)
