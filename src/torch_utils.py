import torch.nn as nn


class TransparentDataParallel(nn.DataParallel):

    @property
    def config(self):
        return self.module.config

    def get_output_embeddings(self, *args, **kwargs):
        return self.module.get_output_embeddings(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.module.prepare_inputs_for_generation(*args, **kwargs)
