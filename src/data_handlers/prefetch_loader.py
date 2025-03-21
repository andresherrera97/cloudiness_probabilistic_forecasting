import torch
from functools import partial
from contextlib import suppress


class PrefetchLoader:
    def __init__(
        self,
        loader,
        device,
        img_dtype=torch.float32,
    ):
        self.loader = loader
        self.device = device
        self.img_dtype = img_dtype
        self.is_cuda = torch.cuda.is_available() and device.type == "cuda"

    def __iter__(self):
        first = True
        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress
        for next_input in self.loader:
            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_input = next_input.to(self.img_dtype)
            if not first:
                yield input
            else:
                first = False
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
            input = next_input
        yield input

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
