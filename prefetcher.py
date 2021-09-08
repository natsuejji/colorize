import torch
class CUDAPrefetcher():
    """CUDA prefetcher.
    Ref:
    https://github.com/NVIDIA/apex/issues/304#
    It may consums more GPU memory.
    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda:0')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.loader = iter(self.ori_loader)
            self.batch = next(self.loader)
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()
