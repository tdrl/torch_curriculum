import torch
from torch.utils.data import DataLoader, Dataset
from torch_playground.d04_name_seq_learner import PaddingCollate, NameSeqLearnerConfig, NameSeqTransformer
from test_util import with_eligible_devices

class RaggedListDataset(Dataset):
    def __init__(self, data: list[list[int]]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestPaddingCollate:

    def test_padding_collate_pads_shorter(self):
        collate = PaddingCollate(padding_value=0)
        batch = [[1, 2, 3], [4, 5], [6]]
        src, tgt, goal = collate(batch)

        for out in (src, tgt, goal):
            assert isinstance(out, torch.Tensor)
            assert out.dtype == torch.long

        expected_src = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]], dtype=torch.long)
        assert torch.equal(src, expected_src)
        expected_tgt = torch.tensor([[0, 1, 2, 3],
                                     [0, 4, 5, 0],
                                     [0, 6, 0, 0]], dtype=torch.long)
        assert torch.equal(tgt, expected_tgt), f'Got {tgt}, expected {expected_tgt}'
        assert torch.equal(goal, expected_tgt), f'Got {goal}, expected {expected_tgt}'

    def test_padding_collate_preserves_order_and_padding_value(self):
        collate = PaddingCollate(padding_value=-3)
        batch = [[7, 8, 4, 4, 4], [9]]
        src, tgt, goal = collate(batch)

        expected_src = torch.tensor([[7, 8, 4, 4, 4],
                                 [9, -3, -3, -3, -3]], dtype=torch.long)
        assert torch.equal(src, expected_src)
        expected_tgt = torch.tensor([[ -3, 7, 8, 4, 4, 4],
                                     [ -3, 9, -3, -3, -3, -3]], dtype=torch.long)
        assert torch.equal(tgt, expected_tgt), f'Got {tgt}, expected {expected_tgt}'
        assert torch.equal(goal, expected_tgt), f'Got {goal}, expected {expected_tgt}'

    def test_padding_collate_no_padding_needed(self):
        collate = PaddingCollate(padding_value=0)
        batch = [[1, 2], [3, 4]]
        src, tgt, goal = collate(batch)

        expected_src = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        assert torch.equal(src, expected_src)
        expected_tgt = torch.tensor([[0, 1, 2],
                                     [0, 3, 4]], dtype=torch.long)
        assert torch.equal(tgt, expected_tgt), f'Got {tgt}, expected {expected_tgt}'
        assert torch.equal(goal, expected_tgt), f'Got {goal}, expected {expected_tgt}'

    @with_eligible_devices()
    def test_forward_with_device(self, device):
        config = NameSeqLearnerConfig(d_embedding=12,  # Has to be multiple of n_heads.
                                      d_feedforward=10,
                                      n_heads=4,
                                      n_encoder_layers=2,
                                      n_decoder_layers=2,
                                      vocab_size=12,
                                      in_seq_length=5,
                                      out_seq_length=5)
        model = NameSeqTransformer.from_config(config).to(device)
        # Ragged and not a multiple of batch_size, so will trigger padding in both dimensions.
        raw_dataset = RaggedListDataset([[1, 2, 3],
                                         [4, 5],
                                         [6],
                                         [7, 8, 9],
                                         [10, 11]])
        data_loader = DataLoader(dataset=raw_dataset, batch_size=2, shuffle=True, collate_fn=PaddingCollate(padding_value=0))
        for batch in data_loader:
            batch = batch[0].to(device)
            src = batch.clone()
            src[:, 2:] = 0
            tgt = batch
            output = model(src, tgt)
            # Note: The requested device (input to this method) is not concrete - it doesn't have a specific
            # device index. We can't compare it directly to the output.device, but we can check that they are of
            # the same type (e.g., both 'cpu' or both 'mps') and that the output shape is as expected.
            assert output.device.type == device.type
            assert output.shape == (batch.size(0), batch.size(1), config.d_embedding)
