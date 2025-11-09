import torch
from torch_playground.d04_name_seq_learner import PaddingCollate

class TestPaddingCollate:

    def test_padding_collate_pads_shorter(self):
        collate = PaddingCollate(padding_value=0)
        batch = [[1, 2, 3], [4, 5], [6]]
        out = collate(batch)

        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.long

        expected = torch.tensor([[1, 2, 3], [4, 5, 0], [6, 0, 0]], dtype=torch.long)
        assert torch.equal(out, expected)

    def test_padding_collate_preserves_order_and_padding_value(self):
        collate = PaddingCollate(padding_value=9)
        batch = [[7, 8, 4, 4, 4], [9]]
        out = collate(batch)

        expected = torch.tensor([[7, 8, 4, 4, 4],
                                 [9, 9, 9, 9, 9]], dtype=torch.long)
        assert torch.equal(out, expected)

    def test_padding_collate_no_padding_needed(self):
        collate = PaddingCollate(padding_value=0)
        batch = [[1, 2], [3, 4]]
        out = collate(batch)

        expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        assert torch.equal(out, expected)
