import torch
from torch_playground.d04_name_seq_learner import PaddingCollate

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
