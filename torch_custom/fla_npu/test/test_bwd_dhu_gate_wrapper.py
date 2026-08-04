import unittest
from unittest import mock

import torch

from fla_npu.ops.ascendc import _aclnn_ctypes as ascendc_ctypes


class _FakeCallContext:
    def tensor(self, tensor, name, **kwargs):
        del name, kwargs
        return tensor

    def int_array(self, values):
        return values


class BwdDhuGateWrapperTest(unittest.TestCase):
    def _call(self, *, g=None, gk=None, h0=None, dht=None, use_exp2=False,
              transpose_state_layout=False, cu_seqlens=None, chunk_indices=None):
        q = torch.zeros((1, 1, 64, 64), dtype=torch.float16)
        w = torch.zeros_like(q)
        d_o = torch.zeros((1, 1, 64, 64), dtype=torch.float16)
        captured = {}

        def fake_call(name, build_args, outputs, **kwargs):
            del kwargs
            captured["name"] = name
            captured["args"] = build_args(_FakeCallContext())
            return outputs

        with mock.patch.object(ascendc_ctypes, "_call_aclnn", fake_call):
            ascendc_ctypes.npu_chunk_gated_delta_rule_bwd_dhu(
                q, q, w, d_o, d_o, 1.0, 64, g=g, gK=gk, h0=h0, dht=dht,
                cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, use_exp2=use_exp2,
                transpose_state_layout=transpose_state_layout,
            )
        return captured

    def test_g_none_is_forwarded_as_optional_null(self):
        captured = self._call(g=None, gk=None, use_exp2=False)
        self.assertEqual(captured["name"], "aclnnChunkGatedDeltaRuleBwdDhu")
        self.assertIsNone(captured["args"][5])
        self.assertIsNone(captured["args"][6])

    def test_gk_fp32_is_forwarded(self):
        gk = torch.ones((1, 1, 64, 64), dtype=torch.float32)
        captured = self._call(g=None, gk=gk, use_exp2=False)
        self.assertEqual(captured["args"][6].dtype, torch.float32)
        torch.testing.assert_close(captured["args"][6], torch.ones_like(gk, dtype=torch.float32))

    def test_use_exp2_is_compatibility_only(self):
        g = torch.ones((1, 1, 64), dtype=torch.float16)
        gk = torch.ones((1, 1, 64, 64), dtype=torch.float32)
        captured_false = self._call(g=g, gk=gk, use_exp2=False)
        captured_true = self._call(g=g, gk=gk, use_exp2=True)

        for captured in (captured_false, captured_true):
            self.assertEqual(captured["args"][5].dtype, torch.float16)
            self.assertEqual(captured["args"][6].dtype, torch.float32)
            torch.testing.assert_close(captured["args"][5], g)
            torch.testing.assert_close(captured["args"][6], gk)

        torch.testing.assert_close(captured_false["args"][5], captured_true["args"][5])
        torch.testing.assert_close(captured_false["args"][6], captured_true["args"][6])

    def test_state_inputs_are_normalized_to_fp32_and_dh0_is_k_first(self):
        h0 = torch.ones((1, 1, 64, 64), dtype=torch.float16)
        dht = torch.full_like(h0, 2)
        captured = self._call(h0=h0, dht=dht)

        self.assertEqual(captured["args"][7].dtype, torch.float32)
        self.assertEqual(captured["args"][8].dtype, torch.float32)
        dh0 = captured["args"][14]
        self.assertEqual(dh0.shape, (1, 1, 64, 64))
        self.assertEqual(dh0.dtype, torch.float32)

    def test_varlen_state_num_comes_from_cu_seqlens(self):
        h0 = torch.ones((2, 1, 64, 64), dtype=torch.bfloat16)
        captured = self._call(
            h0=h0,
            cu_seqlens=[0, 32, 64],
            chunk_indices=[0, 0, 1, 0],
        )
        self.assertEqual(captured["args"][14].shape, (2, 1, 64, 64))

    def test_transpose_state_layout_keeps_historical_k_first_behavior(self):
        h0 = torch.ones((1, 1, 64, 64), dtype=torch.float32)
        captured_false = self._call(h0=h0, transpose_state_layout=False)
        captured_true = self._call(h0=h0, transpose_state_layout=True)

        self.assertEqual(captured_false["args"][7].shape, captured_true["args"][7].shape)
        self.assertEqual(captured_false["args"][14].shape, captured_true["args"][14].shape)
        torch.testing.assert_close(captured_false["args"][7], captured_true["args"][7])


if __name__ == "__main__":
    unittest.main()
