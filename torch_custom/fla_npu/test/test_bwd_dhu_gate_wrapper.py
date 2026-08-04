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
    def _call(self, *, g, gk, use_exp2):
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
                q, q, w, d_o, d_o, 1.0, 64, g=g, gK=gk, use_exp2=use_exp2
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


if __name__ == "__main__":
    unittest.main()
