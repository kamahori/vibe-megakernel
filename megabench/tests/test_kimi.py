"""KDA state layout, attention residuals, native MXFP4 and reset semantics."""

from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from ..cases import select_cases
from ..tasks import kimi
from ..tasks.quantization import unpack_mxfp4


def development(case=None):
    case = case or select_cases('all',['kimi-k3-step'])[0]
    return replace(case,params=case.params | {
        'layers':5,'hidden':64,'latent_hidden':32,'intermediate':64,
        'dense_intermediate':128,'q_heads':4,'head_dim':16,'q_lora_rank':32,
        'kv_lora_rank':32,'qk_nope_dim':8,'qk_rope_dim':8,'v_head_dim':16,
        'vocab':128,'experts':4,'topk':2,'context':4,'attn_res_block_size':2,
    },atol=0.003,rtol=0.003)


class KimiReferenceTests(unittest.TestCase):
    def test_kda_v_first_matches_closed_form_transition_over_rollout(self):
        torch.manual_seed(5)
        heads,dim = 3,16
        state = torch.randn(heads,dim,dim)
        expected = state.clone()
        A,dt = torch.randn(heads),torch.randn(heads,dim)
        for _ in range(8):
            q,k,v,g = (torch.randn(heads,dim) for _ in range(4))
            b = torch.randn(heads)
            normalized_q = q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt()/dim**0.5
            normalized_k = k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
            decay = torch.exp(-5*torch.sigmoid(torch.exp(A)[:,None]*(g+dt)))
            update = torch.eye(dim)[None]-b.sigmoid()[:,None,None]*normalized_k[:,:,None]*normalized_k[:,None,:]
            expected = torch.bmm(expected,torch.diag_embed(decay))@update + b.sigmoid()[:,None,None]*v[:,:,None]*normalized_k[:,None,:]
            output,state = kimi.kda_step(q,k,v,g,b,A,dt,state)
            torch.testing.assert_close(state,expected,rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(output,(expected@normalized_q[:,:,None]).squeeze(-1),rtol=2e-5,atol=2e-6)

    def test_mxfp4_tp_slices_keep_native_blocks(self):
        blocks,scales = kimi.mxfp4_matrix(13,'native',64,128,'cpu')
        decoded = unpack_mxfp4(blocks,scales)
        for columns in ((0,64),(64,128)):
            local,local_scales = kimi.mxfp4_matrix(13,'native',64,128,'cpu',col_range=columns)
            self.assertTrue(torch.equal(unpack_mxfp4(local,local_scales),decoded[:,columns[0]:columns[1]]))
        for rows in ((0,32),(32,64)):
            local,local_scales = kimi.mxfp4_matrix(13,'native',64,128,'cpu',row_range=rows)
            self.assertTrue(torch.equal(unpack_mxfp4(local,local_scales),decoded[rows[0]:rows[1]]))

    def test_reset_clears_both_state_types_and_preserves_inputs(self):
        case = replace(development(),gpus=1,tp=1)
        values = kimi.make_inputs(case,17,'cpu',rank=0)
        for reset in (False,True):
            values['reset'].fill_(reset)
            before = {name:value.clone() for name,value in values.items()}
            actual = kimi.reference(case,values,serial=True)
            repeated = kimi.reference(case,values,serial=True)
            for name in actual:
                self.assertTrue(torch.equal(actual[name],repeated[name]))
                if actual[name].is_floating_point():
                    self.assertTrue(torch.isfinite(actual[name]).all())
            self.assertTrue(all(torch.equal(value,before[name]) for name,value in values.items()))
            self.assertEqual(actual['cache_length'].item(),1 if reset else case.params['context']+1)
            changed = {name:value.clone() for name,value in values.items()}
            for name in changed:
                if name.endswith(('conv_state','recurrent_state','kv_cache','pe_cache')):
                    changed[name].add_(10)
            other = kimi.reference(case,changed,serial=True)
            if reset:
                self.assertTrue(all(torch.equal(actual[name],other[name]) for name in actual))
            else:
                self.assertFalse(torch.equal(actual['logits'],other['logits']))
            changed = values | {'token':(values['token']+1)%case.params['vocab']}
            self.assertFalse(torch.equal(actual['logits'],kimi.reference(case,changed,serial=True)['logits']))


if __name__ == '__main__':
    unittest.main()
