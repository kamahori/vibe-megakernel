"""Real-weight DeepSeek-V3.2 plumbing on a fake dev-geometry FP8 checkpoint.

The fake checkpoint is the synthetic dev case written under checkpoint names
(FP32 norms, BF16 router and head, FP8 matrices with block scales), with a
word-level tokenizer and a small transcript corpus. It checks that the name
mapping and TP slicing reproduce the synthetic shards, that prefill plus the
decode reference agree with Hugging Face's DeepseekV32ForCausalLM on the
same weights, and that the TileRT-style gate grades as documented.
"""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from megabench.cases import DEEPSEEK_CHECKPOINT, select_cases
from megabench.harness.correctness import _compare, tilert_metrics
from megabench.tasks import deepseek_checkpoint as checkpoint
from megabench.tasks import frontier
from megabench.tasks.quantization import unpack_fp8_weight

SEED = 7
FLOAT32 = ('model.norm.weight', 'layernorm.weight', 'k_norm.', 'e_score_correction_bias')


def development(tp: int = 1, index_topk: int = 5):
    (case,) = select_cases('all', ['deepseek-v32-step'])
    return replace(case, checkpoint='', gate='band', gpus=tp, tp=tp, params=case.params | {
        'hidden': 128, 'q_heads': 8, 'q_lora_rank': 128, 'kv_lora_rank': 128,
        'qk_nope_dim': 32, 'qk_rope_dim': 32, 'v_head_dim': 32,
        'intermediate': 256, 'dense_intermediate': 256, 'experts': 8, 'topk': 2,
        'index_heads': 4, 'index_dim': 128, 'index_topk': index_topk, 'vocab': 512,
        'layers': 3, 'first_dense': 1, 'context': 12, 'router_groups': 2,
        'router_top_groups': 1, 'prompts': 2})


def real(case):
    return replace(case, checkpoint='fake/DeepSeek-V3.2@' + '0' * 40, gate='tilert')


def write_checkpoint(directory: Path, case) -> None:
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    top = frontier.global_inputs(case, SEED, 'cpu', rank=0)
    tensors = {'model.embed_tokens.weight': top['embed'], 'lm_head.weight': top['lm_head'],
               'model.norm.weight': top['fnorm']}
    for layer in range(case.params['layers']):
        tensors |= checkpoint.hf_layer(top | frontier.layer_inputs(case, SEED, layer, 'cpu', rank=0), layer, case)
    tensors = {name: value.float() if any(key in name for key in FLOAT32) else value.contiguous()
               for name, value in tensors.items()}
    # Two shards, as a real index splits tensors across files.
    names = sorted(tensors)
    shards = {'model-00001-of-00002.safetensors': names[::2], 'model-00002-of-00002.safetensors': names[1::2]}
    for file, keys in shards.items():
        save_file({name: tensors[name] for name in keys}, directory / file)
    (directory / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {name: file for file, keys in shards.items() for name in keys}}))
    words = ['<bos>'] + [f'w{index}' for index in range(1, case.params['vocab'])]
    tokenizer = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token='w1'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(directory / 'tokenizer.json'))
    generator = torch.Generator().manual_seed(SEED)
    (directory / 'encoding').mkdir()
    for file in checkpoint.CORPUS_FILES:
        ids = torch.randint(1, case.params['vocab'], (20,), generator=generator).tolist()
        (directory / file).write_text(' '.join(words[index] for index in ids))


def hf_model(case, tensors: dict[str, torch.Tensor]):
    """DeepseekV32ForCausalLM in FP32 with the dequantized checkpoint."""
    from transformers import DeepseekV32Config, DeepseekV32ForCausalLM

    p = case.params
    config = DeepseekV32Config(
        vocab_size=p['vocab'], hidden_size=p['hidden'], intermediate_size=p['dense_intermediate'],
        moe_intermediate_size=p['intermediate'], num_hidden_layers=p['layers'],
        num_attention_heads=p['q_heads'], num_key_value_heads=p['q_heads'],
        n_shared_experts=p['shared_experts'], n_routed_experts=p['experts'], num_experts_per_tok=p['topk'],
        n_group=p['router_groups'], topk_group=p['router_top_groups'], kv_lora_rank=p['kv_lora_rank'],
        q_lora_rank=p['q_lora_rank'], qk_rope_head_dim=p['qk_rope_dim'], v_head_dim=p['v_head_dim'],
        qk_nope_head_dim=p['qk_nope_dim'], index_topk=p['index_topk'], index_head_dim=p['index_dim'],
        index_n_heads=p['index_heads'], first_k_dense_replace=p['first_dense'],
        rope_parameters={'rope_type': 'yarn', 'rope_theta': 10000.0, 'factor': 40.0,
                         'original_max_position_embeddings': 4096, 'beta_fast': 32.0, 'beta_slow': 1.0,
                         'mscale': 1.0, 'mscale_all_dim': 1.0},
        attn_implementation='eager', torch_dtype=torch.float32)
    model = DeepseekV32ForCausalLM(config).float().eval()
    state = {}
    for name, value in tensors.items():
        if name.endswith('.weight_scale_inv'):
            continue
        if value.dtype == torch.float8_e4m3fn:
            value = unpack_fp8_weight(value, tensors[name.replace('.weight', '.weight_scale_inv')])
        state[name] = value.float()
    for layer in range(p['first_dense'], p['layers']):
        scope = f'model.layers.{layer}.mlp.experts'
        expert = lambda index, name: state.pop(f'{scope}.{index}.{name}_proj.weight')
        state[scope + '.gate_up_proj'] = torch.stack([torch.cat((expert(index, 'gate'), expert(index, 'up')))
                                                      for index in range(p['experts'])])
        state[scope + '.down_proj'] = torch.stack([expert(index, 'down') for index in range(p['experts'])])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or [name for name in missing if 'rotary' not in name]:
        raise AssertionError(f'missing {missing}, unexpected {unexpected}')
    return model


def _tp_worker(rank: int, root: str) -> None:
    """One rank of a TP2 prefill and graded decode on the fake checkpoint."""
    import torch.distributed as dist

    from megabench.tasks.workloads import oracle

    torch.set_num_threads(1)
    os.environ |= {checkpoint.CHECKPOINT_ENV: f'{root}/checkpoint', checkpoint.CACHE_ENV: f'{root}/tp2-cache'}
    dist.init_process_group('gloo', init_method=f'file://{root}/store', rank=rank, world_size=2)
    case = real(development(tp=2))
    prompts = list(range(case.params['prompts']))
    records = checkpoint.build_prompt_caches([(case, prompt) for prompt in prompts], 'cpu', rank=rank,
                                             keep_caches=rank == 0)
    if rank == 0:
        for prompt, record in zip(prompts, records):
            checkpoint.save_prompt_cache(case, prompt, record)
    dist.barrier()
    values = frontier.make_inputs(case, 1, 'cpu')
    expected, exact = frontier.reference(case, values), oracle(case, values)
    details = _compare(expected, {name: value.clone() for name, value in expected.items()}, case, 'cpu', exact)
    torch.save({'records': records, 'details': details, 'logits': expected['logits']}, f'{root}/tp2-rank{rank}.pt')
    dist.destroy_process_group()


class DeepSeekCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        (root / 'checkpoint').mkdir()
        write_checkpoint(root / 'checkpoint', development())
        cls.environment = mock.patch.dict(os.environ, {checkpoint.CHECKPOINT_ENV: str(root / 'checkpoint'),
                                                       checkpoint.CACHE_ENV: str(root / 'cache')})
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()
        checkpoint._WEIGHTS.clear()
        checkpoint._PREFIX.clear()
        cls.directory.cleanup()

    def test_checkpoint_shards_match_synthetic_tp_shards(self):
        synthetic = development(tp=2)
        for rank in range(2):
            loaded = checkpoint.global_weights(real(synthetic), 'cpu', rank=rank)
            for layer in range(synthetic.params['layers']):
                loaded |= checkpoint.layer_weights(real(synthetic), layer, 'cpu', rank=rank)
            want = frontier.global_inputs(synthetic, SEED, 'cpu', rank=rank)
            for layer in range(synthetic.params['layers']):
                want |= frontier.layer_inputs(synthetic, SEED, layer, 'cpu', rank=rank)
            for name, value in loaded.items():
                with self.subTest(rank=rank, name=name):
                    self.assertEqual(value.shape, want[name].shape)
                    self.assertTrue(torch.equal(value.float(), want[name].float()))
            self.assertEqual(set(loaded), {name for name in want if not name.endswith('_cache') and name != 'token'})

    def decode_against_hugging_face(self, index_topk: int):
        case = real(development(index_topk=index_topk))
        with mock.patch.dict(os.environ, {checkpoint.CACHE_ENV: str(Path(self.directory.name) / f'cache{index_topk}')}):
            jobs = [(case, prompt) for prompt in range(case.params['prompts'])]
            for (_, prompt), record in zip(jobs, checkpoint.build_prompt_caches(jobs, 'cpu', rank=0)):
                checkpoint.save_prompt_cache(case, prompt, record)
            record = checkpoint.load_prefix(case, 1)
            values = frontier.make_inputs(case, 1, 'cpu', rank=0)
        tokens, _ = checkpoint.corpus_tokens(case)
        window = checkpoint.prompt_window(case, tokens, 1)
        self.assertEqual(record['tokens'], window)
        self.assertEqual((window[0], len(window)), (checkpoint.BOS, case.params['context'] + 1))
        self.assertEqual(int(values['token']), window[-1])
        outputs = frontier.reference(case, values, serial=True)

        ckpt = checkpoint.open_checkpoint(case)
        model = hf_model(case, {name: ckpt.get(name, 'cpu') for name in ckpt.weight_map})
        latents, selections = {}, {}
        for layer, module in enumerate(model.model.layers):
            module.self_attn.kv_a_layernorm.register_forward_hook(
                lambda _, __, output, layer=layer: latents.__setitem__(layer, output[0]))
            module.self_attn.indexer.register_forward_hook(
                lambda _, __, output, layer=layer: selections.__setitem__(layer, output[0, -1]))
        with torch.no_grad():
            logits = model(torch.tensor([window])).logits[0, -1].double()
        cosine = lambda a, b: float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0))
        for layer in range(case.params['layers']):
            # The prefill runs in BF16 with dequantized weights; HF in FP32.
            self.assertGreater(cosine(record['kv_cache'][layer], latents[layer][:-1]), 0.9995)
        return outputs, logits, selections, cosine

    def test_dense_selection_decode_matches_hugging_face(self):
        # With top-k covering the window, no near-tie selection can flip, so
        # prefill plus decode must track the FP32 model end to end.
        outputs, logits, _, cosine = self.decode_against_hugging_face(index_topk=64)
        self.assertGreater(cosine(outputs['logits'], logits), 0.995)
        self.assertEqual(int(outputs['next_token']), int(logits.argmax()))

    def test_sparse_selection_matches_hugging_face_indexer(self):
        # Synthetic weights make later layers chaotic, so compare the first
        # layer's selection: rope layout, Hadamard and head weights.
        outputs, _, selections, _ = self.decode_against_hugging_face(index_topk=5)
        self.assertEqual(outputs['sparse_indices'][0].sort().values.tolist(), selections[0].sort().values.tolist())

    def test_tp2_prefill_and_sharded_gate_match_one_rank(self):
        import torch.multiprocessing as mp

        root = self.directory.name
        mp.spawn(_tp_worker, args=(root,), nprocs=2)
        ranks = [torch.load(f'{root}/tp2-rank{rank}.pt', weights_only=False) for rank in range(2)]
        case = real(development())
        with mock.patch.dict(os.environ, {checkpoint.CACHE_ENV: f'{root}/tp1-cache'}):
            serial = checkpoint.build_prompt_caches([(case, 0), (case, 1)], 'cpu', rank=0)
            for prompt, record in enumerate(serial):
                checkpoint.save_prompt_cache(case, prompt, record)
            values = frontier.make_inputs(case, 1, 'cpu', rank=0)
        logits = frontier.reference(case, values, serial=True)['logits']
        self.assertNotIn('kv_cache', ranks[1]['records'][0])
        for sharded, single in zip(ranks[0]['records'], serial):
            # TP partial sums round differently in BF16.
            self.assertAlmostEqual(sharded['mean_nll'], single['mean_nll'], delta=0.05)
            for name in ('kv_cache', 'pe_cache', 'index_cache'):
                cosine = torch.nn.functional.cosine_similarity(
                    sharded[name].double().flatten(), single[name].double().flatten(), dim=0)
                self.assertGreater(float(cosine), 0.9999, name)
        # Both ranks see the same reduced verdicts over vocabulary shards.
        self.assertEqual(ranks[0]['details']['logits'], ranks[1]['details']['logits'])
        self.assertEqual(ranks[0]['details']['next_token'], ranks[1]['details']['next_token'])
        self.assertGreater(ranks[0]['details']['logits']['reference_cosine'], 0.99)
        gathered = torch.cat([rank['logits'] for rank in ranks]).double()
        self.assertGreater(float(torch.nn.functional.cosine_similarity(gathered, logits.double(), dim=0)), 0.99)

    def test_probe_swaps_prompts_with_matching_digest(self):
        from megabench.probes import tilert_dsv32_probe as probe

        case = real(development())
        with mock.patch.dict(os.environ, {checkpoint.CACHE_ENV: str(Path(self.directory.name) / 'probe-cache')}):
            jobs = [(case, prompt) for prompt in range(case.params['prompts'])]
            for (_, prompt), record in zip(jobs, checkpoint.build_prompt_caches(jobs, 'cpu', rank=0)):
                checkpoint.save_prompt_cache(case, prompt, record)
            first, second = (frontier.make_inputs(case, seed, 'cpu', rank=0) for seed in (0, 1))
            # What build_tilert_inputs keeps from the first seed: its token and digested layers.
            kept = {name: value for name, value in first.items()
                    if name == 'token' or name.startswith(('l0_', 'l2_'))}
            caches, token, digest = probe._prompt_inputs(case, 1, kept)
        self.assertEqual(digest, probe._digest(second, case.params['layers']))
        self.assertNotEqual(digest, probe._digest(first, case.params['layers']))
        self.assertEqual(token, int(second['token']))
        self.assertTrue(torch.equal(caches[1][1], second['l1_kv_cache']))

    def test_tilert_gate(self):
        case = real(development())
        generator = torch.Generator().manual_seed(1)
        exact = {'logits': torch.randn(512, generator=generator, dtype=torch.float32),
                 'kv_write': torch.randn(3, 128, generator=generator).to(torch.bfloat16),
                 'sparse_indices': torch.arange(15, dtype=torch.int32).view(3, 5),
                 'expert_ids': torch.tensor([[-1, -1], [1, 2], [3, 4]])}
        exact['next_token'] = exact['logits'].argmax()
        expected = {name: value.clone() for name, value in exact.items()}
        expected['logits'] = exact['logits'] + 0.01 * torch.randn(512, generator=generator)
        expected['next_token'] = expected['logits'].argmax()

        details = _compare(expected, {name: value.clone() for name, value in expected.items()}, case, 'cpu', exact)
        self.assertGreater(details['logits']['cosine'], 0.999)
        self.assertFalse(details['sparse_indices']['graded'])

        # Different sparse selections and experts are reported, not graded.
        other = {name: value.clone() for name, value in expected.items()}
        other['sparse_indices'] = other['sparse_indices'].flip(0)
        other['expert_ids'][2] = torch.tensor([5, 6])
        details = _compare(expected, other, case, 'cpu', exact)
        self.assertEqual(details['expert_ids']['min_recall'], 0.0)
        self.assertLess(details['sparse_indices']['mean_recall'], 1.0)

        # A near-tie token passes; a far token fails.
        ordered = exact['logits'].argsort(descending=True)
        near = {name: value.clone() for name, value in expected.items()}
        near['next_token'] = ordered[1].clone()
        gap = float(exact['logits'][ordered[0]] - exact['logits'][ordered[1]])
        margin = tilert_metrics(expected, near, case, 'cpu', exact)['next_token']['near_tie_margin']
        if gap <= margin:
            _compare(expected, near, case, 'cpu', exact)
        near['next_token'] = ordered[-1].clone()
        with self.assertRaisesRegex(AssertionError, 'next_token'):
            _compare(expected, near, case, 'cpu', exact)

        # Wrong floats fail the cosine/relative-L2 thresholds.
        wrong = {name: value.clone() for name, value in expected.items()}
        wrong['kv_write'] = torch.randn(3, 128, generator=generator).to(torch.bfloat16)
        with self.assertRaisesRegex(AssertionError, 'kv_write'):
            _compare(expected, wrong, case, 'cpu', exact)
        wrong = {name: value.clone() for name, value in expected.items()}
        wrong['logits'] = expected['logits'] * 1.5
        with self.assertRaisesRegex(AssertionError, 'logits'):
            _compare(expected, wrong, case, 'cpu', exact)
        # Shapes are still checked.
        wrong['logits'] = expected['logits'][:-1]
        with self.assertRaisesRegex(AssertionError, 'logits: expected'):
            _compare(expected, wrong, case, 'cpu', exact)

    def test_tilert_gate_reference_or_oracle(self):
        from megabench.harness.correctness import TILERT_GRADED_LAYERS as graded
        case = real(development())
        generator = torch.Generator().manual_seed(2)
        layers = graded + 2
        exact = {'logits': torch.randn(512, generator=generator, dtype=torch.float32),
                 'kv_write': torch.randn(layers, 128, generator=generator).to(torch.bfloat16)}
        exact['next_token'] = exact['logits'].argmax()
        # The reference left the oracle in its late layers and, slightly, in its logits.
        expected = {name: value.clone() for name, value in exact.items()}
        expected['kv_write'][graded:] = torch.randn(2, 128, generator=generator).to(torch.bfloat16)
        expected['logits'] = exact['logits'] + 0.3 * torch.randn(512, generator=generator)
        expected['next_token'] = expected['logits'].argmax()

        # Matching either the reference or the oracle passes.
        details = _compare(expected, {name: value.clone() for name, value in expected.items()}, case, 'cpu', exact)
        self.assertLess(details['logits']['cosine'], 0.99)
        self.assertFalse(details['kv_write']['graded'])
        self.assertNotIn('pass', details['kv_write'])
        self.assertGreater(details[f'kv_write[:{graded}]']['cosine'], 0.999)
        oracle_like = {name: value.clone() for name, value in exact.items()}
        details = _compare(expected, oracle_like, case, 'cpu', exact)
        self.assertLess(details['logits']['cosine_to_reference'], 0.99)

        # Late-layer cache writes are reported; early ones are graded.
        late = {name: value.clone() for name, value in expected.items()}
        late['kv_write'][-1] = torch.randn(128, generator=generator).to(torch.bfloat16)
        _compare(expected, late, case, 'cpu', exact)
        early = {name: value.clone() for name, value in expected.items()}
        early['kv_write'][0] = torch.randn(128, generator=generator).to(torch.bfloat16)
        with self.assertRaisesRegex(AssertionError, r'kv_write\[:'):
            _compare(expected, early, case, 'cpu', exact)

    def test_docker_mounts_checkpoint_and_caches(self):
        from megabench.harness.runner import _checkpoint_mounts
        case = real(development())
        root = Path(self.directory.name)
        cache = checkpoint.cache_path(case, 0).parent
        cache.mkdir(parents=True, exist_ok=True)
        mounts = _checkpoint_mounts(case)
        self.assertIn(f'{(root / "checkpoint").resolve()}:/checkpoint:ro', mounts)
        self.assertIn(f'{checkpoint.CHECKPOINT_ENV}=/checkpoint', mounts)
        self.assertIn(f'{cache.resolve()}:/megabench-cache/deepseek-v32/{cache.name}:ro', mounts)
        self.assertIn(f'{checkpoint.CACHE_ENV}=/megabench-cache', mounts)
        # A Hugging Face snapshot mounts its repository so blob symlinks resolve.
        snapshot = root / 'hub' / 'models--fake' / 'snapshots' / ('0' * 40)
        snapshot.mkdir(parents=True)
        (snapshot / 'model.safetensors.index.json').write_text('{}')
        with mock.patch.dict(os.environ, {checkpoint.CHECKPOINT_ENV: str(snapshot)}):
            mounts = _checkpoint_mounts(case)
        self.assertIn(f'{(root / "hub" / "models--fake").resolve()}:/checkpoint:ro', mounts)
        self.assertIn(f'{checkpoint.CHECKPOINT_ENV}=/checkpoint/snapshots/{"0" * 40}', mounts)
        self.assertEqual(_checkpoint_mounts(development()), [])

    def test_contract_records_checkpoint_and_gate(self):
        (case,) = select_cases('all', ['deepseek-v32-step-ctx4k'])
        self.assertEqual(case.to_dict()['checkpoint'], DEEPSEEK_CHECKPOINT)
        self.assertEqual(case.to_dict()['gate'], 'tilert')
        (glm,) = select_cases('all', ['glm52-step'])
        self.assertNotIn('checkpoint', glm.to_dict())
        self.assertNotIn('gate', glm.to_dict())

    def test_missing_cache_names_the_prepare_command(self):
        case = replace(real(development()), params=development().params | {'context': 13})
        with self.assertRaisesRegex(FileNotFoundError, 'prepare_deepseek_v32 --case deepseek-v32-step'):
            checkpoint.load_prefix(case, 0)


if __name__ == '__main__':
    unittest.main()
