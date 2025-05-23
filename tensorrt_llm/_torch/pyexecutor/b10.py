import flashinfer
import torch
import math
import itertools

from .scheduler import ScheduledRequests

class B10Decoder:
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 1.0
    DEFAULT_TEMPERATURE = 1.0

    # Processes requests with top_p or temperature.
    # returns (request indices which were sampled,
    #          sampled token offset for each such request,
    #          sampled token for each such request)
    @staticmethod
    def custom_decode(scheduled_requests: ScheduledRequests, model_outputs: dict[str, torch.Tensor],
                      process_all_requests: bool = False):
        inputs = B10Decoder._get_custom_sampling_params(scheduled_requests, model_outputs['logits'].device, process_all_requests=process_all_requests)
        return B10Decoder._batch_decode(model_outputs, **inputs)

    @staticmethod
    def _get_custom_sampling_params(scheduled_requests: ScheduledRequests,
                                        device: torch.device,
                                        process_all_requests: bool = False) -> dict[str, torch.Tensor]:

        # requests that require custom sampling
        request_idx = []
        # logits idx of each request from request_idx
        logits_idx = []

        temperature_vals = []
        top_p_vals = []
        disable_mtp_mask = []
        greedy_mask = []
        
        cur_idx = 0
        next_idx = 0
        for i, request in enumerate(itertools.chain(scheduled_requests.context_requests,
                                scheduled_requests.generation_requests)):
            cur_idx = next_idx
            next_idx += request.num_draft_tokens + 1

            if request.is_context_init_state and not request.is_last_context_chunk:
                continue

            is_custom = process_all_requests

            is_greedy = False
            is_mtp_disabled = False

            sampling_config = request.sampling_config

            temperature = sampling_config.temperature[0] if (
                sampling_config.temperature is not None
                and len(sampling_config.temperature) > 0
            ) else None

            top_p = sampling_config.top_p[0] if (
                sampling_config.top_p is not None
                and len(sampling_config.top_p) > 0
            ) else None

            if request.guided_decoding_params is not None:
                is_greedy = True
                is_mtp_disabled = True
                is_custom = True

            if temperature is not None:
                assert len(sampling_config.temperature) == 1
                is_custom = True
                if temperature < 1e-6:
                    is_greedy = True
                    temperature = B10Decoder.DEFAULT_TEMPERATURE

            if (top_p is not None
                and top_p > 0
                and top_p < 1):
                assert len(sampling_config.top_p) == 1
                is_custom = True

            if is_custom:
                logits_idx.append(cur_idx)
                request_idx.append(i)
                disable_mtp_mask.append(is_mtp_disabled)
                greedy_mask.append(is_greedy)
                temperature_vals.append(temperature or B10Decoder.DEFAULT_TEMPERATURE)
                top_p_vals.append(top_p or B10Decoder.DEFAULT_TOP_P)

        request_idx = torch.tensor(request_idx, device=device, dtype=torch.int32)
        logits_idx = torch.tensor(logits_idx, device=device, dtype=torch.int32)
        temperature = torch.tensor(temperature_vals, device=device, dtype=torch.float32)
        top_p = torch.tensor(top_p_vals, device=device, dtype=torch.float32)
        disable_mtp_mask = torch.tensor(disable_mtp_mask, device=device, dtype=torch.bool)
        greedy_mask = torch.tensor(greedy_mask, device=device, dtype=torch.bool)

        return {
            'request_idx': request_idx,
            'logits_idx': logits_idx,
            'temperature': temperature,
            'top_p': top_p,
            'disable_mtp_mask': disable_mtp_mask,
            'greedy_mask': greedy_mask,
        }

    @staticmethod
    def _batch_decode(model_outputs,
                          *,
                          request_idx,
                          logits_idx,
                          temperature,
                          top_p,
                          disable_mtp_mask,
                          greedy_mask) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:

        if len(request_idx) == 0:
            return [], [], []

        logits = model_outputs['logits']
        lens = model_outputs['new_tokens_lens'] if 'new_tokens_lens' in model_outputs else (
            torch.ones((len(model_outputs['logits']),), dtype=torch.int32, device=logits.device)
        )

        sampled_token_offset = (lens[request_idx] - 1) * ~disable_mtp_mask

        # check for garbage output
        if 'new_tokens_lens' in model_outputs and 'new_tokens' in model_outputs:
            disable_mtp_mask |= (model_outputs['new_tokens'][request_idx,sampled_token_offset] == 0)
            sampled_token_offset *= ~disable_mtp_mask

        idxes = logits_idx + sampled_token_offset

        logits = logits[idxes,:]
        logits /= temperature.unsqueeze(1)

        sampled_tokens = torch.empty((len(request_idx),), device=logits.device, dtype=torch.int32)
        
        non_greedy_mask = ~greedy_mask
        logits_greedy = logits[greedy_mask,:]
        logits_not_greedy = logits[non_greedy_mask,:]

        if len(logits_not_greedy) > 0:
            sampled_tokens[non_greedy_mask] = flashinfer.top_k_top_p_sampling_from_logits(logits_not_greedy, B10Decoder.DEFAULT_TOP_K, top_p[non_greedy_mask])

        if len(logits_greedy) > 0:
            sampled_tokens[greedy_mask] = torch.argmax(logits_greedy, dim=1).to(torch.int32)

        return request_idx, sampled_token_offset, sampled_tokens
            