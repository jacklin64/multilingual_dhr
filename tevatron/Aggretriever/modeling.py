import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, PreTrainedModel, AutoModelForMaskedLM
from transformers.modeling_outputs import ModelOutput


from typing import Optional, Dict

from ..arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
import logging

from .utils import aggregate

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True, 
            name='pooler'
    ):
        super(LinearPooler, self).__init__()
        self.name = name
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q)
        elif p is not None:
            return self.linear_p(p)
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, '{}.pt'.format(self.name))
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, '{}.pt'.format(self.name)), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training {} from scratch".format(self.name))
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, '{}.pt'.format(self.name)))
        with open(os.path.join(save_path, '{}_config.json').format(self.name), 'w') as f:
            json.dump(self._config, f)


class DenseModel(nn.Module):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            teacher_model: PreTrainedModel = None,
            pooler: nn.Module = None,
            term_weight_trans: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.teacher_model = teacher_model
        self.pooler = pooler
        self.term_weight_trans = term_weight_trans

        self.softmax = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Todo: 
        self.temperature = 1

        self.effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
            if self.train_args.negatives_x_device \
            else self.train_args.per_device_train_batch_size 



    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
            teacher_scores: Tensor = None,
            # agg_dim: int = 640,
            # semi_aggregate: bool = False,
            # skip_mlm: bool = False

    ):

        
        q_lexical_reps, q_semantic_reps = self.encode_query(query, self.model_args.skip_mlm)
        p_lexical_reps, p_semantic_reps = self.encode_passage(passage, self.model_args.skip_mlm)
        
        # Encode
        if q_lexical_reps is None or p_lexical_reps is None:
            q_reps, p_reps = None, None
            if query is not None:
                q_lexical_reps = aggregate(q_lexical_reps, self.model_args.agg_dim, full=not self.model_args.semi_aggregate)
                if q_semantic_reps is not None:
                    q_reps = self.merge_reps(q_lexical_reps, q_semantic_reps)
                else:
                    q_reps = q_lexical_reps
                
            if passage is not None:
                p_lexical_reps = aggregate(p_lexical_reps, self.model_args.agg_dim, full=not self.model_args.semi_aggregate)
                if p_semantic_reps is not None:
                    p_reps = self.merge_reps(p_lexical_reps, p_semantic_reps)
                else:
                    p_reps = p_lexical_reps
                

            return DenseOutput(
                q_reps = q_reps,
                p_reps = p_reps,
            )

        if self.training:
            if self.train_args.negatives_x_device:
                q_lexical_reps = self.dist_gather_tensor(q_lexical_reps)
                p_lexical_reps = self.dist_gather_tensor(p_lexical_reps)
                q_semantic_reps = self.dist_gather_tensor(q_semantic_reps)
                p_semantic_reps = self.dist_gather_tensor(p_semantic_reps)
                if teacher_scores is not None:
                    teacher_scores = self.dist_gather_tensor(teacher_scores)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            # lexical matching
            q_tok_reps = aggregate(q_lexical_reps, self.model_args.agg_dim, full=not self.model_args.semi_aggregate)
            p_tok_reps = aggregate(p_lexical_reps, self.model_args.agg_dim, full=not self.model_args.semi_aggregate)
            
            lexical_scores = self.listwise_scores(q_tok_reps, p_tok_reps, effective_bsz)

            # semantic matching
            if q_semantic_reps is not None:
                semantic_scores = self.listwise_scores(q_semantic_reps, p_semantic_reps, effective_bsz)
            else:
                semantic_scores = 0
            # fusion
            scores = lexical_scores + semantic_scores
            
            loss = 0

            # tct kd                 
            if self.model_args.tct:
                # KL
                self.teacher_model.eval()
                with torch.no_grad(): 
                    colbert_output = self.teacher_model(query=query, passage=passage, is_teacher=True)
                    tct_teacher_scores = colbert_output.scores
                loss += self.kl_loss(nn.functional.log_softmax(scores , dim=-1), self.softmax(tct_teacher_scores * self.temperature))
                # regularize semantic and lexical components
                loss += 0.5 * self.kl_loss(nn.functional.log_softmax(semantic_scores , dim=-1), self.softmax(tct_teacher_scores * self.temperature * 3 / 4))
                loss += 0.5 * self.kl_loss(nn.functional.log_softmax(lexical_scores , dim=-1), self.softmax(tct_teacher_scores * self.temperature * 1 / 4))
            else:
                if self.model_args.kd:
                    hard_label_scores = torch.nn.functional.pad(teacher_scores, (0 ,scores.shape[-1]), "constant", -20)
                    hard_label_scores = hard_label_scores.view(-1)[:-scores.shape[-1]].view(scores.shape[0],-1)
                    hard_label_scores = self.softmax(hard_label_scores)
                else: #hard label
                    hard_label_scores = torch.arange(
                        lexical_scores.size(0),
                        device=lexical_scores.device,
                        dtype=torch.long
                    )
                    hard_label_scores = hard_label_scores * self.data_args.train_n_passages
                    hard_label_scores = torch.nn.functional.one_hot(hard_label_scores, num_classes=lexical_scores.size(1)).float()

                if q_semantic_reps is not None:
                    loss += self.kl_loss(nn.functional.log_softmax(scores, dim=-1), hard_label_scores) + \
                            0.5 * self.kl_loss(nn.functional.log_softmax(lexical_scores, dim=-1), hard_label_scores) + \
                            0.5 * self.kl_loss(nn.functional.log_softmax(semantic_scores, dim=-1), hard_label_scores)
                else:
                    loss += self.kl_loss(nn.functional.log_softmax(scores, dim=-1), hard_label_scores)

            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return DenseOutput(
                loss=loss,
                scores=scores,
            )

        else: # re-rank
            loss = None

            # lexical matching
            if self.model_args.agg_dim is not None:
                q_tok_reps = aggregate(q_lexical_reps, self.model_args.agg_dim, full=not self.model_args.semi_aggregate)
                p_tok_reps = aggregate(p_lexical_reps, self.model_args.agg_dim, full=not self.model_args.semi_aggregate)
                lexical_scores = (q_tok_reps * p_tok_reps).sum(1)
            else:
                lexical_scores = (q_tok_reps * p_tok_reps).sum(1)

            # semantic matching
            if q_semantic_reps is not None:
                semantic_scores = (q_semantic_reps * p_semantic_reps).sum(1)
            else:
                semantic_scores = 0

            # score fusion
            scores = scores = lexical_scores + semantic_scores


            return DenseOutput(
                loss=loss,
                scores=scores,
            )


    def pairwise_scores(self, q_reps, p_reps, effective_bsz):
        q_reps = q_reps.view(effective_bsz, 1, -1)
        p_reps = p_reps.view(effective_bsz, self.data_args.train_n_passages, -1)
        scores = torch.matmul(q_reps, p_reps.transpose(2, 1)).squeeze()
        return scores


    def listwise_scores(self, q_reps, p_reps, effective_bsz):
        scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
        scores = scores.view(effective_bsz, self.data_args.train_n_passages, -1)

        scores = scores.view(effective_bsz, -1)
        return scores

    
    def encode_passage(self, psg, skip_mlm):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=True)
        p_seq_hidden = psg_out.hidden_states[-1]
        p_cls_hidden = p_seq_hidden[:,0] # get [CLS] embeddings
        p_term_weights = self.term_weight_trans(p_seq_hidden[:,1:]) # batch, seq, 1


        if not skip_mlm:
            p_logits = psg_out.logits[:,1:] # batch, seq, vocab
            p_logits = self.softmax(p_logits)
            attention_mask = psg['attention_mask'][:,1:].unsqueeze(-1)
            p_lexical_reps = torch.max((p_logits * p_term_weights) * attention_mask, dim=-2).values
        else:
            ## w/o MLM
            ## p_term_weights = torch.relu(p_term_weights)
            p_lexical_reps = torch.zeros(p_seq_hidden.shape[0], self.lm_p.embeddings.word_embeddings.weight.shape[0], dtype=p_term_weights.dtype, \
                                         device=p_term_weights.device).scatter_reduce(1, psg.input_ids[:,1:], p_term_weights.squeeze(), reduce='amax')
            
            # p_lexical_reps = torch.zeros(p_seq_hidden.shape[0], p_seq_hidden.shape[1], 30522, dtype=p_seq_hidden.dtype, device=p_seq_hidden.device) # (batch, seq, vocab)
            # p_lexical_reps = torch.scatter(p_lexical_reps, dim=-1, index=psg.input_ids[:,1:,None], src=p_term_weights)
            # p_lexical_reps = p_lexical_reps.max(-2).values


        
        if self.pooler is not None:
            p_semantic_reps = self.pooler(p=p_cls_hidden)  # D * d
        else:
            p_semantic_reps = None

            
        return p_lexical_reps, p_semantic_reps

    def encode_query(self, qry, skip_mlm):
        if qry is None:
            return None, None

        qry_out = self.lm_q(**qry, return_dict=True)
        q_seq_hidden = qry_out.hidden_states[-1] 
        q_cls_hidden = q_seq_hidden[:,0] # get [CLS] embeddings
        
        q_term_weights = self.term_weight_trans(q_seq_hidden[:,1:]) # batch, seq, 1
        
        if not skip_mlm:
            q_logits = qry_out.logits[:,1:] # batch, seq-1, vocab
            q_logits = self.softmax(q_logits)
            attention_mask = qry['attention_mask'][:,1:].unsqueeze(-1)
            q_lexical_reps = torch.max((q_logits * q_term_weights) * attention_mask, dim=-2).values
        else:
            # w/o MLM
            # q_term_weights = torch.relu(q_term_weights)
            q_lexical_reps = torch.zeros(q_seq_hidden.shape[0], self.lm_q.embeddings.word_embeddings.weight.shape[0], dtype=q_term_weights.dtype, \
                                         device=q_term_weights.device).scatter_reduce(1, qry.input_ids[:,1:], q_term_weights.squeeze(), reduce='amax')

            # q_lexical_reps = torch.zeros(q_seq_hidden.shape[0], q_seq_hidden.shape[1], 30522, dtype=q_seq_hidden.dtype, device=q_seq_hidden.device) # (batch, len, vocab)
            # q_lexical_reps = torch.scatter(q_lexical_reps, dim=-1, index=qry.input_ids[:,1:,None], src=q_term_weights)
            # q_lexical_reps = q_lexical_reps.max(-2).values

        
        
        if self.pooler is not None:
            q_semantic_reps = self.pooler(q=q_cls_hidden)
        else:
            q_semantic_reps = None

        return q_lexical_reps, q_semantic_reps

    @staticmethod
    def project_to_original_vacab(lexical_reps, input_type):
        if input_type == 'qry':
            index = torch.tensor(self.lm_q.get_current_vocab_mask(), device=lexical_reps.device).repeat(lexical_reps.shape[0], 1)
            full_hidden_states = torch.zeros(lexical_reps.shape[0], self.lm_q.get_vocab_size(), dtype=torch.float32, device=lexical_reps.device) # (batch, len, vocab)
        else:
            index = torch.tensor(self.lm_p.get_current_vocab_mask(), device=lexical_reps.device).repeat(lexical_reps.shape[0], 1)
            full_hidden_states = torch.zeros(lexical_reps.shape[0], self.lm_p.get_vocab_size(), dtype=torch.float32, device=lexical_reps.device) # (batch, len, vocab)
        return torch.scatter(full_hidden_states, dim=-1, index=index, src=lexical_reps) # fill value

    @staticmethod
    def merge_reps(lexical_reps, semantic_reps):
        dim = lexical_reps.shape[1] + semantic_reps.shape[1]
        merged_reps = torch.zeros(lexical_reps.shape[0], dim, dtype=lexical_reps.dtype, device=lexical_reps.device)
        merged_reps[:, :lexical_reps.shape[1]] = lexical_reps
        merged_reps[:, lexical_reps.shape[1]:] = semantic_reps
        return merged_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @staticmethod
    def build_term_weight_transform(model_args):
        term_weight_trans = LinearPooler(
            model_args.projection_in_dim,
            1,
            tied=not model_args.untie_encoder,
            name="TermWeightTrans"
        )
        term_weight_trans.load(model_args.model_name_or_path)
        return term_weight_trans

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            teacher_model: PreTrainedModel=None, 
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_q = AutoModel.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                if not model_args.skip_mlm:
                    lm_p = AutoModelForMaskedLM.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_p = AutoModel.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
            else:
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                else:
                    lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            if not model_args.skip_mlm:
                lm_q = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            else:
                lm_q = AutoModel.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler and not (model_args.projection_out_dim==0):
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        term_weight_trans = cls.build_term_weight_transform(model_args)


        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            teacher_model=teacher_model,
            pooler=pooler,
            term_weight_trans=term_weight_trans,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)

        if self.pooler is not None:
            self.pooler.save_pooler(output_dir)
        self.term_weight_trans.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

class DenseModelForInference(DenseModel):
    POOLER = LinearPooler  

    def __init__(
            self,
            model_args,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            term_weight_trans: nn.Module = None,
            lamb = 1,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler
        self.term_weight_trans = term_weight_trans
        self.softmax = nn.Softmax(dim=-1)
        self.model_args = model_args

    @torch.no_grad()
    def encode_passage(self, psg, skip_mlm):
        return super(DenseModelForInference, self).encode_passage(psg, skip_mlm)

    @torch.no_grad()
    def encode_query(self, qry, skip_mlm):
        return super(DenseModelForInference, self).encode_query(qry, skip_mlm)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_q = AutoModel.from_pretrained(
                        _qry_model_path,
                        **hf_kwargs
                    )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                if not model_args.skip_mlm:
                    lm_p = AutoModelForMaskedLM.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
                else:
                    lm_p = AutoModel.from_pretrained(
                        _psg_model_path,
                        **hf_kwargs
                    )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                if not model_args.skip_mlm:
                    lm_q = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **hf_kwargs)
                else:
                    lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            if not model_args.skip_mlm:
                lm_q = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **hf_kwargs)
            else:
                lm_q = AutoModel.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        TermWeightTrans_weights = os.path.join(model_name_or_path, 'TermWeightTrans.pt')
        TermWeightTrans_config = os.path.join(model_name_or_path, 'TermWeightTrans_config.json')
        if os.path.exists(TermWeightTrans_weights) and os.path.exists(TermWeightTrans_config):
            logger.info(f'found TermWeightTrans weight and configuration')
            with open(TermWeightTrans_config) as f:
                TermWeightTrans_config_dict = json.load(f)
            # Todo: add name to config
            TermWeightTrans_config_dict['name'] = 'TermWeightTrans'
            term_weight_trans = cls.POOLER(**TermWeightTrans_config_dict)
            term_weight_trans.load(model_name_or_path)
        else:
            term_weight_trans = None
        


        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            term_weight_trans=term_weight_trans,
            model_args=model_args
        )
        return model