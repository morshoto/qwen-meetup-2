"""Exploratory server contracts use their own cooperating budget."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.arc.prompting import system_prompt_for_kind
from src.arc.rsi_comparison_transport import CapturedComparisonTransport
from src.arc.rsi_exploratory_budget import ExploratoryBudget
from src.arc.rsi_exploratory_contract import ContractCheckedExploratoryTransport
from src.arc.rsi_exploratory_transport import CapturedExploratoryTransport


class ExploratoryContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.a='http://127.0.0.1:10001/v1'; self.b='http://127.0.0.1:10002/v1'
        raw=CapturedComparisonTransport(root=self.root,parent_url=self.a,candidate_url=self.b,parent_adapter='none',candidate_adapter='b'*64,max_request_seconds=2)
        self.transport=CapturedExploratoryTransport(raw)
        self.budget=ExploratoryBudget.create(self.root,registration_sha256='a'*64,generation=0)
        self.states={self.a:{'ledger_id':'1'*32,'token_limit':2000000,'reserved_tokens':0},
                     self.b:{'ledger_id':'2'*32,'token_limit':2000000,'reserved_tokens':0}}
        messages=[{'role':'system','content':system_prompt_for_kind('e5_self_solve')},{'role':'user','content':'prompt'}]
        self.reservation=512+len(json.dumps(messages,ensure_ascii=False).encode())+1024
    def make(self):
        return ContractCheckedExploratoryTransport(self.transport,budget=self.budget,registration_sha256='a'*64,
                                                    generation=0,expected_max_prompt_tokens=4096)
    def reply(self,url,*args,**kwargs): return dict(self.states[url])
    @patch('src.arc.rsi_exploratory_contract.verify_guarded_contract')
    def test_requires_exact_reservation_and_routes(self,verify):
        verify.side_effect=self.reply; guard=self.make()
        with patch.object(CapturedExploratoryTransport,'__call__',return_value={'code':'x','request_id':'1'*32,'usage':None}) as dispatch:
            with self.assertRaises(ValueError): guard('prompt',arm='base',adapter='none',task_id='t',sample=0,seed=1,timeout=1)
            self.budget.reserve_model(self.reservation,kind='evaluation')
            guard('prompt',arm='base',adapter='none',task_id='t',sample=0,seed=1,timeout=1)
            self.assertEqual(dispatch.call_count,1)
    @patch('src.arc.rsi_exploratory_contract.verify_guarded_contract')
    def test_rejects_changed_ledgers_or_server_overspend(self,verify):
        verify.side_effect=self.reply; guard=self.make(); self.budget.reserve_model(self.reservation,kind='evaluation')
        self.states[self.a]['reserved_tokens']=self.reservation+1
        with patch.object(CapturedExploratoryTransport,'__call__') as dispatch:
            with self.assertRaises(ValueError): guard('prompt',arm='base',adapter='none',task_id='t',sample=0,seed=1,timeout=1)
            dispatch.assert_not_called()
    @patch('src.arc.rsi_exploratory_contract.verify_guarded_contract')
    def test_rejects_shared_or_overlarge_ledgers(self,verify):
        verify.side_effect=self.reply
        self.states[self.b]['ledger_id']=self.states[self.a]['ledger_id']
        with self.assertRaises(ValueError):self.make()
        self.states[self.b]['ledger_id']='2'*32; self.states[self.b]['token_limit']+=1
        with self.assertRaises(ValueError):self.make()
