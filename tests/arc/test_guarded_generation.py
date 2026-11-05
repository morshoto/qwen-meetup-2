import tempfile
import unittest
from pathlib import Path
from src.arc.guarded_generation import GuardedGenerator
from src.arc.token_ledger import TokenBudgetError, TokenLedger


class Driver:
    def __init__(self, ids): self.ids=ids; self.started=0; self.next_calls=0; self.received=None
    def encode_messages(self, messages): return list(self.ids)
    def generate_ids(self, prompt_ids, max_tokens):
        self.started+=1; self.received=(prompt_ids,max_tokens)
        for token in range(100):
            self.next_calls+=1
            yield token
    def decode(self, ids): return ','.join(map(str,ids))


class GuardedGenerationTests(unittest.TestCase):
    def test_reserves_exact_input_plus_output_and_stops_iteration(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'budget.sqlite',20);driver=Driver([4,5,6])
            result=GuardedGenerator(ledger,driver,max_prompt_tokens=10).generate('one',[],4)
            self.assertEqual(result.text,'0,1,2,3')
            self.assertEqual(result.output_tokens,4)
            self.assertEqual(driver.received,([4,5,6],4))
            self.assertEqual(driver.next_calls,4)
            self.assertEqual(ledger.reserved(),8)

    def test_rejects_before_dispatch_and_never_refunds_failure(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'budget.sqlite',6);driver=Driver([1,2,3])
            guard=GuardedGenerator(ledger,driver,max_prompt_tokens=2)
            with self.assertRaises(ValueError): guard.generate('large',[],1)
            self.assertEqual(driver.started,0); self.assertEqual(ledger.reserved(),0)
            guard=GuardedGenerator(ledger,driver,max_prompt_tokens=3)
            with self.assertRaises(TokenBudgetError): guard.generate('over',[],3)
            self.assertEqual(driver.started,0); self.assertEqual(ledger.reserved(),0)
            with self.assertRaisesRegex(RuntimeError,'decode failed'):
                driver.decode=lambda ids: (_ for _ in ()).throw(RuntimeError('decode failed'))
                guard.generate('failed',[],2)
            self.assertEqual(ledger.reserved(),6)
            with self.assertRaises(TokenBudgetError): guard.generate('failed',[],1)

    def test_stop_token_counts_against_budget(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'budget.sqlite',10);driver=Driver([7])
            result=GuardedGenerator(ledger,driver,max_prompt_tokens=3,stop_token_ids=frozenset({1})).generate('stop',[],5)
            self.assertEqual(result.text,'0'); self.assertEqual(result.output_tokens,2)
            self.assertEqual(ledger.reserved(),7)

    def test_one_lookahead_is_included_in_reserved_total(self):
        class LookaheadDriver(Driver):
            def __init__(self): super().__init__([1,2,3]); self.computed=0
            def generate_ids(self,ids,max_tokens):
                self.computed+=1
                for token in range(max_tokens):
                    self.computed+=1
                    yield token
        with tempfile.TemporaryDirectory() as d:
            ledger=TokenLedger(Path(d)/'budget.sqlite',8); driver=LookaheadDriver()
            result=GuardedGenerator(ledger,driver,max_prompt_tokens=3).generate('one',[],4)
            self.assertEqual(driver.computed,5)
            self.assertEqual(result.reservation.reserved_tokens,3+driver.computed)
            self.assertEqual(ledger.reserved(),8)
