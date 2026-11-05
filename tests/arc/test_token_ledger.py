import tempfile
import threading
import unittest
from pathlib import Path
from src.arc.token_ledger import TokenBudgetError, TokenLedger


class TokenLedgerTests(unittest.TestCase):
    def test_exact_reservation_exhaustion_restart_and_no_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'budget.sqlite'
            ledger=TokenLedger(path, 10)
            item=ledger.reserve('a', [1,2,3], 5, prefetch_allowance=1)
            self.assertEqual((item.reserved_tokens,item.used_after),(9,9))
            with self.assertRaises(TokenBudgetError): ledger.reserve('b',[4],1)
            reopened=TokenLedger(path,10)
            self.assertEqual(reopened.reserved(),9)
            with self.assertRaises(TokenBudgetError): reopened.reserve('a',[1,2,3],5)
            with self.assertRaises(TokenBudgetError): TokenLedger(path,11)
            self.assertEqual(reopened.reserved(),9)

    def test_parallel_reservations_do_not_exceed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'budget.sqlite'; ledger=TokenLedger(path,8)
            barrier=threading.Barrier(3); results=[]
            def worker(name):
                barrier.wait()
                try: ledger.reserve(name,[1,2],2,prefetch_allowance=0); results.append('ok')
                except TokenBudgetError: results.append('exhausted')
            threads=[threading.Thread(target=worker,args=(str(i),)) for i in range(3)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertCountEqual(results,['ok','ok','exhausted'])
            self.assertEqual(TokenLedger(path,8).reserved(),8)

    def test_rejects_malformed_input_before_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=TokenLedger(Path(directory)/'budget.sqlite',100)
            for ids, n in (([],1),([True],1),([-1],1),([1],0),([2**64],1)):
                with self.assertRaises(ValueError): ledger.reserve('x',ids,n)
            self.assertEqual(ledger.reserved(),0)

    def test_server_binding_persists_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget.sqlite'; ledger=TokenLedger(path,10)
            ledger.bind('a'*64)
            TokenLedger(path,10).bind('a'*64)
            with self.assertRaises(TokenBudgetError): TokenLedger(path,10).bind('b'*64)
            self.assertEqual(TokenLedger(path,10).reserved(),0)
