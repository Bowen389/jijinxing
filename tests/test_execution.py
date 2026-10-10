import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.execution import Ledger, affordable, execute_conc, fee, sell_fee
from engine.security import status_for, limit_flags

D = pd.Timestamp('2026-10-12')
CFG = dict(k=3, open_cost=.0005, close_cost=.001, min_cost=5, slip=.001)

class Quotes:
    def __init__(self, overrides=None):
        self.overrides = overrides or {}
    def quote(self, d, j):
        return dict(raw_open=10., raw_close=100., factor=1., susp=False,
                    dn_open=False, up_open=False, buy_ok=True, **self.overrides.get(j, {}))

def held():
    a = Ledger(40000)
    for j in ('SH600001', 'SH600002', 'SH600003'):
        a.s['positions'][j] = dict(shares=100, cost=1000, buy_date='2026-10-09', last_px=10, factor=1)
    return a

class Market:
    def __init__(self, blocked=None):
        self.blocked = blocked or {}
    def quote(self, d, j):
        q = dict(raw_open=10., raw_close=100., factor=1., susp=False,
                 dn_open=False, up_open=False, buy_ok=True)
        q.update(self.blocked.get(j, {}))
        return q

def pending():
    return dict(sell=['SH600001'], buy=[dict(inst='SH600004')], backup=[dict(inst='SH600005')], slot=999999)

@pytest.mark.parametrize('reason', ['dn_open', 'susp'])
def test_failed_sale_cannot_create_fourth_holding(reason):
    a = held()
    execute_conc(a, Market({'SH600001': {reason: True}}), D, pending(), {}, CFG)
    assert set(a.s['positions']) == {'SH600001', 'SH600002', 'SH600003'}
    assert len(a.s['positions']) == 3

@pytest.mark.parametrize('block', [{'up_open': True}, {'susp': True}, {'buy_ok': False}, {'raw_open': np.nan}])
def test_backup_and_cash(block):
    a = held()
    a.mark(Market(), D, at='open')
    execute_conc(a, Market({'SH600004': block}), D, pending(), {}, CFG)
    assert 'SH600005' in a.s['positions'] and 'SH600004' not in a.s['positions']
    assert len(a.s['positions']) == 3 and a.s['cash'] >= 0

def test_open_value_does_not_read_close_or_signal_slot():
    a = held()
    a.mark(Market(), D, at='open')
    assert a.total() == 43000
    execute_conc(a, Market(), D, pending(), {}, CFG)
    assert a.s['positions']['SH600004']['shares'] == 1400
    a.mark(Market(), D)
    assert a.s['positions']['SH600004']['last_px'] == 100

def test_minimum_commission_and_separate_stamp():
    assert sell_fee(1000, CFG, D) == 5.5
    assert sell_fee(1000, CFG, pd.Timestamp('2023-08-27')) == 6
    assert fee(1000, .0005) == 5

@pytest.mark.parametrize('budget,expected', [(1999, 0), (2506, 250), (2010, 200)])
def test_star_minimum_and_single_share_increments(budget, expected):
    assert affordable('SH688001', budget, 10, .0005, 5) == expected

def test_tplus1_and_no_negative_cash():
    a = Ledger(1005)
    a.trade(D, '买入', 'SH600001', 100, 10, 5)
    with pytest.raises(ValueError, match='T\\+1'):
        a.trade(D, '卖出', 'SH600001', 100, 10, 5)
    with pytest.raises(ValueError, match='cash'):
        a.trade(D, '买入', 'SH600002', 100, 10, 5)

def test_corporate_action_applied_once():
    a = held()
    a.mark(Market({'SH600001': dict(factor=2)}), D, at='open')
    a.mark(Market({'SH600001': dict(factor=2)}), D)
    assert a.s['positions']['SH600001']['shares'] == 200

def test_missing_status_and_listing_info_never_buy(tmp_path, monkeypatch):
    import engine.security as s
    monkeypatch.setattr(s, 'EXTRA', str(tmp_path))
    f = pd.DataFrame([dict(datetime=D, instrument='SH600001')])
    assert not status_for(f).buy_ok.iloc[0]
    pd.DataFrame([dict(datetime=D, instrument='SH600001', isST=0, tradestatus=1)]).to_parquet(tmp_path/'security_status.parquet')
    assert not status_for(f).buy_ok.iloc[0]

@pytest.mark.parametrize('st,expected', [(0, True), (1, False)])
def test_dated_status_and_not_backdated_name(tmp_path, monkeypatch, st, expected):
    import engine.security as s
    monkeypatch.setattr(s, 'EXTRA', str(tmp_path))
    f = pd.DataFrame([dict(datetime=D, instrument='SH600001')])
    pd.DataFrame([dict(datetime=D, instrument='SH600001', isST=st, tradestatus=1)]).to_parquet(tmp_path/'security_status.parquet')
    pd.DataFrame([dict(instrument='SH600001', ipoDate='2000-01-01', outDate='', code_name='退测试', observed='2026-10-13')]).to_parquet(tmp_path/'security_basic.parquet')
    out = status_for(f)
    assert bool(out.buy_ok.iloc[0]) == expected
    assert not out.is_delisted.iloc[0]

def test_st_and_unknown_mainboard_exit_block():
    inst = pd.Series(['SH600001', 'SH600002', 'SH688001'])
    flags = limit_flags(inst, pd.Series([D]*3), pd.Series([9.5,9.5,8.]), pd.Series([9.5,9.5,8.]),
                        pd.Series([-.05,-.05,-.2]), pd.Series([False]*3), pd.Series([True,False,False]), pd.Series([True,False,True]))
    assert flags['dn_open'].all()
    np.testing.assert_allclose(flags['lim'], [.05,.05,.2])

def test_missing_previous_price_blocks_execution():
    flags=limit_flags(pd.Series(['SH600001']), pd.Series([D]), pd.Series([10.]), pd.Series([10.]),
                      pd.Series([np.nan]),pd.Series([False]),pd.Series([False]),pd.Series([True]))
    assert flags['up_open'].iloc[0] and flags['dn_open'].iloc[0]

def test_historical_gem_st_before_twenty_percent_regime():
    from engine.common import limit_pct
    out=limit_pct(pd.Series(['SZ300001','SZ300001']),pd.to_datetime(pd.Series(['2020-08-21','2020-08-24'])),[True,True])
    np.testing.assert_allclose(out,[.05,.2])
