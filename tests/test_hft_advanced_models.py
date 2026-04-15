import unittest
import numpy as np
import torch
from src.hft.models.sde.avellaneda_stoikov import AvellanedaStoikovMarketMaker
from src.hft.models.dl.deeplob import CompactDeepLOB
from src.hft.execution.fill_prob import FillProbabilityGate

class TestHFTAdvancedModels(unittest.TestCase):

    def test_sde_inventory_quote_skew(self):
        # High Risk Aversion, Inventory = +10 (long position)
        # Should skew the quotes downward to offload inventory.
        mm = AvellanedaStoikovMarketMaker(risk_aversion=0.5, time_horizon=1.0)
        bid, ask = mm.calculate_quotes(mid_price=100.0, inventory=10.0, volatility=0.1, current_time=0.0)

        # Base spread should be symmetric around the reservation price
        reservation_price = 100.0 - (10.0 * 0.5 * (0.1**2) * 1.0) # r = 100 - (5 * 0.01) = 99.95
        spread = ask - bid
        self.assertAlmostEqual((bid + ask) / 2.0, reservation_price, places=2)
        self.assertTrue(reservation_price < 100.0) # Downward skew proven
        self.assertTrue(spread > 0)

    def test_compact_deeplob_forward_pass(self):
        # Input features: 10, Seq Len: 50, Batch size: 8
        model = CompactDeepLOB(input_features=10, sequence_length=50, num_classes=3)
        dummy_input = torch.randn(8, 50, 10)
        output = model(dummy_input)

        # Output should be (Batch Size, Num Classes)
        self.assertEqual(output.shape, (8, 3))

    def test_fill_probability_gate(self):
        gate = FillProbabilityGate(min_fill_prob=0.5)

        # Training data: feature = [queue_position_percentile]
        # 0.1 (front of queue) -> filled (1)
        # 0.9 (back of queue) -> missed (0)
        X_train = np.array([[0.1], [0.2], [0.8], [0.9]])
        y_train = np.array([1, 1, 0, 0])

        gate.fit(X_train, y_train)
        self.assertTrue(gate.is_fitted)

        # High fill probability expected for front of queue
        front_prob = gate.estimate_probability(np.array([0.15]))
        self.assertTrue(front_prob > 0.5)
        self.assertTrue(gate.is_executable(np.array([0.15])))

        # Low fill probability expected for back of queue
        back_prob = gate.estimate_probability(np.array([0.85]))
        self.assertTrue(back_prob < 0.5)
        self.assertFalse(gate.is_executable(np.array([0.85])))

if __name__ == '__main__':
    unittest.main()
