
import unittest
from unittest.mock import MagicMock
import random
from bandit.loop import BanditLoop
from bandit.schemas import LoopContext, BanditState, SelectionResult, DispatchResult

class TestBanditLoop(unittest.TestCase):
    def setUp(self):
        self.loop = BanditLoop()
        self.context = LoopContext()
        self.context.bandit_state = BanditState()
        self.context.rng = random.Random(42)
        self.context.log_writer = MagicMock()

    def test_import_success(self):
        # This will test if lazy imports now succeed
        
        # Test posterior (now updater)
        engine = self.loop._get_posterior()
        self.assertIsNotNone(engine, "Posterior engine should not be None")
        from bandit.updater import PosteriorUpdateEngine
        self.assertIsInstance(engine, PosteriorUpdateEngine)

        # Test rollback (now safety)
        net = self.loop._get_rollback()
        self.assertIsNotNone(net, "Rollback net should not be None")
        from bandit.safety import RollbackSafetyNet
        self.assertIsInstance(net, RollbackSafetyNet)

if __name__ == "__main__":
    unittest.main()
