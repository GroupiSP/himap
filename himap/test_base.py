import unittest
from base import HSMM, HMM, GaussianHSMM

class TestHSMM(unittest.TestCase):
    def test_default_initialization(self):
        model = HSMM()
        self.assertEqual(model.n_states, 2)
        self.assertEqual(model.n_durations, 5)
        self.assertEqual(model.n_iter, 20)
        self.assertEqual(model.tol, 1e-2)
        self.assertFalse(model.left_to_right)
        self.assertIsNone(model.obs_state_len)
        self.assertIsNone(model.f_value)
        self.assertIsNone(model.random_state)
        self.assertEqual(model.name, "hsmm")

    def test_custom_initialization(self):
        model = HSMM(n_states=3, n_durations=4, n_iter=10, tol=1e-3, left_to_right=True, obs_state_len=5, f_value=10.0, random_state=42, name="custom_hsmm")
        self.assertEqual(model.n_states, 3)
        self.assertEqual(model.n_durations, 4)
        self.assertEqual(model.n_iter, 10)
        self.assertEqual(model.tol, 1e-3)
        self.assertTrue(model.left_to_right)
        self.assertEqual(model.obs_state_len, 5)
        self.assertEqual(model.f_value, 10.0)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.name, "custom_hsmm")

    def test_invalid_n_states(self):
        with self.assertRaises(ValueError):
            HSMM(n_states=1)

    def test_invalid_n_durations(self):
        with self.assertRaises(ValueError):
            HSMM(n_durations=0)

    def test_missing_obs_state_len(self):
        with self.assertRaises(ValueError):
            HSMM(obs_state_len=5)

    def test_missing_f_value(self):
        with self.assertRaises(ValueError):
            HSMM(f_value=10.0)

class TestHMM(unittest.TestCase):
    def test_default_initialization(self):
        model = HMM()
        self.assertEqual(model.n_states, 2)
        self.assertEqual(model.n_obs_symbols, 30)
        self.assertEqual(model.n_iter, 100)
        self.assertEqual(model.tol, 1e-2)
        self.assertTrue(model.left_to_right)
        self.assertEqual(model.name, "hmm")

    def test_custom_initialization(self):
        model = HMM(n_states=3, n_obs_symbols=40, n_iter=50, tol=1e-3, left_to_right=False, name="custom_hmm")
        self.assertEqual(model.n_states, 3)
        self.assertEqual(model.n_obs_symbols, 40)
        self.assertEqual(model.n_iter, 50)
        self.assertEqual(model.tol, 1e-3)
        self.assertFalse(model.left_to_right)
        self.assertEqual(model.name, "custom_hmm")

    def test_invalid_n_states(self):
        with self.assertRaises(AssertionError):
            HMM(n_states=1)

    def test_invalid_n_obs_symbols(self):
        with self.assertRaises(AssertionError):
            HMM(n_obs_symbols=0)

    def test_invalid_n_iter(self):
        with self.assertRaises(AssertionError):
            HMM(n_iter=0)

    def test_invalid_tol(self):
        with self.assertRaises(AssertionError):
            HMM(tol=0)

class TestGaussianHSMM(unittest.TestCase):
    def test_default_initialization(self):
        model = GaussianHSMM()
        self.assertEqual(model.n_states, 2)
        self.assertEqual(model.n_durations, 5)
        self.assertEqual(model.n_iter, 100)
        self.assertEqual(model.tol, 0.5)
        self.assertTrue(model.left_to_right)
        self.assertIsNone(model.obs_state_len)
        self.assertIsNone(model.f_value)
        self.assertIsNone(model.random_state)
        self.assertEqual(model.name, "hsmm")

    def test_custom_initialization(self):
        model = GaussianHSMM(n_states=3, n_durations=4, n_iter=10, tol=1e-3, left_to_right=True, obs_state_len=5, f_value=10.0, random_state=42, name="custom_GaussianHSMM")
        self.assertEqual(model.n_states, 3)
        self.assertEqual(model.n_durations, 4)
        self.assertEqual(model.n_iter, 10)
        self.assertEqual(model.tol, 1e-3)
        self.assertTrue(model.left_to_right)
        self.assertEqual(model.obs_state_len, 5)
        self.assertEqual(model.f_value, 10.0)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.name, "custom_GaussianHSMM")

    def test_invalid_n_states(self):
        with self.assertRaises(ValueError):
            GaussianHSMM(n_states=1)

    def test_invalid_n_durations(self):
        with self.assertRaises(ValueError):
            GaussianHSMM(n_durations=0)

    def test_missing_obs_state_len(self):
        with self.assertRaises(ValueError):
            GaussianHSMM(obs_state_len=5)

    def test_missing_f_value(self):
        with self.assertRaises(ValueError):
            GaussianHSMM(f_value=10.0)

if __name__ == '__main__':
    unittest.main()