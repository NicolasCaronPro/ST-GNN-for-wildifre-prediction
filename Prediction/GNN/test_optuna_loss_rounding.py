"""Non-regression : l'arrondi de lisibilite ne doit pas ecraser l'espace de recherche.

`train_optuna` arrondit les floats proposes par `suggest_loss_params` avant de
construire la loss. L'arrondi d'origine, `round(v, 2)`, etait destructeur pour
les parametres tires en log-uniforme sur des plages basses : il envoyait une
grande part des tirages sur exactement 0.0. Les plages testees ici sont celles
reellement declarees dans `suggest_loss_params`.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent
for _p in (G, G.parent, G.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from GNN.pytorch_model_tools import round_loss_params

# (nom, borne basse, borne haute) des suggest_float(log=True) a plage basse
PLAGES_LOG_BASSES = [
    ('cllt_wkmin', 1e-4, 0.1),      # cllt
    ('tviolation', 1e-4, 0.5),      # ordinalnocoverage*, cornwithgains
    ('cllt_t', 1e-3, 0.5),          # cllt
]


class TestArrondiParamsLoss(unittest.TestCase):

    def _tirages_log(self, lo, hi, n=20000):
        rng = np.random.default_rng(0)
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

    def test_aucun_tirage_ecrase_a_zero(self):
        for nom, lo, hi in PLAGES_LOG_BASSES:
            with self.subTest(nom):
                arrondis = [round_loss_params({nom: float(v)})[nom]
                            for v in self._tirages_log(lo, hi, 2000)]
                zeros = sum(1 for v in arrondis if v == 0.0)
                self.assertEqual(zeros, 0,
                                 f"{nom} : {zeros} tirages mis a 0.0 "
                                 f"(round(v, 2) en produisait 26 a 57 %)")

    def test_resolution_preservee_sur_toute_la_plage(self):
        """Chaque decade de la plage log doit rester distinguable."""
        for nom, lo, hi in PLAGES_LOG_BASSES:
            with self.subTest(nom):
                tirages = self._tirages_log(lo, hi, 20000)
                distincts = len({round_loss_params({nom: float(v)})[nom] for v in tirages})
                self.assertGreater(distincts, 1000,
                                   f"{nom} : plus que {distincts} valeurs explorables")

    def test_erreur_relative_bornee(self):
        """4 chiffres significatifs : l'erreur reste relative, pas absolue."""
        for nom, lo, hi in PLAGES_LOG_BASSES:
            with self.subTest(nom):
                for v in self._tirages_log(lo, hi, 2000):
                    a = round_loss_params({nom: float(v)})[nom]
                    self.assertLess(abs(a - v) / v, 1e-3)

    def test_garde_fou_amont_non_defait(self):
        """`suggest_loss_params` fait `max(t_viol, 1e-6)` AVANT l'arrondi ;
        celui-ci ne doit pas ramener la valeur sous le plancher."""
        self.assertGreaterEqual(round_loss_params({'tviolation': 1e-6})['tviolation'], 1e-6)

    def test_lisibilite_conservee_sur_les_plages_usuelles(self):
        """Ce que l'arrondi est cense faire : couper les decimales parasites."""
        self.assertEqual(round_loss_params({'cewk_C': 0.7318492013})['cewk_C'], 0.7318)
        self.assertEqual(round_loss_params({'flwk_gamma': 2.000000001})['flwk_gamma'], 2.0)

    def test_non_floats_intacts(self):
        d = {'alpha_type': 'scalar', 'num_classes': 5, 'weight': None, 'flag': True}
        self.assertEqual(round_loss_params(d), d)


if __name__ == '__main__':
    unittest.main(verbosity=2)
