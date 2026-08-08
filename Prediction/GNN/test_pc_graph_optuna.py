"""Non-regression sur le cablage des parametres Optuna du PC-graph.

Contexte : une recherche de 300 essais n'a rien trouve parce que 5 des 7
parametres explores n'atteignaient jamais l'entrainement. `PCGraphModel` se
termine par `**_ignored`, donc `lambda_ordinality`, `lambda_coverage`,
`coverage_risk_shift`, `label_clamp_prob` et `weight_decay_theta` sont avales
par le constructeur : leur seule voie d'usage est `self.model_params`, qui
etait affecte sous garde `is None` -- donc fige sur le premier essai.

Symptome mesurable a l'epoque : un facteur 1100 sur `lambda_coverage` entre
deux essais par ailleurs comparables ne deplacait le score que de 0.04.
"""
import sys, types, unittest
from pathlib import Path
G = Path(__file__).resolve().parent
for p in (G, G.parent, G.parent.parent):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from GNN.pytorch_model_pc_graph import PCGraphTraining, PCGraphClmTraining

class FakeTrial:
    def __init__(s, n, vals): s.number, s._v = n, vals
    def suggest_int(s, name, lo, hi): return int(s._v[name])
    def suggest_float(s, name, lo, hi, log=False): return float(s._v[name])

class T(unittest.TestCase):
    def _t(self):
        t = PCGraphTraining.__new__(PCGraphTraining)
        t.model_params = None
        t.features_name = ['temp', 'elevation', 'fwi']
        t.ks, t.out_channels, t.task_type, t.device, t.horizon = 0, 5, 'classification', 'cpu', 0
        return t

    def test_model_params_suit_l_essai_courant(self):
        t = self._t()
        _, p0 = t.make_model(None, {'lambda_coverage': 20218.5, 'lr_x': 0.15})
        self.assertEqual(t.model_params['lambda_coverage'], 20218.5)
        _, p1 = t.make_model(None, {'lambda_coverage': 114.6, 'lr_x': 0.55})
        self.assertEqual(t.model_params['lambda_coverage'], 114.6,
                         "model_params fige sur le 1er essai : tous les essais "
                         "Optuna s'entraineraient avec ses lambdas")
        self.assertEqual(t.model_params['lr_x'], 0.55)

    def test_suggest_memorise_les_params_par_essai(self):
        t = PCGraphClmTraining.__new__(PCGraphClmTraining)
        t.OPTUNA_SPACE = {'lambda_coverage': ('float_log', 10., 30000.),
                          'n_internal_spatial': ('int', 8, 48)}
        t.suggest_loss_params(FakeTrial(0, {'lambda_coverage': 20218.5, 'n_internal_spatial': 11}), 'pc-gen')
        t.suggest_loss_params(FakeTrial(5, {'lambda_coverage': 114.6, 'n_internal_spatial': 23}), 'pc-gen')
        self.assertEqual(t._optuna_trial_params[0]['n_internal_spatial'], 11)
        self.assertEqual(t._optuna_trial_params[5]['n_internal_spatial'], 23,
                         "sans historique par essai, train_optuna ne peut pas "
                         "reconstruire l'architecture du meilleur essai")

    def test_restore_reaccorde_les_slices_avec_le_masque(self):
        """`train_optuna` charge les poids du meilleur essai dans le modele du
        DERNIER essai. `n_internal` etant fixe, le state_dict a la bonne forme et
        le chargement passe en silence -- mais `internal_spatial_slice`, qui n'est
        pas un buffer, reste desaccorde du masque charge."""
        t = self._t()
        # Noms au format attendu par `get_static_temporal_idx` (`<var>_<stat>`) :
        # les deux derniers sont statiques, les trois premiers temporels.
        t.features_name = ['temp_mean', 'fwi_mean', 'prcp_mean',
                           'foret_mean', 'population_mean']
        t._optuna_trial_params = {5: {'n_internal_spatial': 23, 'lr_x': 0.55}}
        base = {'topology': 'bimodal', 'n_internal': 128}

        dernier, _ = t.make_model(None, dict(base, n_internal_spatial=21))
        meilleur, _ = t.make_model(None, dict(base, n_internal_spatial=23))

        # Sans reconstruction : chargement silencieux, slices desaccordees.
        dernier.load_state_dict(meilleur.state_dict())
        n_sp = len(t._spatial_feature_indices())
        deg = dernier.state_dict()['core.mask'][dernier.sensory_slice,
                                                dernier.internal_slice].sum(0)
        self.assertEqual(int((deg == n_sp).sum()), 23, "le masque charge porte bien 23")
        self.assertEqual(dernier.internal_spatial_slice.stop
                         - dernier.internal_spatial_slice.start, 21,
                         "la slice, elle, est restee a 21 : c'est le bug")

        # Avec reconstruction : les deux concordent.
        rebati = t.restore_optuna_trial(5, None, base)
        rebati.load_state_dict(meilleur.state_dict())
        deg = rebati.state_dict()['core.mask'][rebati.sensory_slice,
                                               rebati.internal_slice].sum(0)
        self.assertEqual(rebati.internal_spatial_slice.stop
                         - rebati.internal_spatial_slice.start,
                         int((deg == n_sp).sum()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
