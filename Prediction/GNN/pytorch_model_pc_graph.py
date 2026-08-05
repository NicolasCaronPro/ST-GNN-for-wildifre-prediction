"""Classe Training dediee au modele PC-graph (predictive coding sur graphe).

`PCGraphTraining` reutilise le pipeline de pre-processing / prediction du
projet en heritant de `Model_Torch` (pytorch_model_class.py) -- meme famille
"flat" (pas de graphe spatial transmis au forward, cf.
GNN.dataloader.wrapped_train_deep_learning_1D) que GRU/itransformer/TFN.

Point important, different des autres modeles du pipeline : **un PC-graph ne
s'entraine pas avec une loss externe**. L'article (Algorithme 1) clampe
sensoriel *et* label ensemble, relaxe les noeuds internes jusqu'a l'equilibre,
puis fait une seule mise a jour de poids sur l'energie (Eq. 2) a cet
equilibre -- il n'y a jamais de comparaison "prediction vs cible" via un
`criterion` externe. `launch_batch` est donc surcharge pour utiliser
`PCGraphModel.training_energy(...)` comme signal d'apprentissage a la place
de `Training.calculate_loss`, et `get_loss` est neutralise (retourne un
objet vide : aucune fonction de perte n'est jamais instanciee). Le champ
JSON `"loss"` est conserve uniquement pour ne pas casser le nommage des
dossiers de sortie (`check_.../{model}_{infos}`) -- sa valeur ('pc' par
convention) n'a aucun effet fonctionnel.

`make_model` (construction du `PCGraphModel` plutot que passage par le
factory global `forecasting_models.pytorch.tools_2.make_model`) et
`get_optimizer` (weight decay dedie a `theta`) sont egalement surcharges.
`train_run`, `_predict_tensor`, `predict`, `predict_proba`,
`shapley_additive_explanation`, etc. viennent de
`Training`/`SplitTraining`/`Model_Torch` sans aucune modification : a
l'inference/prediction (label libre, jamais clampe), `PCGraphModel.forward`
respecte bien le contrat `forward(x, z_prev=None) -> (output, logits,
hidden)` attendu par ce code herite.

Ajoute en plus les outils de visualisation/interpretabilite propres au
graphe PC (convergence de l'energie, topologie apprise, classement causal
des features) demandes en complement de l'entrainement/prediction standard.
"""
import numpy as np
import torch
from torch import optim
from pathlib import Path

from GNN.pytorch_model_class import Model_Torch
from GNN.tools import check_and_create_path, logger
from GNN.config import ids_columns, targets_columns
from forecasting_models.pytorch.pc_graph import PCGraphModel


class _NoOpCriterion:
    """Placeholder pour `self.criterion` : un PC-graph n'utilise aucune loss
    externe (cf. docstring du module). Volontairement vide -- l'absence de
    `transform`/`calibrate`/`get_learnable_parameters`/`_preprocess` fait que
    tous les `hasattr`/`has_method` de `Training` le traitent comme un
    no-op, sans qu'il faille toucher a ce code herite."""
    pass


_DEFAULT_PARAMS = {
    'n_internal': 128,
    't_train': 20,
    't_query': 50,
    'lr_x': 0.5,
    'init_std': 0.05,
    'weight_decay_theta': 1e-2,
    'grad_mode': 'phantom',   # 'phantom' (defaut, gradient au point fixe) ou 'bptt' (recherche)
    'topology': 'full',       # 'full' (graphe plein) ou 'layered' (sensoriel<->interne<->label)
}

# Probabilite par defaut de clamper le label pendant l'entrainement du
# PCGraph "generatif semi-supervise" (PCGraphGenTraining) -- cf. docstring de
# la classe. Surchargeable via le JSON ("params": {"label_clamp_prob": ...}).
_DEFAULT_LABEL_CLAMP_PROB = 0.7


class PCGraphTraining(Model_Torch):

    def make_model(self, graph, custom_model_params):
        params = dict(_DEFAULT_PARAMS)
        params.update({
            'in_dim': len(self.features_name),
            'k_days': self.ks,
            'out_channels': self.out_channels,
            'task_type': self.task_type,
            'device': self.device,
            'horizon': self.horizon,
        })
        if custom_model_params:
            params.update(custom_model_params)

        if params.get('topology') == 'bimodal' and 'spatial_feature_idx' not in params:
            params['spatial_feature_idx'] = self._spatial_feature_indices()
            n_static = len(params['spatial_feature_idx'])
            logger.info(
                f'[PCGraph] topologie bimodale (separation via get_static_temporal_idx) : '
                f'{n_static} features statiques / {len(self.features_name) - n_static} temporelles, '
                f'{params.get("n_internal_spatial", 16)} noeuds internes statiques '
                f'sur {params["n_internal"]}'
            )

        model = PCGraphModel(**params).to(self.device)

        if self.model_params is None:
            self.model_params = params

        return model, params

    def _spatial_feature_indices(self):
        """Indices des features STATIQUES (constantes a l'interieur d'un
        cluster) pour `topology='bimodal'`.

        Delegue a `GNN.tools.get_static_temporal_idx`, la fonction de
        separation statique/temporel deja utilisee par le reste du pipeline
        (cf. ses appels dans `pytorch_model_tools.py` pour `static_idx` /
        `temporal_idx`) -- plutot qu'une re-classification maison, pour que
        le PC-graph partitionne EXACTEMENT comme les autres modeles et suive
        automatiquement toute evolution des listes de variables.

        Verifie sur ce jeu de donnees : 51 statiques / 106 temporelles, et
        les 51 ont bien 100% de variance inter-cluster (donc strictement
        constantes dans un cluster), contre ~5-45% pour les temporelles."""
        from GNN.tools import get_static_temporal_idx

        static_idx, _ = get_static_temporal_idx(list(self.features_name))
        return [int(i) for i in static_idx]

    def get_optimizer(self, criterion):
        parameters = self.get_learnable_parameters(criterion)
        weight_decay = _DEFAULT_PARAMS['weight_decay_theta']
        if self.model_params is not None:
            weight_decay = self.model_params.get('weight_decay_theta', weight_decay)
        return optim.Adam(parameters, lr=self.lr, weight_decay=weight_decay)

    def get_loss(self, loss_name, loss_params):
        # Un PC-graph n'utilise jamais de loss externe (cf. docstring du
        # module) -- `loss_name` (le "pc" du JSON) ne sert qu'au nommage des
        # dossiers de sortie, pas a selectionner une fonction de perte ici.
        return _NoOpCriterion()

    def _prepare_energy_inputs(self, data):
        """Facteur commun a `PCGraphTraining.launch_batch` et
        `PCGraphGenTraining.launch_batch` : reconstruit les entrees
        sensorielles (`inputs_horizon`) et le label one-hot/continu clampable
        (`y_clamp`) a partir d'un batch du loader, plus les poids
        (echantillonnage/under-over-sampling, independants du bruit de
        label) a appliquer sur l'energie."""
        inputs, labels, _ = data

        if self.horizon != 0:
            raise NotImplementedError(
                'PCGraphTraining.launch_batch : horizon > 0 non supporte en v1 '
                '(apprentissage energetique natif, cf. docstring du module).'
            )

        target, weights = self.compute_weights_and_target(
            labels, -1, ids_columns, False, None, -1 - self.horizon
        )
        inputs_horizon = self.compute_inputs(inputs, -1 - self.horizon, 'current')

        if self.task_type in ('classification', 'binary', 'uclassification', 'ordinal-classification'):
            temperature = None
            if self.model_params is not None:
                temperature = self.model_params.get('ordinal_target_temperature')
            if temperature:
                dept = labels[:, ids_columns.index('departement'), -1].view(-1)
                y_clamp = self._ordinal_soft_target(
                    target, self.model.out_channels, temperature, departement=dept
                )
            else:
                y_clamp = torch.nn.functional.one_hot(
                    target.long().view(-1), num_classes=self.model.out_channels
                ).float()
        else:
            y_clamp = target.view(-1, self.model.out_channels).float()

        return inputs_horizon, y_clamp, weights

    def _risk_scale_table(self):
        """Table `m[departement, classe]` = valeur moyenne de la cible REELLE
        pour chaque (departement, classe vraie), en unites sigma. Calculee une
        seule fois sur le train, puis mise en cache.

        C'est la "regle graduee" de l'espace cible : au lieu de supposer les
        classes equidistantes, on utilise l'ecart de risque REELLEMENT observe
        entre elles. Verifie empiriquement, l'ecart au lineaire depend beaucoup
        de la cible -- rapport ecart_max/ecart_min de 1.7 pour `nbsinister`
        (des classes kmeans sur un comptage entier retombent sur le comptage),
        mais 6.0 pour `time_intervention` et 6.3 pour `ressource`. Supposer
        l'equidistance est donc faux sur la plupart des cibles.

        Deux choix qui suivent le pipeline existant plutot que des conventions
        maison :
        - normalisation par `self.scoring.sigma`, exactement comme
          `Scoring.evaluation_scoring` qui fait `df['Y'] /= self.sigma` avant
          d'ajuster ses mu -- la geometrie d'entrainement est ainsi dans les
          memes unites que celle de l'evaluation ;
        - un `m` PAR DEPARTEMENT, avec retrecissement (empirique-bayesien)
          vers l'estimation mutualisee proportionnellement a l'effectif de la
          cellule. Les departements ont des echelles de risque non comparables
          et il ne faut pas les ecraser ; mais les cellules rares (ex. classe 4
          d'un departement, parfois 0 echantillon) retombent proprement sur
          l'estimation globale au lieu de produire un NaN."""
        if getattr(self, '_risk_scale_cache', None) is not None:
            return self._risk_scale_cache

        n_classes = self.model.out_channels
        dept_idx = ids_columns.index('departement')
        target_idx = len(ids_columns) + targets_columns.index(self.scoring_target_column())

        sums, counts = {}, {}
        for _, labels, _ in self.train_loader:
            tgt, _ = self.compute_weights_and_target(labels, -1, ids_columns, False, None, -1)
            k = tgt.view(-1).long()
            y = labels[:, target_idx, -1].view(-1).float()
            d = labels[:, dept_idx, -1].view(-1).long()
            for di, ki, yi in zip(d.tolist(), k.tolist(), y.tolist()):
                sums[(di, ki)] = sums.get((di, ki), 0.0) + yi
                counts[(di, ki)] = counts.get((di, ki), 0) + 1

        depts = sorted({di for di, _ in sums})
        pooled_sum = [0.0] * n_classes
        pooled_cnt = [0] * n_classes
        for (di, ki), s in sums.items():
            pooled_sum[ki] += s
            pooled_cnt[ki] += counts[(di, ki)]
        pooled = [pooled_sum[c] / pooled_cnt[c] if pooled_cnt[c] else 0.0 for c in range(n_classes)]

        # Retrecissement : lambda = n / (n + n0), n0 = 10 echantillons. Une
        # cellule vide donne lambda = 0 (pur pooled), une cellule bien fournie
        # tend vers son estimation propre.
        n0 = 10.0
        table = torch.zeros(len(depts), n_classes)
        for i, di in enumerate(depts):
            for c in range(n_classes):
                n = counts.get((di, c), 0)
                own = sums[(di, c)] / n if n else 0.0
                lam = n / (n + n0)
                table[i, c] = lam * own + (1.0 - lam) * pooled[c]

        sigma = getattr(self.scoring, 'sigma', None) or 1.0
        table = table / sigma

        self._risk_scale_cache = (torch.tensor(depts, dtype=torch.long), table)
        logger.info(
            f'[PCGraph] echelle de risque (unites sigma, sigma={sigma:.4f}) par departement :\n'
            + '\n'.join(f'    dept {d:>3} : ' + ' '.join(f'{v:7.3f}' for v in table[i].tolist())
                        for i, d in enumerate(depts))
        )
        return self._risk_scale_cache

    def scoring_target_column(self):
        """Colonne de `targets_columns` portant la valeur REELLE que le
        scoring monotone cherche a ordonner (cf. `dataloader.py`, qui choisit
        `y_true_real` selon le nom de la cible avant d'appeler
        `evaluation_scoring`)."""
        name = self.target_name
        for candidate in ('time_intervention', 'ressource', 'burned_area', 'nbsinister'):
            if candidate in name:
                return candidate
        return 'nbsinister'

    def _ordinal_soft_target(self, target, num_classes, temperature, departement=None):
        """Cible ORDINALE douce : `y_j = exp(-|j - k| / T)` pour la classe
        vraie `k`, utilisee a la place du one-hot quand
        `ordinal_target_temperature` est renseigne.

        Motivation : l'energie du PC-graph est un ecart quadratique
        (`E = 1/2 * sum_i (x_i - mu_i)^2`), donc le one-hot la rend AVEUGLE a
        la distance ordinale -- pour une cible en classe 0, predire 1 ou
        predire 4 coute exactement 2.0 dans les deux cas. Les cibles ci-
        dessous rendent ce cout croissant avec |j - k| (~0.86 contre ~2.13
        pour ce meme exemple a T=1), ce qui donne enfin au reseau une raison
        d'ordonner ses niveaux de risque -- le probleme identifie sur le
        profil mu (non monotone) des runs precedents.

        `exp(-|j-k|/T)` plutot qu'un `softmax(-|j-k|/T)` : le maximum vaut
        exactement 1.0 en `j == k`, donc les valeurs clampees gardent
        l'echelle du one-hot. Un softmax normaliserait a somme 1 et
        ecraserait le maximum a ~0.5, diluant le poids du terme label face
        au terme sensoriel dans l'energie globale (`energy_per_sample` somme
        sur TOUS les noeuds).

        `T` regle la portee : T -> 0 redonne le one-hot, T grand aplatit vers
        une cible uniforme (plus aucune information de classe).

        Si `departement` est fourni, la distance entre classes n'est plus
        l'ecart d'INDICE `|j - k|` (qui suppose les classes equidistantes)
        mais l'ecart de RISQUE REEL `|m(j) - m(k)|` lu dans
        `_risk_scale_table()`, en unites sigma et propre a chaque
        departement. Confondre deux classes coute alors proportionnellement
        a leur vraie difference de risque, ce qui est l'information que le
        scoring monotone evalue -- et non une equidistance postulee."""
        k = target.long().view(-1)
        if departement is None:
            j = torch.arange(num_classes, device=k.device).view(1, -1)
            dist = (j - k.view(-1, 1)).abs().float()
        else:
            depts, table = self._risk_scale_table()
            table = table.to(k.device)
            # departement -> ligne de la table (recherche par egalite, le
            # nombre de departements est petit)
            rows = (departement.view(-1, 1).long() == depts.to(k.device).view(1, -1)).float().argmax(dim=1)
            m = table[rows]                                   # (B, n_classes)
            m_true = m.gather(1, k.view(-1, 1))               # (B, 1)
            dist = (m - m_true).abs()
        return torch.exp(-dist / float(temperature))

    def launch_batch(self, data, criterion, batch_type, do_update):
        if data[0].shape[0] == 1:
            return 0, 0

        inputs_horizon, y_clamp, weights = self._prepare_energy_inputs(data)

        energy_per_sample = self.model.training_energy(inputs_horizon, y_clamp)
        w = weights.view(-1).clamp_min(0)
        total_loss = (energy_per_sample * w).sum() / w.sum().clamp_min(1e-8)

        return total_loss, {'total_loss': total_loss}

    # ------------------------------------------------------------------
    # Visualisation / interpretabilite
    # ------------------------------------------------------------------

    def _analysis_dir(self, dir_output=None) -> Path:
        dir_output = Path(dir_output) if dir_output is not None else self.dir_log / 'pcgraph_analysis'
        check_and_create_path(dir_output / '_')
        return dir_output

    def _sample_batch(self, loader=None, n=32):
        loader = loader or self.val_loader or self.train_loader
        if loader is None:
            raise ValueError('No loader available to sample a batch from (train/val loader is None).')
        inputs, _, _ = next(iter(loader))
        return inputs[:n].to(self.device)

    def plot_energy_convergence(self, x=None, loader=None, t_steps=None, outname='energy_convergence', dir_output=None):
        """Courbe energie (Eq. 2) vs pas de relaxation, pour verifier que
        `t_query` suffit a la convergence de l'inference (diagnostic Fig.
        15-17 de l'article) -- reutilise un batch du loader de validation si
        `x` n'est pas fourni."""
        from matplotlib import pyplot as plt

        model = self.model
        model.eval()
        x = x if x is not None else self._sample_batch(loader)
        t_steps = t_steps or model.t_query

        with torch.no_grad():
            B = x.shape[0]
            x_sensory = x.reshape(B, -1)
            x_init = x_sensory.new_zeros(B, model.n_nodes)
            x_init[:, model.sensory_slice] = x_sensory
            clamp_mask = model.clamp_mask
            free = (~clamp_mask).float()

            energies = []
            xi = x_init.clone()
            for _ in range(t_steps):
                xi = xi.detach().requires_grad_(True)
                with torch.enable_grad():
                    E = model.core.energy(xi)
                    (grad,) = torch.autograd.grad(E, xi)
                energies.append(E.item())
                xi = (xi - model.lr_x * grad * free).detach()

        dir_output = self._analysis_dir(dir_output)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(energies, lw=1.5)
        ax.set_xlabel('pas de relaxation t')
        ax.set_ylabel('energie (Eq. 2)')
        ax.set_title(f'Convergence de la relaxation ({self.name})')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(dir_output / f'{outname}.png', dpi=150)
        plt.close(fig)
        logger.info(f'[PCGraph] energy convergence plot saved to {dir_output / f"{outname}.png"}')
        return energies

    def _aggregate_theta_graph(self):
        """Facteur commun a `plot_topology`/`plot_internal_topology` :
        agrege `|theta_masked()|` (n, n) en un graphe a `F + n_internal +
        out_channels` noeuds, en sommant les contributions des `T` pas
        temporels d'une meme feature sensorielle (F*T -> F). Retourne
        `(agg, labels, groups)` -- `groups[i] in {'sensory', 'internal',
        'label'}` pour chaque noeud agrege `i`."""
        model = self.model
        with torch.no_grad():
            theta = model.core.theta_masked().abs().cpu().numpy()

        F, T = model.in_dim, model.seq_len
        n_internal, n_label = model.n_internal, model.out_channels
        feat_names = list(self.features_name)[:F]

        n_agg = F + n_internal + n_label
        agg = np.zeros((n_agg, n_agg))
        sensory_group = np.repeat(np.arange(F), T)
        # lignes/colonnes sensorielles -> agregees par feature ; internes/label inchangees
        idx_map = np.concatenate([sensory_group, np.arange(F, F + n_internal + n_label)])
        for i_agg in range(n_agg):
            rows = np.where(idx_map == i_agg)[0]
            for j_agg in range(n_agg):
                cols = np.where(idx_map == j_agg)[0]
                agg[i_agg, j_agg] = theta[np.ix_(rows, cols)].sum()

        labels = feat_names + [f'internal_{i}' for i in range(n_internal)] + [f'label_{i}' for i in range(n_label)]
        groups = (['sensory'] * F) + (['internal'] * n_internal) + (['label'] * n_label)
        return agg, labels, groups

    def plot_topology(self, outname='topology', dir_output=None, top_edges=300, top_label_edges=15):
        """Visualisation networkx de la topologie apprise : un noeud par
        feature (les T pas temporels d'une meme feature sont agreges), un
        noeud par unite interne, un noeud par classe de sortie. Largeur
        d'arete proportionnelle a |theta| agrege -- ne garde que les
        `top_edges` aretes les plus fortes (toutes sources confondues) pour
        rester lisible.

        Un top global se fait dominer par deux categories numeriquement
        ecrasantes : les aretes label<->label (les noeuds de sortie
        s'influencent fortement entre eux, poids observes jusqu'a 2x plus
        forts que la meilleure arete feature->label) et la reconstruction
        sensorielle/interne (bien plus d'aretes possibles que de
        feature->label). Resultat verifie empiriquement : une feature peut
        avoir une arete directe solide vers un label (top 1% de toutes les
        aretes ->label) sans jamais apparaitre dans le top 300 global, alors
        que `feature_causal_ranking` la classe tres haut -- incoherence
        purement due au fait que ces deux fonctions ne regardent pas le meme
        sous-ensemble d'aretes. On reserve donc en plus, pour CHAQUE label
        SEPAREMENT (pas un budget global partage entre les 5), les
        `top_label_edges` aretes les plus fortes dont la SOURCE est une
        feature ou un noeud interne et la CIBLE ce label
        (`groups[i] != 'label'`, pour ne pas dupliquer les aretes
        label<->label deja capturees par le top global) -- unies (sans
        doublon) au top global, elles restent dessinees meme si elles
        n'auraient pas fait le top 300 seules.

        Un budget GLOBAL (partage entre les 5 labels) se refait dominer par
        les memes labels que le top 300 lui-meme : verifie empiriquement,
        avec un budget global de 15, seuls 9/20 features du top causal
        (`feature_causal_ranking`) obtenaient une arete visible, et les 24
        aretes reservees se repartissaient 12/9/3/0/0 entre label_0..4 --
        label_3 et label_4 n'avaient royalement RIEN. Par label, chacun des
        5 labels recoit desormais son propre budget de `top_label_edges`,
        quelle que soit la force relative des autres labels."""
        from matplotlib import pyplot as plt
        import networkx as nx

        agg, labels, groups = self._aggregate_theta_graph()
        n_agg = len(labels)
        color_map = {'sensory': '#0072B2', 'internal': '#999999', 'label': '#E69F00'}

        G = nx.DiGraph()
        for i, (lab, grp) in enumerate(zip(labels, groups)):
            G.add_node(i, label=lab, group=grp)

        edges_all = [(i, j, agg[i, j]) for i in range(n_agg) for j in range(n_agg) if i != j and agg[i, j] > 0]
        edges_all.sort(key=lambda e: -e[2])
        edges_global = edges_all[:top_edges]

        label_nodes = [j for j in range(n_agg) if groups[j] == 'label']
        edges_label = []
        for lbl in label_nodes:
            candidates = [(i, j, w) for i, j, w in edges_all if j == lbl and groups[i] != 'label']
            candidates.sort(key=lambda e: -e[2])
            edges_label.extend(candidates[:top_label_edges])

        seen = set()
        edges = []
        for i, j, w in edges_global + edges_label:
            if (i, j) not in seen:
                seen.add((i, j))
                edges.append((i, j, w))

        for i, j, w in edges:
            G.add_edge(i, j, weight=w)

        pos = nx.spring_layout(G, seed=0, k=1.5 / max(n_agg, 1) ** 0.5)
        node_colors = [color_map[groups[i]] for i in G.nodes()]
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        widths = 0.3 + 3.0 * (weights / weights.max()) if len(weights) > 0 else []

        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, edge_color='#555555', arrows=True, ax=ax)
        show_labels = {i: labels[i] for i in G.nodes() if groups[i] != 'internal'}
        nx.draw_networkx_labels(G, pos, labels=show_labels, font_size=7, ax=ax)
        ax.set_title(f'Topologie apprise (theta) -- {self.name}')
        ax.axis('off')
        fig.tight_layout()

        dir_output = self._analysis_dir(dir_output)
        fig.savefig(dir_output / f'{outname}.png', dpi=150)
        plt.close(fig)
        logger.info(f'[PCGraph] topology plot saved to {dir_output / f"{outname}.png"}')
        return G

    def plot_internal_topology(self, outname='internal_topology', dir_output=None,
                                top_edges_per_group=2, top_label_edges_per_node=1,
                                n_hops=3, reach_threshold=0.05, prune_top_k=None):
        """Sous-graphe de `plot_topology` restreint aux aretes dont la
        SOURCE est un noeud interne (`theta[interne, :]`, cf. convention
        `mask[j, i]` = connexion j -> i du module) -- `plot_topology` garde
        un top global d'aretes toutes sources confondues, ce qui peut faire
        disparaitre completement des noeuds internes du dessin (topologie
        'full' = structurellement tout est autorise, mais `theta` peut avoir
        converge vers ~0 sur leurs aretes pendant l'entrainement).

        Le top est calcule SEPAREMENT par groupe cible (sensoriel / interne /
        label) plutot que globalement : avec ~157 features et 128 noeuds
        internes contre seulement `out_channels` noeuds de label, un top
        global se fait quasi-systematiquement rafler par le groupe le plus
        nombreux (sensoriel) et n'affiche jamais aucune arete vers le label,
        meme quand une reconnexion existe. On garde donc au plus
        `top_edges_per_group` aretes sortantes par noeud interne vers le
        groupe sensoriel et vers le groupe interne, et au moins
        `top_label_edges_per_node` arete(s) vers le label -- meme si elle est
        beaucoup plus faible que ses meilleures aretes sensorielles/internes,
        elle reste dessinee (avec une epaisseur proportionnelle a son poids
        reel) pour que la comparaison visuelle reste honnete.

        Calcule aussi, via `causal_strength_mediated` (memes chemins
        multi-sauts que `feature_causal_ranking`, restreints a des
        intermediaires INTERNES uniquement -- pas de detour par un autre
        noeud interne agissant hors-representation ni par le sensoriel), la
        force de reconnexion de chaque noeud interne vers N'IMPORTE QUEL
        noeud de label (directe OU indirecte via d'autres noeuds internes,
        jusqu'a `n_hops` sauts) : `reach_to_label`. Les noeuds dont
        `reach_to_label <= reach_threshold * max(reach_to_label)` sont
        consideres orphelins (jamais connectes, meme indirectement, a un
        label) -- coloris en rouge sur le graphe, listes a part dans les
        logs et dans le DataFrame retourne.

        `prune_top_k` (cf. `PCGraphCore.causal_strength_matrix`) s'applique
        uniquement au calcul de `reach_to_label` (le calcul dense reste
        utilise par defaut, `None`) -- pas aux aretes dessinees, qui
        viennent de `_aggregate_theta_graph` (agregation brute, pas de
        multi-sauts) et de `top_edges_per_group`/`top_label_edges_per_node`
        uniquement. Attention, ce cap ne s'applique qu'aux aretes SORTANTES
        de chaque noeud interne (source) -- un noeud interne peut tres bien
        recevoir beaucoup plus d'aretes ENTRANTES que `top_edges_per_group`
        s'il est la cible privilegiee de nombreux AUTRES noeuds internes
        (un "hub" du graphe, cf. logs).

        Retourne `(G, df_reach)`."""
        from matplotlib import pyplot as plt
        import networkx as nx
        import pandas as pd

        model = self.model
        agg, labels, groups = self._aggregate_theta_graph()
        n_agg = len(labels)
        internal_idx = [i for i, g in enumerate(groups) if g == 'internal']

        with torch.no_grad():
            internal_to_label = model.core.causal_strength_mediated(
                model.internal_slice, model.internal_slice, model.label_slice,
                n_hops=n_hops, prune_top_k=prune_top_k,
            ).cpu().numpy()  # (n_internal, out_channels)
        reach = internal_to_label.max(axis=1)  # meilleure reconnexion, toutes classes confondues
        df_reach = pd.DataFrame({
            'internal_node': [f'internal_{i}' for i in range(model.n_internal)],
            'reach_to_label': reach,
        }).sort_values('reach_to_label', ascending=False).reset_index(drop=True)

        reach_cutoff = reach_threshold * reach.max() if reach.max() > 0 else 0.0
        orphan_set = set(df_reach.loc[df_reach['reach_to_label'] <= reach_cutoff, 'internal_node'])
        logger.info(
            f'[PCGraph] {len(orphan_set)}/{model.n_internal} noeuds internes sous le seuil de reconnexion '
            f'aux labels ({reach_threshold:.0%} du max, n_hops={n_hops}).'
        )

        G = nx.DiGraph()
        for i in internal_idx:
            G.add_node(i, label=labels[i], group=groups[i])

        edges = []
        top_k_per_group = {'sensory': top_edges_per_group, 'internal': top_edges_per_group,
                            'label': top_label_edges_per_node}
        for i in internal_idx:
            for grp, k in top_k_per_group.items():
                candidates = [(i, j, agg[i, j]) for j in range(n_agg)
                              if j != i and groups[j] == grp and agg[i, j] > 0]
                candidates.sort(key=lambda e: -e[2])
                edges.extend(candidates[:k])

        for i, j, w in edges:
            if j not in G:
                G.add_node(j, label=labels[j], group=groups[j])
            G.add_edge(i, j, weight=w)

        internal_in_degree = [G.in_degree(i) for i in internal_idx]
        max_in_degree = max(internal_in_degree) if internal_in_degree else 0
        logger.info(
            f'[PCGraph] degre sortant plafonne a {2 * top_edges_per_group + top_label_edges_per_node} par noeud '
            f'interne (source) ; degre ENTRANT non plafonne (un noeud peut etre la cible privilegiee de '
            f'plusieurs autres noeuds internes) -- max observe : {max_in_degree}.'
        )

        color_map = {'sensory': '#0072B2', 'internal': '#999999', 'label': '#E69F00'}
        node_colors = [
            '#D55E00' if labels[i] in orphan_set else color_map[groups[i]]
            for i in G.nodes()
        ]

        pos = nx.spring_layout(G, seed=0, k=1.5 / max(G.number_of_nodes(), 1) ** 0.5)
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        widths = 0.3 + 3.0 * (weights / weights.max()) if len(weights) > 0 else []

        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, edge_color='#555555', arrows=True, ax=ax)
        show_labels = {i: labels[i] for i in G.nodes() if groups[i] != 'internal' or labels[i] in orphan_set}
        nx.draw_networkx_labels(G, pos, labels=show_labels, font_size=7, ax=ax)
        ax.set_title(
            f'Connexions sortantes des noeuds internes -- {self.name}\n'
            f'(rouge = orphelin, reconnexion label <= {reach_threshold:.0%} du max, {n_hops} sauts)'
        )
        ax.axis('off')
        fig.tight_layout()

        dir_output = self._analysis_dir(dir_output)
        fig.savefig(dir_output / f'{outname}.png', dpi=150)
        plt.close(fig)
        df_reach.to_csv(dir_output / f'{outname}_reach.csv', index=False)
        logger.info(f'[PCGraph] internal topology plot saved to {dir_output / f"{outname}.png"}')
        return G, df_reach

    def feature_causal_ranking(self, n_hops=3, top_k=20, outname='causal_ranking', dir_output=None,
                                prune_top_k=None):
        """Classement des features par force d'influence cumulee (jusqu'a
        `n_hops` sauts dans le graphe appris) sur chaque noeud de label --
        lecture directe des poids `theta` (pas de nouvelle relaxation),
        analogue interpretable au SHAP mais propre a la structure du graphe
        PC. Sauve un CSV + un barplot par classe de sortie, suivant la
        convention de sortie de run_shap_analysis.py/run_causal_analysis.py.

        Utilise `causal_strength_mediated` (pas `causal_strength_matrix`) :
        les chemins comptes sont restreints a sensoriel -> interne -> ... ->
        interne -> label, tous les noeuds INTERMEDIAIRES devant etre des
        noeuds internes. `causal_strength_matrix` autoriserait n'importe
        quel intermediaire, y compris un detour par un AUTRE noeud
        sensoriel (ex. Hêtre_mean -> Hêtre_max, une arete de reconstruction
        a 0.48 qui n'a rien a voir avec le label) -- ce qui mesurerait alors
        une correlation feature<->feature plutot qu'une influence mediee
        par la representation interne du reseau. Verifie empiriquement :
        seuls 4/15 features du top se recouvrent entre les deux versions.

        `prune_top_k` (cf. `PCGraphCore.causal_strength_matrix`) : si None
        (defaut), calcul dense/exact sur toutes les aretes -- avec une
        topologie 'full', la tres grande majorite des aretes sortantes de
        chaque noeud sont proches du bruit d'initialisation et peuvent, une
        fois sommees sur `n_hops` sauts, pesez plus lourd au total que les
        quelques aretes reellement significatives (verifie empiriquement :
        pour un noeud interne donne, les 275 aretes les plus faibles pesent
        a elles seules plus que les 15 plus fortes). Fournir `prune_top_k`
        (ex. 15) ne garde que les aretes sortantes les plus fortes de chaque
        noeud avant d'accumuler les sauts, pour un classement base
        uniquement sur le squelette structurellement significatif."""
        import pandas as pd
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        model = self.model
        sensory_rows = model.core.causal_strength_mediated(
            model.sensory_slice, model.internal_slice, model.label_slice,
            n_hops=n_hops, prune_top_k=prune_top_k,
        ).cpu().numpy()  # (F*T, out_channels)

        F, T = model.in_dim, model.seq_len
        feat_names = list(self.features_name)[:F]
        sensory_rows = sensory_rows.reshape(F, T, -1).sum(axis=1)     # agrege les T pas temporels -> (F, out_channels)

        dir_output = self._analysis_dir(dir_output)
        label_cols = [f'label_{i}' for i in range(sensory_rows.shape[1])]
        df = pd.DataFrame(sensory_rows, index=feat_names, columns=label_cols)
        df['mean_strength'] = df.mean(axis=1)
        df = df.sort_values('mean_strength', ascending=False)
        df.to_csv(dir_output / f'{outname}.csv')

        # Heatmap features x labels (pas juste la moyenne) : cette somme est
        # TOUJOURS >= 0 (`.abs()` dans causal_strength_mediated) -- job
        # "magnitude", rampe sequentielle une seule teinte (bleu, cf. skill
        # dataviz), pas de rouge/divergent puisqu'il n'y a pas de signe ici.
        top = df.head(top_k).iloc[::-1]
        data = top[label_cols].to_numpy()
        seq_cmap = LinearSegmentedColormap.from_list('seq_blue', ['#fcfcfb', '#6da7ec', '#0d366b'])
        fig, ax = plt.subplots(figsize=(1.3 * len(label_cols) + 3, max(4, 0.35 * len(top))))
        im = ax.imshow(data, cmap=seq_cmap, vmin=0, aspect='auto')
        ax.set_xticks(range(len(label_cols)))
        ax.set_xticklabels(label_cols)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index)
        vmax = data.max() if data.size else 1.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'white' if data[i, j] > vmax * 0.6 else '#0b0b0b'
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=7, color=color)
        fig.colorbar(im, ax=ax, label=f'force causale cumulee ({n_hops} sauts, abs, sommee sur les chemins)')
        ax.set_title(f'Top {top_k} features les plus causales, par label -- {self.name}')
        fig.tight_layout()
        fig.savefig(dir_output / f'{outname}.png', dpi=150)
        plt.close(fig)
        logger.info(f'[PCGraph] causal ranking saved to {dir_output / f"{outname}.csv"} / .png')
        return df

    def _reconstruct_chain(self, model, theta, feat_idx, t_idx, mids, label_idx):
        """Facteur commun a `plot_strongest_paths`/`plot_top_explicative_features` :
        reconstruit la chaine lisible (noms) et les poids signes de CHAQUE
        saut a partir des indices bruts (`mids` en indices locaux a
        `internal_slice`, ou None/liste vide pour un chemin direct).
        Retourne `(feat_label, chain_labels, hop_values)`."""
        F, T = model.in_dim, model.seq_len
        feat_names = list(self.features_name)[:F]
        feat_label = feat_names[feat_idx] if T == 1 else f'{feat_names[feat_idx]}_t{t_idx}'
        mids = mids or []
        chain_labels = [feat_label] + [f'internal_{m}' for m in mids] + [f'label_{label_idx}']

        global_idx = [model.sensory_slice.start + feat_idx * T + t_idx]
        global_idx += [model.internal_slice.start + m for m in mids]
        global_idx += [model.label_slice.start + label_idx]
        hop_values = [theta[global_idx[k], global_idx[k + 1]].item() for k in range(len(global_idx) - 1)]
        return feat_label, chain_labels, hop_values

    def plot_strongest_paths(self, outname='strongest_paths', dir_output=None, max_hops=3):
        """Pour CHAQUE label separement, LE chemin individuel le plus fort
        (feature -> [noeud(s) interne(s)] -> label) parmi tous les points de
        depart possibles (n'importe quelle feature sensorielle) et toutes
        les longueurs de 1 a `max_hops` sauts -- avec le SIGNE reel de
        chaque saut, pour lire le "sens" de la combinaison (est-ce que cette
        feature, via ce chemin, pousse le label vers le haut ou vers le
        bas ?).

        Different de `feature_causal_ranking`, qui SOMME en valeur absolue
        sur TOUS les chemins possibles (le signe est perdu, et mille chemins
        faibles peuvent peser autant qu'un seul chemin fort) : ici on
        cherche le meilleur chemin UNIQUE (`PCGraphCore.strongest_paths`),
        seule une explication lisible et son sens reel nous interessent.

        Dessine, par label, la chaine complete avec le poids signe de
        CHAQUE saut individuel (vert = positif, rouge = negatif) -- pas
        seulement le signe du produit final -- pour voir si un signe
        negatif quelque part dans la chaine s'annule ou se cumule. Sauve un
        CSV recapitulatif (une ligne par label). Retourne le DataFrame."""
        from matplotlib import pyplot as plt
        import pandas as pd
        import numpy as np

        model = self.model
        out_channels = model.out_channels

        results = model.core.strongest_paths(
            model.sensory_slice, model.internal_slice, model.label_slice, max_hops=max_hops
        )
        with torch.no_grad():
            theta = model.core.theta_masked()  # signe reel, indices GLOBAUX

        rows = []
        dir_output = self._analysis_dir(dir_output)
        fig, axes = plt.subplots(out_channels, 1, figsize=(9, 2.3 * out_channels))
        axes = np.atleast_1d(axes)

        for label_idx in range(out_channels):
            best_value, best_hops, best_start, best_mids = None, None, None, None
            for hops, (value, mids) in enumerate(results, start=1):
                col = value[:, label_idx]
                start_local = torch.argmax(col.abs()).item()
                v = col[start_local].item()
                if best_value is None or abs(v) > abs(best_value):
                    best_value = v
                    best_hops = hops
                    best_start = start_local
                    best_mids = None if mids is None else mids[start_local, label_idx].tolist()

            feat_idx, t_idx = divmod(best_start, model.seq_len)
            feat_label, chain_labels, hop_values = self._reconstruct_chain(
                model, theta, feat_idx, t_idx, best_mids, label_idx
            )

            rows.append({
                'label': f'label_{label_idx}',
                'feature': feat_label,
                'hops': best_hops,
                'chain': ' -> '.join(chain_labels),
                'hop_weights': [round(v, 4) for v in hop_values],
                'net_product': best_value,
                'sens': 'positif (augmente le risque)' if best_value > 0 else 'negatif (diminue le risque)',
            })

            ax = axes[label_idx]
            n_nodes = len(chain_labels)
            xs = np.linspace(0.08, 0.92, n_nodes)
            node_colors = ['#0072B2'] + ['#999999'] * (n_nodes - 2) + ['#E69F00']
            ax.scatter(xs, [0] * n_nodes, s=350, color=node_colors, zorder=3)
            for k in range(n_nodes):
                ax.text(xs[k], 0.14, chain_labels[k], ha='center', fontsize=8)
            for k in range(n_nodes - 1):
                color = '#009E73' if hop_values[k] > 0 else '#D55E00'
                ax.annotate('', xy=(xs[k + 1], 0), xytext=(xs[k], 0),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.85))
                ax.text((xs[k] + xs[k + 1]) / 2, -0.13, f'{hop_values[k]:+.3f}',
                        ha='center', fontsize=8, color=color)
            net_color = '#009E73' if best_value > 0 else '#D55E00'
            ax.text(1.0, 0, f'net={best_value:+.4f}\n({best_hops} saut{"s" if best_hops > 1 else ""})',
                    ha='left', va='center', fontsize=8, color=net_color, transform=ax.transData)
            ax.set_xlim(-0.02, 1.25)
            ax.set_ylim(-0.35, 0.35)
            ax.set_title(f'label_{label_idx}', fontsize=10, loc='left')
            ax.axis('off')

        fig.suptitle(f'Chemin causal individuel le plus fort par label -- {self.name}')
        fig.tight_layout()
        fig.savefig(dir_output / f'{outname}.png', dpi=150)
        plt.close(fig)

        df = pd.DataFrame(rows)
        df.to_csv(dir_output / f'{outname}.csv', index=False)
        logger.info(f'[PCGraph] strongest paths saved to {dir_output / f"{outname}.png"} / .csv')
        return df

    def plot_top_explicative_features(self, outname='top_explicative_features', dir_output=None,
                                       top_k=20, max_hops=3):
        """Pour CHAQUE feature sensorielle et CHAQUE label separement (pas de
        collapse sur le label comme dans une version anterieure de cette
        methode) : la valeur signee du meilleur chemin individuel (parmi
        toutes les longueurs de 1 a `max_hops` sauts et tous les pas
        temporels) reliant cette feature a ce label. Complementaire du
        classement par SOMME (perdant le signe) de `feature_causal_ranking`.

        Les `top_k` features sont choisies par la plus forte valeur absolue
        obtenue pour N'IMPORTE LEQUEL de leurs 5 labels, mais la figure
        affiche les 5 valeurs (une par label) pour chacune -- pas seulement
        celle du label gagnant, pour ne pas cacher qu'une feature peut avoir
        un effet net oppose ou nul sur les autres classes.

        Sauve un CSV complet (les 157 features x 5 labels, pas seulement le
        top_k) et une heatmap divergente (bleu = positif, rouge = negatif,
        gris = proche de zero, cf. skill dataviz) des `top_k`. Retourne le
        DataFrame complet, trie par force absolue maximale decroissante."""
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import pandas as pd
        import numpy as np

        model = self.model
        F, T = model.in_dim, model.seq_len
        feat_names = list(self.features_name)[:F]
        out_channels = model.out_channels

        results = model.core.strongest_paths(
            model.sensory_slice, model.internal_slice, model.label_slice, max_hops=max_hops
        )

        # Meilleure valeur signee par (feature, label), independamment pour
        # CHAQUE label -- on ne prend le max que sur les hops et les pas
        # temporels, jamais sur l'axe label.
        per_feat_label = np.zeros((F, out_channels))
        for feat_idx in range(F):
            for label_idx in range(out_channels):
                best_value = None
                for t in range(T):
                    s = feat_idx * T + t
                    for value, _ in results:
                        v = value[s, label_idx].item()
                        if best_value is None or abs(v) > abs(best_value):
                            best_value = v
                per_feat_label[feat_idx, label_idx] = best_value

        label_cols = [f'label_{e}' for e in range(out_channels)]
        dir_output = self._analysis_dir(dir_output)
        df = pd.DataFrame(per_feat_label, index=feat_names, columns=label_cols)
        df['max_abs'] = np.abs(per_feat_label).max(axis=1)
        df = df.sort_values('max_abs', ascending=False)
        df.to_csv(dir_output / f'{outname}.csv')

        top = df.head(top_k).iloc[::-1]
        data = top[label_cols].to_numpy()
        vmax = np.abs(data).max() if data.size else 1.0
        div_cmap = LinearSegmentedColormap.from_list('diverging_blue_red', ['#e34948', '#f0efec', '#2a78d6'])
        fig, ax = plt.subplots(figsize=(1.3 * len(label_cols) + 3, max(4, 0.35 * len(top))))
        im = ax.imshow(data, cmap=div_cmap, vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(label_cols)))
        ax.set_xticklabels(label_cols)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'white' if abs(data[i, j]) > vmax * 0.6 else '#0b0b0b'
                ax.text(j, i, f'{data[i, j]:+.3f}', ha='center', va='center', fontsize=7, color=color)
        fig.colorbar(im, ax=ax, label='valeur signee du chemin le plus fort (par label)')
        ax.set_title(f'Top {top_k} features les plus explicatives, par label -- {self.name}')
        fig.tight_layout()
        fig.savefig(dir_output / f'{outname}.png', dpi=150)
        plt.close(fig)
        logger.info(f'[PCGraph] top explicative features (par label) saved to {dir_output / f"{outname}.png"} / .csv')
        return df


class PCGraphGenTraining(PCGraphTraining):
    """Variante "generative semi-supervisee" de `PCGraphTraining`, pour les
    cibles bruitees du projet (un risque note 0 peut en realite correspondre
    a un risque eleve non detecte/non reporte). `PCGraphTraining.launch_batch`
    clampe le label sur *tous* les echantillons du batch (Algorithme 1
    litteral de l'article) -- les poids sont donc pousses a expliquer
    exactement chaque valeur de target, bruit inclus.

    Ici, seul un sous-ensemble stochastique des echantillons (probabilite
    `label_clamp_prob`, defaut `_DEFAULT_LABEL_CLAMP_PROB`) voit son label
    clampe a chaque batch ; pour les autres, le label reste libre pendant
    toute la relaxation (comme a l'inference, `PCGraphModel.forward`) et son
    energie au point fixe ne compare donc jamais ce noeud a une cible
    potentiellement fausse. Le graphe apprend ainsi en partie *sans regarder
    le label* (reconstruction sensorielle + noeuds internes, cf.
    `PCGraphModel.training_energy_semi_supervised`), ce qui reduit la
    dependance du modele a l'exactitude ponctuelle de chaque target -- un
    "dropout de supervision" plutot qu'un clamp dur systematique.

    Meme architecture, mêmes hyperparametres de relaxation (`n_internal`,
    `t_train`, `t_query`, `lr_x`, `topology`, ...) que `PCGraphTraining` :
    `make_model`, `get_optimizer`, `get_loss` et les outils de
    visualisation/interpretabilite sont herites tels quels, seul
    `launch_batch` change de signal d'apprentissage. `label_clamp_prob` se
    regle via le JSON ("params": {"label_clamp_prob": 0.7, ...})."""

    def launch_batch(self, data, criterion, batch_type, do_update):
        if data[0].shape[0] == 1:
            return 0, 0

        inputs_horizon, y_clamp, weights = self._prepare_energy_inputs(data)

        label_clamp_prob = _DEFAULT_LABEL_CLAMP_PROB
        if self.model_params is not None:
            label_clamp_prob = self.model_params.get('label_clamp_prob', label_clamp_prob)

        energy_per_sample, label_clamped = self.model.training_energy_semi_supervised(
            inputs_horizon, y_clamp, label_clamp_prob=label_clamp_prob
        )
        w = weights.view(-1).clamp_min(0)
        total_loss = (energy_per_sample * w).sum() / w.sum().clamp_min(1e-8)

        return total_loss, {
            'total_loss': total_loss,
            'supervised_fraction': label_clamped.float().mean(),
        }


# ======================================================================
# Variante CORN : sortie ordinale + objectif distributionnel
# ======================================================================

_DEFAULT_CORN_LOGIT = 3.0      # amplitude des logits conditionnels clampes
_DEFAULT_LAMBDA_ORD = 1.0
_DEFAULT_LAMBDA_COV = 1.0


class PCGraphCornTraining(PCGraphTraining):
    """PC-graph a sortie ORDINALE (CORN) et objectif DISTRIBUTIONNEL.

    Deux changements par rapport a `PCGraphTraining`, l'un de representation,
    l'autre d'objectif.

    1. REPRESENTATION -- les noeuds de label portent `[s, logits conditionnels]`
       au lieu d'un one-hot. `s` est le risque scalaire ("combien de sinistres
       attendus"), les `K-1` suivants sont les logits CORN dont
       `corn_class_probs` reconstruit P(y=k). L'ordinalite devient
       STRUCTURELLE : P(y>k) est un produit cumule de facteurs <= 1, donc
       forcement decroissant. Mesure sur le one-hot precedent, les mu par
       classe ne s'ordonnaient pas (0.62, 0.65, 0.65, 0.64, 0.63 -- non
       monotone), et un softmax applique a des noeuds entraines a valoir 0/1
       plafonnait les probabilites vers l'uniforme (~0.2 chacune).

    2. OBJECTIF -- le risque existe independamment de sa realisation : un jour
       a risque eleve ou rien ne s'allume reste un jour a risque eleve. Viser
       `pred == y_true` par echantillon est donc conceptuellement faux. On
       remplace cette exigence par deux contraintes de POPULATION :

         - ordinalite : mu(classe predite) doit croitre. Ne dit jamais a
           quelle classe appartient un jour donne, seulement que la moyenne
           du niveau k+1 doit depasser celle du niveau k. C'est l'AGREGATION
           qui rend le critere tolerant au bruit de realisation.
         - couverture : la distribution des classes predites doit rester
           proche d'une reference `q` (distance ordinale sur les cumulees,
           comme `_coverage_loss` de ordinal_loss_2). Sans elle l'objectif
           s'effondre : `score_k` ne penalise pas l'abstention (ponderation
           par couverture desactivee, `min_k=0`), et un modele n'emettant que
           2 niveaux sur 5 obtient un meilleur score qu'un modele complet.

    IMPORTANT -- ce que ca preserve du predictive coding, et ce que ca change.
    Les deux termes n'entrent PAS dans `relax` : la relaxation reste locale et
    par echantillon, et l'inference est inchangee (aucune dependance au lot).
    Ils s'ajoutent uniquement a l'objectif de mise a jour des poids, ou ils
    atteignent `theta` par le meme chemin que l'energie
    (`core.predict(x_conv)` -> logits -> p). Seule propriete perdue :
    l'objectif d'apprentissage n'est plus decomposable par echantillon -- ce
    qui est inevitable, "le risque est ordinal" etant une propriete de
    population et non d'un jour isole.

    Repartition du travail entre les deux signaux :
      energie PC          -> aretes sensoriel <-> interne (reconstruction)
      ordinalite+couverture -> aretes interne -> label (lecture du risque)
    Cette seconde famille ne recevait plus rien des que les labels etaient
    libres (leur terme d'energie propre tend vers zero au point fixe) : les
    termes distributionnels comblent exactement ce vide.
    """

    def make_model(self, graph, custom_model_params):
        params = dict(custom_model_params) if custom_model_params else {}
        params.setdefault('label_mode', 'corn')
        return super().make_model(graph, params)

    # ---------------- cibles clampees ----------------

    def _prepare_energy_inputs(self, data):
        """Cible clampee au format CORN : `[s, logits conditionnels]`.

        `s` recoit `sqrt(y)/sigma` -- racine parce que `y` brut monte a 8 pour
        un ecart-type de 0.61, soit une etendue de 13 sigma qui ecraserait le
        terme sensoriel (features z-scorees, ~[-3, 3]) et saturerait le
        `Hardtanh` de `predict` ; `sqrt(y)/sigma` reste dans [0, 4.6]. C'est
        aussi la convention du projet (`burnedareaRoot`).

        Les logits conditionnels recoivent `+c` pour les seuils franchis et
        `-c` sinon (encodage thermometre CORN de la classe vraie). On clampe
        des logits FRANCS et non 0/1 : un sigmoid sur des valeurs resserrees
        ramenerait tous les facteurs vers 0.5, reproduisant sous une autre
        forme le defaut du softmax."""
        inputs, labels, _ = data
        if self.horizon != 0:
            raise NotImplementedError(
                'PCGraphCornTraining : horizon > 0 non supporte (cf. docstring du module).'
            )

        target, weights = self.compute_weights_and_target(
            labels, -1, ids_columns, False, None, -1 - self.horizon
        )
        inputs_horizon = self.compute_inputs(inputs, -1 - self.horizon, 'current')

        K = self.model.out_channels
        k = target.long().view(-1)
        c = float((self.model_params or {}).get('corn_logit_scale', _DEFAULT_CORN_LOGIT))

        thresholds = torch.arange(K - 1, device=k.device).view(1, -1)
        corn = torch.where(k.view(-1, 1) > thresholds, c, -c).float()

        y_idx = len(ids_columns) + targets_columns.index(self.scoring_target_column())
        y_raw = labels[:, y_idx, -1].view(-1, 1).float().clamp_min(0)
        sigma = getattr(self.scoring, 'sigma', None) or 1.0
        s = torch.sqrt(y_raw) / sigma

        return inputs_horizon, torch.cat([s, corn], dim=1), weights

    # ---------------- termes distributionnels ----------------

    def _target_distribution(self, device):
        """Distribution de reference `q` pour la couverture, en cache.

        Par defaut la distribution empirique des classes du train, decalee
        vers le haut par `coverage_risk_shift` : le risque existant sans
        realisation, il doit y avoir PLUS de jours a risque eleve que de jours
        ayant effectivement brule fort. Ce decalage se choisit -- les donnees
        n'observent que les realisations et ne peuvent pas l'apprendre."""
        if getattr(self, '_q_cache', None) is None:
            K = self.model.out_channels
            counts = torch.zeros(K)
            for _, labels, _ in self.train_loader:
                t, _ = self.compute_weights_and_target(labels, -1, ids_columns, False, None, -1)
                counts += torch.bincount(t.view(-1).long().cpu(), minlength=K).float()
            q = counts / counts.sum().clamp_min(1.0)

            shift = float((self.model_params or {}).get('coverage_risk_shift', 0.0))
            if shift != 0.0:
                # deplace de la masse vers les classes hautes, en gardant une
                # distribution valide
                w = torch.exp(shift * torch.arange(K).float())
                q = q * w
                q = q / q.sum()
            self._q_cache = q
            logger.info(f'[PCGraph] distribution de couverture q (shift={shift}) : '
                        + ' '.join(f'{v:.4f}' for v in q.tolist()))
        return self._q_cache.to(device)

    def _distribution_terms(self, p, labels):
        """Ordinalite + couverture, calcules sur le LOT.

        Retourne `(L_ord, L_cov, diagnostics)`. `mu` utilise une assignation
        DOUCE (chaque jour contribue a toutes les classes, pondere par p) :
        en assignation dure la classe 4, a 0.5% de prevalence, n'apporterait
        que ~0.3 echantillon par lot de 64 et mu_4 serait inestimable."""
        eps = 1e-8

        y_idx = len(ids_columns) + targets_columns.index(self.scoring_target_column())
        y = labels[:, y_idx, -1].view(-1).float()
        d = labels[:, ids_columns.index('departement'), -1].view(-1).long()

        # Centrage par departement : analogue differentiable des effets fixes
        # de `fit_spline_mu` (Y = alpha[niveau] + FE_zone + FE_date), qui
        # neutralise les differences d'echelle entre departements SANS
        # fragmenter les donnees.
        y = y.clone()
        for dept in torch.unique(d):
            msk = d == dept
            y[msk] = y[msk] - y[msk].mean()
        sigma = getattr(self.scoring, 'sigma', None) or 1.0
        y = y / sigma

        w = p / p.sum(dim=0, keepdim=True).clamp_min(eps)
        mu = (w * y.view(-1, 1)).sum(dim=0)                      # (K,)

        # softplus(-delta) : cout nul tant que mu croit, ne mord que sur les
        # INVERSIONS. On n'impose jamais une valeur, seulement un ordre.
        gaps = mu[1:] - mu[:-1]
        l_ord = torch.nn.functional.softplus(-gaps).sum()

        p_bar = p.mean(dim=0)
        q = self._target_distribution(p.device)
        l_cov = (torch.cumsum(p_bar, 0) - torch.cumsum(q, 0)).pow(2).mean()

        n_eff = (p.sum(0) ** 2 / p.pow(2).sum(0).clamp_min(eps))
        return l_ord, l_cov, {'mu': mu.detach(), 'p_bar': p_bar.detach(),
                              'n_eff': n_eff.detach(), 'gaps': gaps.detach()}

    def _lambdas(self):
        mp = self.model_params or {}
        return (float(mp.get('lambda_ordinality', _DEFAULT_LAMBDA_ORD)),
                float(mp.get('lambda_coverage', _DEFAULT_LAMBDA_COV)))

    def _energy_and_terms(self, data, energy_fn):
        """Facteur commun aux deux `launch_batch` : energie PC (via `energy_fn`,
        supervisee ou semi-supervisee) + les deux termes distributionnels."""
        inputs_horizon, y_clamp, weights = self._prepare_energy_inputs(data)
        energy_per_sample, extra = energy_fn(inputs_horizon, y_clamp)

        w = weights.view(-1).clamp_min(0)
        e_pc = (energy_per_sample * w).sum() / w.sum().clamp_min(1e-8)

        # Lecture des labels a labels LIBRES (regime d'inference) : c'est la
        # distribution que le modele produira reellement, donc celle qu'il faut
        # contraindre -- pas celle obtenue avec les labels clampes.
        probs, _, _ = self.model(inputs_horizon)
        l_ord, l_cov, diag = self._distribution_terms(probs, data[1])

        lam_ord, lam_cov = self._lambdas()
        total = e_pc + lam_ord * l_ord + lam_cov * l_cov

        logs = {'total_loss': total, 'energy_pc': e_pc.detach(),
                'l_ordinality': l_ord.detach(), 'l_coverage': l_cov.detach()}
        logs.update(extra)
        for i, v in enumerate(diag['n_eff'].tolist()):
            logs[f'n_eff_{i}'] = v
        for i, v in enumerate(diag['mu'].tolist()):
            logs[f'mu_{i}'] = v
        return total, logs

    def launch_batch(self, data, criterion, batch_type, do_update):
        if data[0].shape[0] == 1:
            return 0, 0
        return self._energy_and_terms(
            data, lambda x, y: (self.model.training_energy(x, y), {})
        )


class PCGraphGenCornTraining(PCGraphCornTraining):
    """Variante semi-supervisee de `PCGraphCornTraining` : le clamp du label
    est stochastique (`label_clamp_prob`), exactement comme
    `PCGraphGenTraining`. `label_clamp_prob = 0` donne le regime cible --
    aucun label jamais clampe, `s` et les logits conditionnels entierement
    libres, et les deux termes distributionnels comme unique signal reliant
    les features au risque."""

    def launch_batch(self, data, criterion, batch_type, do_update):
        if data[0].shape[0] == 1:
            return 0, 0

        prob = _DEFAULT_LABEL_CLAMP_PROB
        if self.model_params is not None:
            prob = self.model_params.get('label_clamp_prob', prob)

        def energy_fn(x, y):
            e, clamped = self.model.training_energy_semi_supervised(x, y, label_clamp_prob=prob)
            return e, {'supervised_fraction': clamped.float().mean().detach()}

        return self._energy_and_terms(data, energy_fn)


_DEFAULT_CLM_TAU = 0.3


class PCGraphClmTraining(PCGraphCornTraining):
    """PC-graph a sortie ordinale par **modele a lien cumulatif (CLM)**.

    Remplace la sortie CORN de `PCGraphCornTraining`, structurellement
    inutilisable ici : dans CORN, `P(y=k) = F_{k-1} - F_k` avec `F` un produit
    cumule, donc `p_0 > p_1 > ... > p_{K-2}` par construction et l'argmax ne
    peut valoir que `0` ou `K-1`. Constate en entrainement reel -- distribution
    predite `[1483, 0, 0, 1536, 2821]`, classes 1 et 2 vides, loss de couverture
    bloquee a son maximum sans qu'aucun reglage de lambda n'y change rien.

    Le CLM applique au contraire des seuils ORDONNES au MEME score scalaire :
    `p_k = sigmoid((theta_k - s)/tau) - sigmoid((theta_{k-1} - s)/tau)` est une
    bosse, et l'argmax selectionne l'intervalle contenant `s`.

    Un seul noeud de label (`s`), et des seuils **fixes** : les frontieres de
    classe sont deja connues (le kmeans du pipeline discretise un comptage
    entier, d'ou des frontieres aux demi-entiers). Rien a apprendre, donc pas
    de gradient manquant -- l'objection qui condamnait la version a seuils
    appris ne s'applique pas.

    `clm_tau` n'est PAS cosmetique : la sigmoide standard transitionne sur
    ~4 unites alors que les seuils sont espaces de ~0.6-0.85 en unites
    `sqrt(y)/sigma`. A tau=1 les bosses se recouvrent et seules 2 classes sont
    atteignables ; a tau <= 0.5 les 5 le sont.
    """

    def make_model(self, graph, custom_model_params):
        params = dict(custom_model_params) if custom_model_params else {}
        params['label_mode'] = 'clm'
        params.setdefault('clm_tau', _DEFAULT_CLM_TAU)
        model, params = PCGraphTraining.make_model(self, graph, params)
        model.set_clm_thresholds(self._clm_thresholds(model.out_channels))
        return model, params

    def _clm_thresholds(self, n_classes):
        """Seuils du CLM, en unites STANDARDISEES (`s` est centre-reduit avant
        d'y etre compare, cf. PCGraphModel.class_probs).

        Places aux quantiles de la distribution cible : si l'on veut 53% de la
        masse en classe 0, le premier seuil est le quantile 0.53 d'une normale
        standard. La couverture est ainsi presque satisfaite des
        l'initialisation, ce qui laisse l'optimisation se consacrer a
        l'ordinalite plutot qu'a rattraper une echelle.

        Remplace les seuils absolus en `sqrt(y)/sigma` : ceux-ci supposaient que
        le graphe produirait `s` dans [0.8, 3.3], ce que rien ne garantit une
        fois le clampage retire -- d'ou l'effondrement en classe 0 observe.
        Seul l'ORDRE de `s` importe pour le scoring monotone ; son echelle est
        donc un artefact a neutraliser, pas une grandeur a apprendre."""
        if getattr(self, '_clm_thr_cache', None) is not None:
            return self._clm_thr_cache

        counts = torch.zeros(n_classes)
        for _, labels, _ in self.train_loader:
            t, _ = self.compute_weights_and_target(labels, -1, ids_columns, False, None, -1)
            counts += torch.bincount(t.view(-1).long().cpu(), minlength=n_classes).float()
        q = counts / counts.sum().clamp_min(1.0)

        # quantile normal standard : Phi^-1(p) = sqrt(2) * erfinv(2p - 1)
        cum = torch.cumsum(q, 0)[:-1].clamp(1e-4, 1 - 1e-4)
        thr = (torch.erfinv(2 * cum - 1) * (2 ** 0.5)).tolist()
        for i in range(1, len(thr)):
            thr[i] = max(thr[i], thr[i - 1] + 1e-3)

        self._clm_thr_cache = thr
        logger.info('[PCGraph] seuils CLM (unites standardisees, quantiles de q) : '
                    + ' '.join(f'{v:.3f}' for v in thr))
        return thr

    def _prepare_energy_inputs(self, data):
        """Cible clampee : uniquement le scalaire `s = sqrt(y)/sigma`.

        Racine carree parce que `y` brut monte a 8 pour un ecart-type de 0.61,
        soit ~13 sigma -- hors d'echelle face aux features z-scorees (~[-3, 3]),
        ce qui saturerait le `Hardtanh` de `predict` et ecraserait le terme
        sensoriel dans l'energie. `sqrt(y)/sigma` reste dans [0, 4.6]. C'est
        aussi la convention du projet (`burnedareaRoot`)."""
        inputs, labels, _ = data
        if self.horizon != 0:
            raise NotImplementedError('PCGraphClmTraining : horizon > 0 non supporte.')

        _, weights = self.compute_weights_and_target(
            labels, -1, ids_columns, False, None, -1 - self.horizon
        )
        inputs_horizon = self.compute_inputs(inputs, -1 - self.horizon, 'current')

        y_idx = len(ids_columns) + targets_columns.index(self.scoring_target_column())
        y_raw = labels[:, y_idx, -1].view(-1, 1).float().clamp_min(0)
        sigma = getattr(self.scoring, 'sigma', None) or 1.0
        return inputs_horizon, torch.sqrt(y_raw) / sigma, weights

    def launch_batch(self, data, criterion, batch_type, do_update):
        if data[0].shape[0] == 1:
            return 0, 0
        return self._energy_and_terms(
            data, lambda x, y: (self.model.training_energy(x, y), {})
        )


class PCGraphGenClmTraining(PCGraphClmTraining):
    """Variante semi-supervisee de `PCGraphClmTraining` : clamp stochastique de
    `s` (`label_clamp_prob`). A 0, `s` n'est jamais clampe -- regime cible, ou
    seuls les termes distributionnels relient les features au risque."""

    def launch_batch(self, data, criterion, batch_type, do_update):
        if data[0].shape[0] == 1:
            return 0, 0

        prob = _DEFAULT_LABEL_CLAMP_PROB
        if self.model_params is not None:
            prob = self.model_params.get('label_clamp_prob', prob)

        def energy_fn(x, y):
            e, clamped = self.model.training_energy_semi_supervised(x, y, label_clamp_prob=prob)
            return e, {'supervised_fraction': clamped.float().mean().detach()}

        return self._energy_and_terms(data, energy_fn)
