import random
from functools import reduce


class GridSearch(object):

    def __init__(self, base_config, combinations, run_name=None, exclude_old_runs=False, exclude_combs=None):
        self.base_config = base_config
        self.combinations = combinations

        self._excluded_combs = set()

        if exclude_old_runs:
            if (run_name is None and exclude_combs is None) or (run_name is not None and exclude_combs is not None):
                raise IOError("If you want to exclude old runs you have to define the exactly one of the following"
                              "two parameters: run_name, exclude_combs (as set of dicts)")

            if run_name is not None:
                try:
                    self._request_old_runs(run_name)
                except Exception as e:
                    pass
            else:
                if not isinstance(exclude_combs, set):
                    raise IOError("If specified, exclude_combs needs to be a set of dicts")
                self._excluded_combs = exclude_combs

    def _request_old_runs(self, run_name):
        pass
        # try:
        #     coll = LS2MongoDB("MA").get_database(user="write").get_collection("runs")
        #
        #     for run in coll.find({"run_name": run_name}, {"config": 1}):
        #         comb = {k: v for k, v in run["config"].items() if k in self.combinations.keys()}
        #         self._excluded_combs.add(str(comb))
        #
        # except Exception:
        #     print("Could not connect to MongoDB")

    def random_grid_search(self):
        combs = [[{k: t} for t in v] for k, v in self.combinations.items()]

        persistent_config = self.base_config.copy()

        permutation_size = reduce(lambda x, y: x * y, [len(v) for v in self.combinations.values()]) - len(
            self._excluded_combs)

        old_combs = set()

        upd = dict()
        while len(old_combs) <= permutation_size:

            for i in range(len(combs)):
                c = combs[i][random.randint(0, len(combs[i]) - 1)]
                upd.update(c)

            if str(upd) in old_combs | self._excluded_combs:
                continue
            else:
                persistent_config.update(upd)
                yield persistent_config, upd

                old_combs.add(str(upd))
                persistent_config = self.base_config.copy()

