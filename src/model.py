import functools
import itertools
import math
from collections import defaultdict

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class CFG:
    n_strings = 3
    letters = [
        "🎅",  # father christmas
        "🤶",  # mother christmas
        "🦌",  # reindeer
        "🧝",  # elf
        #        "🎄",  # christmas tree
        #        "🎁",  # gift
        #        "🎀",  # ribbon
    ]
    wildcard = "🌟"  # star
    n = len(letters)
    node_depot = itertools.repeat("0", n)  # a starting dummy node
    depot = 0
    inf = 999  # number to represent "infinite distance", could try sys.maxsize for hard constraint

    # 検索のソリューション制限
    n_solutions = None
    # 時間制限
    n_minutes = 2
    # 指定した距離以上のノードを削除する
    lim_distance = None  # None or integer


class Model:
    def __init__(self, cfg):
        # parameters
        self.n = cfg.n
        self.letters = cfg.letters
        self.n_strings = cfg.n_strings
        self.node_depot = cfg.node_depot
        self.depot = cfg.depot
        self.inf = cfg.inf

        # node
        self.locations = self._make_nodes()

        # IndexManager: solver内部で使われているIndexを意識することなく、
        #               自分が用意したNodeで扱えるようにするためのもの
        # 引数: 地点数、乗り物の数、デポのnode(各ルートの一番最初/最後にいるべき場所)
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.locations), self.n_strings, self.depot
        )

        # Routing Model: IndexManagerを引数にとる
        self.routing = pywrapcp.RoutingModel(self.manager)

        # 距離計算のcallbackをRoutingModelに登録
        transit_callback_index = self.routing.RegisterTransitCallback(
            self._distance_callback
        )
        # callbackをコストとしてセット(ここでセットしたものを最小にしようとする)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # デフォルトでは、オプティマイザーはソリューションを
        # 検索するときにすべての文字列の全長のみを考慮します。
        # 最長の文字列の最小の長さを見つけるという目的を実行するために、
        # 追加のディメンションを作成します。

        # ルートに沿って累積されたモデル数量をディメンション化します。
        # ここにディメンションを追加して、
        # 各ルートの長さ（つまり、映画のスケジュール）を累積し、
        # 最大の長さと最小の長さの差に比例するコストを設定します。
        # このコストを追加すると、オプティマイザは最大の長さを
        # できるだけ短くするようになります。
        dimension_name = "Length"
        self.routing.AddDimension(
            transit_callback_index,
            0,  # ゆるみの設定
            10 ** 16,  # max length per route; set to some large-enough number
            True,  # start with total length of 0
            dimension_name,
        )
        # TODO: length_dimensionはどこで使ってる？
        length_dimension = self.routing.GetDimensionOrDie(dimension_name)
        length_dimension.SetGlobalSpanCostCoefficient(
            100
        )  # total cost += 100 * (max_length - min_length)

        # Set constraint that each vehicle must have all permutations beginning with 🎅🤶
        n_all = math.factorial(self.n - 2)
        for vehicle in range(self.n_strings):
            for node in range(n_all):
                self.routing.SetAllowedVehiclesForIndex(
                    [vehicle], self.manager.NodeToIndex(1 + vehicle * n_all + node)
                )

        # エッジ削除
        if cfg.lim_distance is not None:
            self._remove_forbidden_edges(
                lambda p, q: self._distance(p, q) >= cfg.lim_distance
            )

        # 検索の戦略を変更
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self.search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        self.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )

        # 終了条件を設定
        if cfg.n_solutions is not None:
            self.search_parameters.solution_limit = cfg.n_solutions
        if cfg.n_minutes is not None:
            self.search_parameters.time_limit.seconds = cfg.n_minutes * 60

    def run(self):
        """
        Solverを実行
        """
        self.solution = self.routing.SolveWithParameters(self.search_parameters)

    def make_sub(self):
        """
        提出用に出力
        """
        submission = None
        if self.solution:
            routes = self._get_routes()
            words = self._get_schedules(routes)
            submission = pd.Series(words, name="schedule")
        return submission

    def _make_nodes(self):
        """
        順列を生成
        node -> 順列

        Describes
        ----------
        🎅🤶で始まる順列がすべてのスケジュールに含まれている
        必要があるという条件を適用するために、
        スケジュールごとにそのようなノードのセットを作成します
        """
        perms = list(itertools.permutations(self.letters, self.n))
        all_ = perms[: math.factorial(self.n - 2)]  # permutations beginning with 🎅🤶
        some = perms[math.factorial(self.n - 2) :]  # everything else
        return [self.node_depot] + (all_ * self.n_strings) + some

    def _distance(self, p, q):
        """
        距離を計算

        Descrives
        ---------
        ノード間の最大距離を無限大（または実際には大きな数）に
        設定することにより、検索スペースを削減します。
        この変更は、ノード🎅🤶🦌🧝🎄🎁🎀からノード🧝🎄🎁🎀🎅🤶🦌に
        移動することを決して考慮しないことを意味します。
        """
        if p == self.node_depot or q == self.node_depot:
            return 0
        for num in range(self.n):  # never choose maximum distance nodes (the N+1 case)
            if p[num:] == q[: self.n - num]:
                return self.n
        return self.inf  # max distance N becomes distance infinity

    def _distance_callback(self, from_index, to_index):
        """
        ノード間の距離を測定する方法を定義

        Describes
        ---------
        距離関数の変更は、問題の制約を変更したり、
        検索スペースを変更したりする1つの方法です。
        まず、depot nodeの「ダミー」を作成するために、
        すべてのノードまでの距離を0と定義します。
        """
        from_node = self.manager.IndexToNode(from_index)
        from_perm = self.locations[from_node]
        to_node = self.manager.IndexToNode(to_index)
        to_perm = self.locations[to_node]
        return self._distance(
            from_perm,
            to_perm,
        )

    def _remove_forbidden_edges(self, forbidden_fn):
        """
        forbidden_fnの条件に合致するエッジを削除する
        """
        n_removed = 0
        n_nodes = len(self.locations)

        for from_index, to_index in itertools.product(range(n_nodes), range(n_nodes)):
            from_node = self.manager.IndexToNode(from_index)
            from_perm = self.locations[from_node]
            to_node = self.manager.IndexToNode(to_index)
            to_perm = self.locations[to_node]
            if self.routing.NextVar(from_index).Contains(to_index) and forbidden_fn(
                from_perm, to_perm
            ):
                n_removed += 1
                self.routing.NextVar(from_index).RemoveValue(to_index)
        print(f"Removed {n_removed} edges.")

    def _get_routes(self):
        """
        solutionから3つのルートを生成
        """
        routes = defaultdict(list)
        for vehicle_id in range(self.n_strings):
            index = self.routing.Start(vehicle_id)
            while not self.routing.IsEnd(index):
                idx_node = self.manager.IndexToNode(index)
                if idx_node != self.depot:
                    routes[vehicle_id].append(idx_node)
                index = self.solution.Value(self.routing.NextVar(index))
        return routes

    def _get_schedules(self, routes: dict) -> list:
        """
        ルートのリストを超置換形式に変換（圧縮）
        """
        words = [
            self._route_to_schedule(routes[vehicle_id], self.locations)
            for vehicle_id in range(self.n_strings)
        ]
        return words

    def _route_to_schedule(self, route, nodes):
        def overlap(a, b):
            return max(i for i in range(len(b) + 1) if a.endswith(b[:i]))

        def squeeze(ws):
            return functools.reduce(lambda a, b: a + b[overlap(a, b) :], ws)

        return squeeze(["".join(nodes[i]) for i in route])


if __name__ == "__main__":
    model = Model(CFG)
    model.run()
    sub = model.make_sub()
    if sub is not None:
        print(sub)
        print(sub.apply(len).rename("Length"))
